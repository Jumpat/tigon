import os
import copy
import json
import torch
import torch.nn as nn
from functools import partial
from contextlib import nullcontext

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .utils import *
from .base import Trainer
from ..utils.general_utils import *
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils


class BasicTrainer(Trainer):
    """
    Trainer for basic training loop.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - FP16 mode: {self.fp16_mode}')
        return '\n'.join(lines)
            
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        self.use_deepspeed = kwargs.get('use_deepspeed', False)
        self.deepspeed_config = kwargs.get('deepspeed_config', 'ds_config.json')
        if self.fp16_mode == 'deepspeed':
            self.use_deepspeed = True

        if self.world_size > 1 and not self.use_deepspeed:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            # DeepSpeed or single-machine mode: do not wrap with DDP, keep the original model, and ensure a no_sync stub exists
            self.training_models = self.models
            for _m in self.training_models.values():
                if not hasattr(_m, 'no_sync'):
                    _m.no_sync = nullcontext

        # Build master params
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
        , [])
        if self.fp16_mode == 'amp':
            self.master_params = self.model_params
            self.scaler = torch.GradScaler()
        elif self.fp16_mode == 'deepspeed':
            # DeepSpeed bf16 path: do not use an external GradScaler or inflat_all
            self.master_params = self.model_params
            self.scaler = None
        elif self.fp16_mode == 'inflat_all':
            self.master_params = make_master_params(self.model_params)
            self.fp16_scale_growth = self.fp16_scale_growth
            self.log_scale = 20.0
        elif self.fp16_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'FP16 mode {self.fp16_mode} is not implemented.')

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

        # Initialize DeepSpeed (ZeRO-2 + bf16; enabled when fp16_mode == "deepspeed")
        if self.use_deepspeed:
            import deepspeed

            class _DSContainer(nn.Module):
                def __init__(self, modules_dict):
                    super().__init__()
                    # Reference existing submodules directly without copying
                    self.modules = nn.ModuleDict(modules_dict)
                def forward(self, *args, **kwargs):
                    raise NotImplementedError

            ds_cfg = self._resolve_deepspeed_config(self.deepspeed_config if self.fp16_mode == 'deepspeed' else self.deepspeed_config)
            container = _DSContainer(self.training_models)
            self.ds_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=container,
                model_parameters=self.model_params,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler if self.lr_scheduler_config is not None else None,
                config=ds_cfg
            )

    def _resolve_deepspeed_config(self, cfg_src):
        """Load/patch DS config for bf16 ZeRO-2 and align micro-batch/accum steps.
        Returns a dict suitable for deepspeed.initialize.
        """
        # Load from path or use dict directly
        if isinstance(cfg_src, str):
            try:
                with open(cfg_src, 'r') as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        elif isinstance(cfg_src, dict):
            cfg = copy.deepcopy(cfg_src)
        else:
            cfg = {}
        # Ensure dict exists
        if 'zero_optimization' not in cfg:
            cfg['zero_optimization'] = {"stage": 2}
        # Force ZeRO-2
        cfg['zero_optimization']['stage'] = 2
        # Bucket sizes default if missing (elements, not bytes)
        cfg['zero_optimization'].setdefault('reduce_scatter', True)
        cfg['zero_optimization'].setdefault('overlap_comm', True)
        cfg['zero_optimization'].setdefault('contiguous_gradients', True)
        cfg['zero_optimization'].setdefault('allgather_bucket_size', 500000000)
        cfg['zero_optimization'].setdefault('reduce_bucket_size', 500000000)
        cfg['zero_optimization'].setdefault('ignore_unused_parameters', True)
        # Precision: bf16 on, fp16 off
        cfg['bf16'] = cfg.get('bf16', {})
        cfg['bf16']['enabled'] = True
        cfg['fp16'] = cfg.get('fp16', {})
        cfg['fp16']['enabled'] = False
        # Batch sizing: per-micro and accumulation align to Trainer
        micro = max(1, int(self.batch_size_per_gpu // self.batch_split))
        cfg['train_micro_batch_size_per_gpu'] = int(cfg.get('train_micro_batch_size_per_gpu', micro) or micro)
        cfg['gradient_accumulation_steps'] = int(cfg.get('gradient_accumulation_steps', self.batch_split) or self.batch_split)
        return cfg

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        NOTE: save DETACHED tensors (not Parameter objects) to avoid pickling
        issues with distributed ProcessGroup references attached by ZeRO hooks.
        """
        if self.fp16_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)

        # Use each model's current state_dict as the base
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}

        # Map flattened master parameters back to each model's named parameters, but save them as detached tensors
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()],
            []
        )
        for i, (model_name, param_name) in enumerate(master_params_names):
            p = master_params[i]
            t = p.detach() if torch.is_tensor(p) else torch.as_tensor(p)  # Key point: use a tensor that has already been detached
            state_dicts[model_name][param_name] = t
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        """
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        if self.fp16_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
            
        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')), map_location=self.device, weights_only=True)
            model_ckpts[name] = model_ckpt
            model.load_state_dict(model_ckpt)
            if self.fp16_mode == 'inflat_all':
                model.convert_to_fp16()
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt'), map_location=self.device, weights_only=True)
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        
        misc_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')), map_location=torch.device('cpu'), weights_only=False)
        if not getattr(self, 'use_deepspeed', False):
            self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.fp16_mode == 'amp':
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.fp16_mode == 'inflat_all':
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            if not getattr(self, 'use_deepspeed', False):
                self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        # Load DeepSpeed sharded optimizer/scheduler state
        if getattr(self, 'use_deepspeed', False):
            if self.world_size > 1:
                dist.barrier()
            ds_dir = os.path.join(load_dir, 'ckpts', f'ds_step{step:07d}')
            try:
                # Let DeepSpeed restore the optimizer and scheduler; model weights have already been loaded above
                # (If you want DeepSpeed to load model weights entirely, change load_module_only=True to False)
                self.ds_engine.load_checkpoint(
                    ds_dir, tag=str(step),
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                    load_module_only=False  # Here False means not restricting the load to model weights only; DeepSpeed may also load model weights, which does not conflict with the load above
                )
            except Exception as e:
                if self.is_master:
                    print(f'\nWarning: DeepSpeed checkpoint not found or failed to load from {ds_dir}: {e}')

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """

        if self.is_master:
            print(f'\nSaving checkpoint at step {self.step}...', end='')
            
            model_ckpts = self._master_params_to_state_dicts(self.master_params)
            for name, model_ckpt in model_ckpts.items():
                torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))
            
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
                for name, ema_ckpt in ema_ckpts.items():
                    torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt'))

            misc_ckpt = {
                'step': self.step,
                'data_sampler': self.data_sampler.state_dict(),
            }
            if not getattr(self, 'use_deepspeed', False):
                misc_ckpt['optimizer'] = self.optimizer.state_dict()
            if self.fp16_mode == 'amp':
                misc_ckpt['scaler'] = self.scaler.state_dict()
            elif self.fp16_mode == 'inflat_all':
                misc_ckpt['log_scale'] = self.log_scale
            if self.lr_scheduler_config is not None:
                misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
            if self.elastic_controller_config is not None:
                misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
            if self.grad_clip is not None and not isinstance(self.grad_clip, float):
                misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
            torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))

        # 2) DeepSpeed sharded checkpoints: all ranks must participate
        if getattr(self, 'use_deepspeed', False):
            if self.world_size > 1:
                dist.barrier()  # Wait for the master rank to finish writing regular files, then save DeepSpeed checkpoints together
            ds_dir = os.path.join(self.output_dir, 'ckpts', f'ds_step{self.step:07d}')
            tag = str(self.step)
            self.ds_engine.save_checkpoint(ds_dir, tag=tag)
            if self.world_size > 1:
                dist.barrier()

        if self.is_master:
            print(' Done.')

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                model_ckpt = torch.load(read_file_dist(finetune_ckpt[name]), map_location=self.device, weights_only=True)
                for k, v in model_ckpt.items():
                    if model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                        model_ckpt[k] = model_state_dict[k]
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)
                if self.fp16_mode == 'inflat_all':
                    model.convert_to_fp16()
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    def run_step(self, data_list):
        """
        Run a training step.
        """
        step_log = {'loss': {}, 'status': {}}
        if self.fp16_mode == 'amp':
            amp_context = partial(torch.autocast, device_type='cuda')
        elif self.fp16_mode == 'deepspeed':
            # DeepSpeed bf16：Use bfloat16 autocast for the forward pass; backward and accumulation are handled by DeepSpeed
            amp_context = partial(torch.autocast, device_type='cuda', dtype=torch.bfloat16)
        else:
            amp_context = nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        if getattr(self, 'use_deepspeed', False):
            # Prefer zero_grad from DeepSpeed or the optimizer to stay consistent with ZeRO behavior
            if hasattr(self, 'ds_engine'):
                self.ds_engine.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)
        else:
            zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = [self.training_models[name].no_sync for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss['loss'] / len(data_list)
                ## backward
                if self.fp16_mode == 'amp':
                    self.scaler.scale(l).backward()
                elif self.fp16_mode == 'deepspeed':
                    # DeepSpeed handles backward, accumulation, and overflow management
                    self.ds_engine.backward(l)
                elif self.fp16_mode == 'inflat_all':
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        ## gradient clip
        if self.grad_clip is not None:
            if self.fp16_mode == 'amp':
                self.scaler.unscale_(self.optimizer)  # Note: under DeepSpeed, self.optimizer has already been wrapped
            elif self.fp16_mode == 'deepspeed':
                # DeepSpeed provides global gradient clipping
                pass
            elif self.fp16_mode == 'inflat_all':
                model_grads_to_master_grads(self.model_params, self.master_params)
                self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
            if isinstance(self.grad_clip, float):
                if getattr(self, 'use_deepspeed', False) and hasattr(self, 'ds_engine'):
                    grad_norm = self.ds_engine.clip_grad_norm(self.grad_clip)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]['grad_norm'] = grad_norm.item()
        ## step
        if self.fp16_mode == 'amp':
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.fp16_mode == 'deepspeed':
            prev_scale = 1.0
            # Let DeepSpeed advance micro-steps and accumulation; optimizer.step/scheduler.step happens only at the accumulation boundary
            self.ds_engine.step()
        elif self.fp16_mode == 'inflat_all':
            prev_scale = 2 ** self.log_scale
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                    self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
                self.optimizer.step()
                master_params_to_model_params(self.model_params, self.master_params)
                self.log_scale += self.fp16_scale_growth
            else:
                self.log_scale -= 1
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m') 
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            if self.fp16_mode == 'deepspeed':
                # DeepSpeed advances the scheduler at the accumulation boundary; only record lr here
                statuses[-1]['lr'] = self.optimizer.param_groups[0]['lr']
            else:
                statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
                self.lr_scheduler.step()

        # Logs
        step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    param_norms[name] = param.norm().item()
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        param_grads[name] = param.grad.norm().item() / prev_scale
            step_log['param_norms'] = param_norms
            step_log['param_grads'] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return step_log
