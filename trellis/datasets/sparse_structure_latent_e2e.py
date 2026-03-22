import json
import os
from typing import *
import numpy as np
import torch
from tqdm import tqdm
import logging
import utils3d.torch
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin, InterleaveConditionedMixin
from ..modules.sparse.basic import SparseTensor
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.dist_utils import read_file_dist
from ..utils.data_utils import load_balanced_group_indices
from ..utils.loss_utils import psnr, ssim, lpips

class SLatUnifyVisMixin:
    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.slat_dec = None
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        
    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(read_file_dist(ckpt_path), map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        self.slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        del self.slat_dec
        self.slat_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4, delete=True):
        self._loading_slat_dec()
        reps = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            reps.append(self.slat_dec(z[i:i+batch_size]))
        reps = sum(reps, [])
        if delete:
            self._delete_slat_dec()
        return reps

    def calculate_metrics(self, samples):
        x_0 = samples['sample']['value']
        intrinsics = samples['intrinsics']['value']
        extrinsics = samples['extrinsics']['value']
        gt_images = samples['images']['value'].cuda()
        reps = self.decode_latent(x_0.cuda())
        renderer = get_renderer(reps[0])
        renders = []
        renders_gt = []
        gt = []
        all_psnr = []
        all_psnr_gt = []
        all_ssim = []
        all_ssim_gt = []
        all_lpips = []
        all_lpips_gt = []
        
        image_gts = []
        image_render = []
        image_render_gt = []
        
        
        for representation, extrinsic, \
            intrinsic, gt_image in zip(reps, extrinsics, intrinsics, gt_images):
            res = renderer.render(representation, extrinsic, intrinsic)

            renders.append(res['color'])
            loss_image = res['color'].unsqueeze(0)#.permute(0,3,1,2)
            image_gt = gt_image.unsqueeze(0)#.permute(0,3,1,2)
            psnr_loss = psnr(loss_image, image_gt)
            lpips_loss = lpips(loss_image,image_gt)
            ssim_loss = ssim(loss_image, image_gt)
            all_psnr.append(psnr_loss)
            
            all_ssim.append(ssim_loss)
            
            all_lpips.append(lpips_loss)
            
            image_render.append(loss_image)
            image_gts.append(image_gt)
        return_dict = {
            # "image_render":{'value':torch.cat(image_render,dim=0),'type':"image"},
            # "image_gt":{'value':torch.cat(image_gts,dim=0),'type':"image"},
            # "image_render_gt":{"value":torch.cat(image_render_gt,dim=0),"type":"image"},
            "all_psnr":{"value":torch.tensor([all_psnr]),"type":"metrics"},
            # "all_psnr_gt":{"value":torch.tensor([all_psnr_gt]),"type":"metrics"},
            "all_lpips":{"value":torch.tensor([all_lpips]),"type":"metrics"},
            # "all_lpips_gt":{"value":torch.tensor([all_lpips_gt]),"type":"metrics"},
            "all_ssim":{"value":torch.tensor([all_ssim]),"type":"metrics"},
            # "all_ssim_gt":{"value":torch.tensor([all_ssim_gt]),"type":"metrics"}
        }
        return return_dict
            
            
    @torch.no_grad()
    def visualize_sample(self, x_0):
        x_0 = x_0 if isinstance(x_0, SparseTensor) or isinstance(x_0,torch.Tensor) else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
        images = torch.stack(images)
            
        return images
    
class SLatUnify(SLatUnifyVisMixin, StandardDatasetBase):
    """
    structured latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        # self.loads = [self.metadata.loc[sha256, 'num_voxels'] for _, sha256 in tqdm(self.instances)]
        all_sha256 = [sha256 for _, sha256 in tqdm(self.instances)]

        # Use isin for batched lookups
        self.loads = self.metadata.loc[all_sha256, 'num_voxels'].tolist()
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean'])#.reshape(1, -1)
            self.std = torch.tensor(self.normalization['std'])#.reshape(1, -1)
            self.mean = self.mean.view(self.mean.shape[0],1,1,1).repeat(1,16,16,16)
            self.std = self.std.view(self.std.shape[0],1,1,1).repeat(1,16,16,16)

      
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[~metadata[f'latent_{self.latent_model}'].isna() & metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))

        feats = torch.tensor(data['feats']).float().squeeze(0)
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        return {
            # 'coords': coords,
            'x_0': feats,
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        group_idx = [list(range(len(batch)))]
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            feats = []
            layout = []
            start = 0
            keys = [k for k in sub_batch[0].keys() ]

            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        return packs[0]
        
class SLatTemplateUnify(SLatUnifyVisMixin, StandardDatasetBase):
    """
    structured latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        # self.loads = [self.metadata.loc[sha256, 'num_voxels'] for _, sha256 in tqdm(self.instances)]
        all_sha256 = [sha256 for _, sha256 in tqdm(self.instances)]

        # Use isin for batched lookups
        self.loads = self.metadata.loc[all_sha256, 'num_voxels'].tolist()
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
      
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[~metadata[f'latent_{self.latent_model}'].isna() & metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        return {
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        group_idx = [list(range(len(batch)))]

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            feats = []
            layout = []
            start = 0

            keys = [k for k in sub_batch[0].keys() ]

            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs
        
        
class TextConditionedSLatUnify(TextConditionedMixin, SLatUnify):
    """
    Text conditioned structured latent dataset
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        use_qwen_condition: bool = False,
        use_structured_condition: bool = False,
        **kwargs
    ):
        
        self.use_qwen = use_qwen_condition
        self.use_structured = use_structured_condition
        super().__init__(
            roots,
            latent_model=latent_model,
            max_num_voxels=max_num_voxels,
            min_aesthetic_score = min_aesthetic_score,
            normalization=normalization,
            pretrained_slat_dec = pretrained_slat_dec,
            slat_dec_path = slat_dec_path,
            slat_dec_ckpt = slat_dec_ckpt,
            **kwargs
        )

    def filter_metadata(self, metadata):
        if self.use_qwen:
            try:
                metadata['captions'] = metadata['qwen_3d_captions']
            except:
                pass
        if self.use_structured:
            try:
                metadata['captions'] = metadata['structured_qwen_captions']
            except:
                pass

        metadata, stats = super().filter_metadata(metadata)

        return metadata, stats
        


class ImageConditionedSLatUnify(ImageConditionedMixin, SLatUnify):
    """
    Image conditioned structured latent dataset
    """
    pass

class InterleaveConditionedSLatUnify(InterleaveConditionedMixin, SLatUnify):
    """
    Text-conditioned sparse structure dataset
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        use_qwen_condition: bool = False,
        use_structured_condition: bool = False,
        **kwargs
    ):
        self.use_qwen = use_qwen_condition
        self.use_structured = use_structured_condition
        super().__init__(
            roots,
            latent_model=latent_model,
            max_num_voxels=max_num_voxels,
            min_aesthetic_score = min_aesthetic_score,
            normalization=normalization,
            pretrained_slat_dec = pretrained_slat_dec,
            slat_dec_path = slat_dec_path,
            slat_dec_ckpt = slat_dec_ckpt,
            **kwargs
        )

    def filter_metadata(self, metadata):
        if self.use_qwen:
            try:
                metadata['captions'] = metadata['qwen_3d_captions']
            except:
                pass
        if self.use_structured:
            try:
                metadata['captions'] = metadata['structured_qwen_captions']
            except:
                pass

        metadata, stats = super().filter_metadata(metadata)

        return metadata, stats
        
    @staticmethod
    def collate_fn(batch):
        x = torch.stack([item['x_0'] for item in batch])
        cond = [item['cond'] for item in batch]
        return {'x_0': x, 'cond': cond}
