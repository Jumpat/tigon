from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock, ModulatedTransformerCrossBlockDualAttn
from ..modules.spatial import patchify, unpatchify
import math

def _naive_sdpa(q, k):
    """
    Naive implementation of scaled dot product attention.
    """
    q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)

    attn_weight = attn_weight.mean(dim = 1)

    return attn_weight

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

import warnings
from collections import OrderedDict
from safetensors.torch import load_file as safetensors_load

class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        pretrained_path: str = None,
        freeze_pretrained: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()

        if pretrained_path:
            self.safe_load_checkpoint(pretrained_path, freeze_pretrained)

        if use_fp16:
            self.convert_to_fp16()

    def safe_load_checkpoint(self, checkpoint_path, freeze_pretrained=True):
        """Safely load a checkpoint, skipping shape-mismatched parameters and freezing matching ones
        
        Args:
            checkpoint_path: checkpoint file path
            freeze_pretrained: whether to freeze pretrained parameters (default: False)
        """
        # 1. Load the checkpoint
        checkpoint = {}
        try:
            if '.safetensors' in checkpoint_path:
                checkpoint = safetensors_load(checkpoint_path)
            else:  # .pt/.pth or other formats
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
 
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e
            
        model_state = self.state_dict()
        
        # 2. Filter matching parameters
        matched_state = OrderedDict()
        matched_names = []  # Record matched parameter names
        skipped_names = []  # Record skipped parameter names
        
        for name, param in checkpoint.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    matched_state[name] = param
                    matched_names.append(name)
                else:
                    warnings.warn(f"Parameter '{name}' shape mismatch: "
                                f"checkpoint {tuple(param.shape)} vs model {tuple(model_state[name].shape)}")
                    skipped_names.append(name)
            else:
                skipped_names.append(name)
    
        # 3. Load matched parameters
        try:
            self.load_state_dict(matched_state, strict=False)
        except RuntimeError as e:
            warnings.warn(f"Partial parameter loading failed: {str(e)}")
        
        # 4. Freeze matched parameters
        if freeze_pretrained:
            for name, param in self.named_parameters():
                if name in matched_names:
                    param.requires_grad = False
                    print(f"Freezing parameter: {name}")
            
        # 5. Print the detailed report
        print(f"Successfully loaded parameters: {len(matched_names)}/{len(checkpoint)}")
        print(f"Frozen parameters: {len(matched_names) if freeze_pretrained else 0}")
        
        if skipped_names:
            print("\nSkipped parameters:")
            for name in skipped_names:
                print(f"  - {name}")
        

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h

class SparseStructureFlowMixCondModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        cond_channels_txt: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        pretrained_path: str = None,
        text_pretrained_path: str = None,
        freeze_pretrained: bool = True,
        use_bridge: bool = True,
        mix_type: str = 'mean',
        txt_weight: float = 0.5,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.cond_channels_txt = cond_channels_txt

        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        self.text_t_embedder = TimestepEmbedder(model_channels)

        self.use_bridge = use_bridge
        self.txt_weight = txt_weight
        
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            text_pos_emb = pos_embedder(coords)
            
            self.register_buffer("pos_emb", pos_emb)
            self.register_buffer("text_pos_emb", text_pos_emb)
            

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        self.text_input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.text_blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels_txt,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.mix_type = mix_type
        if self.mix_type == 'adaptive' or self.mix_type == 'nonlinear':
            coe = 4 if self.mix_type == 'adaptive' else 1
            self.mix_alpha_block = ModulatedTransformerCrossBlockDualAttn(
                model_channels * coe,
                cond_channels,
                cond_channels_txt,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio // coe,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )

            mix_out_channels = out_channels if self.mix_type == 'nonlinear' else 1
            if self.mix_type == 'adaptive' :
                self.mix_alpha_out_layer = nn.Sequential(
                    nn.Linear(model_channels * coe, mix_out_channels * patch_size**3),
                    nn.Sigmoid()
                )
            self.mix_alpha_t_embedder = TimestepEmbedder(model_channels * coe)

        if self.use_bridge:
            self.h_zero_conv_v_t = nn.ModuleList([nn.Linear(model_channels, model_channels) for i in range(num_blocks)])
            self.h_zero_conv_t_v = nn.ModuleList([nn.Linear(model_channels, model_channels) for i in range(num_blocks)])
        
        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)
        self.text_out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()

        if pretrained_path:
            self.safe_load_checkpoint(pretrained_path, freeze_pretrained)

        if text_pretrained_path:
            self.safe_load_checkpoint(text_pretrained_path, freeze_pretrained)

        if use_fp16:
            self.convert_to_fp16()

    def safe_load_checkpoint(self, checkpoint_path, freeze_pretrained=True):
        """Safely load a checkpoint, skipping shape-mismatched parameters and freezing matching ones
        
        Args:
            checkpoint_path: checkpoint file path
            freeze_pretrained: whether to freeze pretrained parameters (default: False)
        """
        # 1. Load the checkpoint
        checkpoint = {}
        try:
            if '.safetensors' in checkpoint_path:
                checkpoint = safetensors_load(checkpoint_path)
            else:  # .pt/.pth or other formats
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']  # Extract model parameters
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e
            
        model_state = self.state_dict()
        
        # 2. Filter matching parameters
        matched_state = OrderedDict()
        matched_names = []  # Record matched parameter names
        skipped_names = []  # Record skipped parameter names
        
        for name, param in checkpoint.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    matched_state[name] = param
                    matched_names.append(name)
                else:
                    warnings.warn(f"Parameter '{name}' shape mismatch: "
                                f"Checkpoint {tuple(param.shape)} vs Model {tuple(model_state[name].shape)}")
                    
                    skipped_names.append(name)
            else:
                skipped_names.append(name)
    
        # 3. Load matched parameters
        try:
            self.load_state_dict(matched_state, strict=False)
        except RuntimeError as e:
            warnings.warn(f"Failed to load some parameters: {str(e)}")
        
        # 4. Freeze matched parameters
        if freeze_pretrained:
            for name, param in self.named_parameters():
                if name in matched_names:
                    param.requires_grad = False
                    print(f"Freezing parameter: {name}")
            
        # 5. Print the detailed report
        print(f"Successfully loaded parameters: {len(matched_names)}/{len(checkpoint)}")
        print(f"Frozen parameters: {len(matched_names) if freeze_pretrained else 0}")
        
        if skipped_names:
            print("\nSkipped parameters:")
            for name in skipped_names:
                print(f"  - {name}")
        

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)
        
        self.text_blocks.apply(convert_module_to_f16)

        if self.mix_type == 'adaptive' or self.mix_type == 'nonlinear':
            self.mix_alpha_block.apply(convert_module_to_f16)
        
        if self.use_bridge:
            self.h_zero_conv_v_t.apply(convert_module_to_f16)
            self.h_zero_conv_t_v.apply(convert_module_to_f16)
        
    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
        self.text_blocks.apply(convert_module_to_f32)

        if self.mix_type == 'adaptive' or self.mix_type == 'nonlinear':
            self.mix_alpha_block.apply(convert_module_to_f32)
        
        if self.use_bridge:
            self.h_zero_conv_v_t.apply(convert_module_to_f32)
            self.h_zero_conv_t_v.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.text_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.text_t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for block in self.text_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.mix_type == 'adaptive' or self.mix_type == 'nonlinear':
            nn.init.constant_(self.mix_alpha_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.mix_alpha_block.adaLN_modulation[-1].bias, 0)
            nn.init.normal_(self.mix_alpha_t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.mix_alpha_t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

        nn.init.constant_(self.text_out_layer.weight, 0)
        nn.init.constant_(self.text_out_layer.bias, 0)

        if self.mix_type == 'adaptive':
            nn.init.constant_(self.mix_alpha_out_layer[0].weight, 0)
            nn.init.constant_(self.mix_alpha_out_layer[0].bias, 0)

        if self.use_bridge:
            for layer, layer2 in zip(self.h_zero_conv_v_t, self.h_zero_conv_t_v):
                # Initialize weights to zero
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
    
                nn.init.constant_(layer2.weight, 0)
                nn.init.constant_(layer2.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"


        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        vh = self.input_layer(h)
        txt_h = self.text_input_layer(h)
        
        h = vh + self.pos_emb[None]

        txt_h =txt_h + self.text_pos_emb[None]
        
        t_emb = self.t_embedder(t)
        
        text_t_emb = self.text_t_embedder(t)

        txt_cond = cond.get("txt_cond", None)
        
        image_cond = cond.get("image_cond", None)
        cond = image_cond

        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        text_t_emb = text_t_emb.type(self.dtype)
        
        h = h.type(self.dtype)

        txt_h = txt_h.type(self.dtype)

        initial_h = h
        initial_txt_h = txt_h
        
        cond = cond.type(self.dtype)

        txt_cond = txt_cond.type(self.dtype)

        if self.mix_type == 'attn':
            image_attns = []
            text_attns = []
        for i in range(len(self.blocks)):
            if self.mix_type != 'attn':
                h = self.blocks[i](h, t_emb, cond)
                txt_h = self.text_blocks[i](txt_h, text_t_emb, txt_cond)
            else:
                h, iq, ik = self.blocks[i](h, t_emb, cond, True)
                txt_h, tq, tk = self.text_blocks[i](txt_h, text_t_emb, txt_cond, True)


                with torch.no_grad():
                    # N L
                    image_attns.append(_naive_sdpa(ik.detach(), iq.detach()).mean(dim = 1))
                    text_attns.append(_naive_sdpa(tk.detach(), tq.detach()).mean(dim = 1))

                    
                
            if self.use_bridge:
                vt_h = self.h_zero_conv_v_t[i](h) 
                tv_h = self.h_zero_conv_t_v[i](txt_h)
            
                h = tv_h + h
                txt_h = vt_h + txt_h

        h = h.type(x.dtype)
        inner_h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(inner_h)

        txt_h = txt_h.type(x.dtype)
        inner_txt_h = F.layer_norm(txt_h, txt_h.shape[-1:])
        txt_h = self.text_out_layer(inner_txt_h)


        if self.mix_type == 'attn':
            with torch.no_grad():
                image_attns = torch.stack(image_attns, dim = 1).mean(1)
                text_attns = torch.stack(text_attns, dim = 1).mean(1)
    
                weight = image_attns / (image_attns + text_attns)
                weight = weight.unsqueeze(-1)
                t_w = 1-weight
                
        elif self.mix_type == 'mean':
            t_w = self.txt_weight
            weight = 1-t_w
            
        elif self.mix_type == 'confidence_mean':
            h_norm = torch.norm(h, p = 2, dim = -1)
            txt_h_norm = torch.norm(txt_h, p = 2, dim = -1)
            
            weight = h_norm / (h_norm + txt_h_norm)
            weight = weight.unsqueeze(-1)
            weight[weight <= 0.5] = 0.5
            t_w = 1-weight
        elif self.mix_type == "img_plus_text_orth":
            eps = 1e-6
            # h, txt_h: (B, D)
            # Remove the component of text aligned with image and keep only the complementary information
            old_h_norm = torch.norm(h, p=2, dim=-1, keepdim=True) + eps
            
            dot = (txt_h * h).sum(dim=-1, keepdim=True)              # (B, 1)
            h2  = (h * h).sum(dim=-1, keepdim=True) + eps            # (B, 1)
            txt_orth = txt_h - dot / h2 * h                          # (B, D)
        
            gamma = 1.0   # text influence strength
            weight = 1.0
            t_w = gamma
        elif self.mix_type == 'adaptive':
            m_t_emb = self.mix_alpha_t_embedder(t)
            m_t_emb = m_t_emb.type(self.dtype)
            mix_alpha_input = torch.cat([initial_h, initial_txt_h, inner_h, inner_txt_h], dim = -1)
            mix_alpha_input = mix_alpha_input.type(self.dtype)
            weight = self.mix_alpha_block(mix_alpha_input.detach(), m_t_emb, cond, txt_cond)
            weight = weight.type(h.dtype)
            weight = F.layer_norm(weight, weight.shape[-1:])
            weight = self.mix_alpha_out_layer(weight)
            t_w = 1-weight

        elif self.mix_type == 'nonlinear':
            m_t_emb = self.mix_alpha_t_embedder(t)
            m_t_emb = m_t_emb.type(self.dtype)
            mix_input = torch.cat([inner_h, inner_txt_h], dim = 1)
            num_tokens = inner_h.shape[1]
            mix_input = mix_input.type(self.dtype)
            mix_output = self.mix_alpha_block(mix_input, m_t_emb, cond, txt_cond)
            mix_output = mix_output.type(h.dtype)
            mix_output = mix_output[:,:num_tokens,:] + mix_output[:,num_tokens:,:]
            mix_output = F.layer_norm(mix_output, mix_output.shape[-1:])
            mix_output = (self.out_layer(mix_output) + self.text_out_layer(mix_output)) / 2

        if self.mix_type == 'nonlinear':
            h = mix_output
        else:
            h = weight * h + t_w * txt_h
        if self.mix_type == "img_plus_text_orth":

            
            h_norm = torch.norm(h, p=2, dim=-1, keepdim=True)
    
            h = h * (old_h_norm / (h_norm + eps)).detach()

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h

class SparseStructureFlowAlternateModel(SparseStructureFlowMixCondModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alternate = False

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
    
        
        if self.alternate:

            vh = self.input_layer(h)
            
            h = vh + self.pos_emb[None]
    
            t_emb = self.t_embedder(t)
            
            image_cond = cond.get("image_cond", None)
            cond = image_cond
    
            if self.share_mod:
                t_emb = self.adaLN_modulation(t_emb)
            t_emb = t_emb.type(self.dtype)
            
            h = h.type(self.dtype)
    
            cond = cond.type(self.dtype)
    
            for i in range(len(self.blocks)):
                h = self.blocks[i](h, t_emb, cond)

            h = h.type(x.dtype)
            h = F.layer_norm(h, h.shape[-1:])
            h = self.out_layer(h)

        else:
    
            
            txt_h = self.text_input_layer(h)

            txt_h =txt_h + self.text_pos_emb[None]
            
            text_t_emb = self.text_t_embedder(t)
    
            txt_cond = cond.get("txt_cond", None)

            cond = txt_cond
    
            text_t_emb = text_t_emb.type(self.dtype)
            
            txt_h = txt_h.type(self.dtype)
            
            cond = cond.type(self.dtype)
    
            for i in range(len(self.blocks)):
                txt_h = self.text_blocks[i](txt_h, text_t_emb, cond)
    
            h = txt_h.type(x.dtype)
            h = F.layer_norm(h, h.shape[-1:])
            h = self.text_out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()
        # self.alternate = not self.alternate

        return h