from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations import MeshExtractResult
from ...representations.mesh import SparseFeatures2Mesh
from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from ...modules.sparse import SparseTensor, sparse_cat
from ...utils.coords import coords_to_voxel_ply, find_features
from safetensors.torch import load_file

class SparseOccHead(nn.Module):
    def __init__(self, channels: int, out_channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            sp.SparseLinear(channels, int(channels * mlp_ratio)),
            sp.SparseGELU(approximate="tanh"),
            sp.SparseLinear(int(channels * mlp_ratio), out_channels),
        )

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return self.mlp(x)


class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h


class SLatMeshDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self.mesh_extractor = SparseFeatures2Mesh(res=self.resolution*4, use_color=self.rep_config.get('use_color', False))
        self.out_channels = self.mesh_extractor.feats_channels
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)  
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        h = super().forward(x)
        for block in self.upsample:
            h = block(h)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        return self.to_representation(h)
    

class ElasticSLatMeshDecoder(SparseTransformerElasticMixin, SLatMeshDecoder):
    """
    Slat VAE Mesh decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass



class SparseSubdivideBlock3dwithOcc(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
        self.pruning_head = SparseOccHead(self.out_channels, out_channels=1)

    def forward(self, x: sp.SparseTensor, enable_occ=False, is_training=True):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        if enable_occ and is_training==True:
            occ_prob = self.pruning_head(h)
            return h, occ_prob
        else:
            return h, None

class SLatMeshDecoderV3(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        preload_path: str= "",
        feats_latent_channels: int = 8,
        upsample_num : int = 3,
        chunk_size: int = 2
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self.mesh_extractor = SparseFeatures2Mesh(res=self.resolution*4, use_color=self.rep_config.get('use_color', False))
        self.out_channels = self.mesh_extractor.feats_channels
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3dwithOcc(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4 # 192 / 32
            ),
            SparseSubdivideBlock3dwithOcc(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)  
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret

    def pred_occ(self, x: sp.SparseTensor):
        ret = []
        for i in range(x.shape[0]):
            valid_occ = self.mesh_extractor.forward_flexicubes(x[i])
            label = torch.full((valid_occ.shape[0], 1), i, dtype=valid_occ.dtype, device=valid_occ.device)
            coords = torch.cat([label, valid_occ], dim=1)
            ret.append(coords)
        return torch.cat(ret, dim=0)
    
    def pred_occ_dilate(self, x: sp.SparseTensor):
        ret = []
        for i in range(x.shape[0]):
            valid_occ = self.mesh_extractor.forward_flexicubes(x[i])
            label = torch.full((valid_occ.shape[0], 1), i, dtype=valid_occ.dtype, device=valid_occ.device)
            coords = torch.cat([label, valid_occ], dim=1)
            ret.append(coords)
        return torch.cat(ret, dim=0)

    @torch.no_grad()
    def forward_slat(self, x: sp.SparseTensor):
        return super().forward(x)

    @torch.no_grad()
    def decode_mesh(self, x: sp.SparseTensor):
        h = super().forward(x)
        for block in self.upsample:
            h, _ = block(h, enable_occ=False, is_training=False)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        return self.to_representation(h)

    def forward(self, x: sp.SparseTensor):
        occupancy = []
        h = super().forward(x)
        for block in self.upsample:
            h, occ_prob = block(h, enable_occ=True, is_training=self.training)
            occupancy.append(occ_prob)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        if self.training:
            return self.to_representation(h), occupancy
        else:
            return self.to_representation(h), []


class SLatMeshDecoderV3U(nn.Module):
    '''
    A simple upsampler that loads V3 model and extends it to 512 resolution.
    Input: 64 resolution latent -> V3 (frozen) -> 256 resolution -> new upsample block -> 512 resolution
    '''
    def __init__(
        self,
        # Inherit all V3 parameters
        resolution: int = 64,  # V3 input resolution
        model_channels: int = 768,
        latent_channels: int = 8,
        num_blocks: int = 12,
        num_heads: Optional[int] = 12,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = True,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        feats_latent_channels: int = 32,
        upsample_num: int = 3,
        chunk_size: int = 2,
        # V3U specific parameters
        target_resolution: int = 512,  # Final output resolution
        trainable_v3: bool = False,  # Whether to train V3 parameters
        preload_path: str = "",  # Path to pretrained V3 model
        use_occ: bool = True # Use occ loss or not
    ):  
        super().__init__()

        self.resolution = resolution
        self.target_resolution = target_resolution
        self.trainable_v3 = trainable_v3
        self.use_fp16 = use_fp16
        self.rep_config = representation_config or {"use_color": True}
        self.use_occ = use_occ
        # Initialize V3 model with the same parameters
        self.v3_model = SLatMeshDecoderV3(
            resolution=resolution,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            representation_config=representation_config,
            preload_path="",  # We'll load weights manually
            feats_latent_channels=feats_latent_channels,
            upsample_num=upsample_num,
            chunk_size=chunk_size
        )
        
        # Load pretrained V3 weights if provided
        self.load_v3_weights(preload_path)
        
        # Freeze V3 parameters if not trainable
        if not self.trainable_v3:
            for param in self.v3_model.parameters():
                param.requires_grad = False
            self.v3_model.eval()
            self.v3_model.training = False

        # Calculate channel sizes once for consistency
        v3_out_channels = model_channels // 8
        final_channels = model_channels // 16

        BlockClass = SparseSubdivideBlock3dwithOcc if self.use_occ else SparseSubdivideBlock3d
        upsample_blocks = [
            BlockClass(
                channels=v3_out_channels,
                resolution=resolution * 4,
                out_channels=final_channels,
                num_groups=8
            )
        ]
        if self.target_resolution > 512:
            final_channels = model_channels // 32
            upsample_blocks.append(
                BlockClass(
                    channels=final_channels * 2,
                    resolution=resolution * 8,
                    out_channels=final_channels,
                    num_groups=8
                )
            )

        # Convert to ModuleList
        self.upsample = nn.ModuleList(upsample_blocks)

        # Initialize mesh extractor for 512 resolution
        self.mesh_extractor = SparseFeatures2Mesh(
            res=self.target_resolution,
            use_color=self.rep_config.get('use_color', True)
        )
        
        # Output layer for final features
        self.out_layer = sp.SparseLinear(
            final_channels, 
            self.mesh_extractor.feats_channels
        )
        
        # Initialize new layers
        self.initialize_new_weights()
        
        # Convert to fp16 if needed
        if use_fp16 and not self.trainable_v3:
            # Only convert new layers if V3 is frozen (V3 already in fp16)
            self.convert_new_layers_to_fp16()
    
    def load_v3_weights(self, path: str):
        """Load pretrained V3 model weights from SafeTensors format."""
        if path is not None and path != '':
            state_dict = load_file(path)
            try:
                # Try to load directly to V3 model
                self.v3_model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded V3 weights from {path}")
            except:
                raise NotImplementedError
        
    def initialize_new_weights(self):
        """Initialize weights for new layers only."""
        # Initialize output layer with zeros (following V3 pattern)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
    
    def convert_new_layers_to_fp16(self):
        """Convert only new layers to fp16."""
        self.upsample.apply(convert_module_to_f16)
        # self.out_layer = self.out_layer.half()
        # self.mesh_extractor = self.mesh_extractor.half()
    
    def convert_to_fp16(self):
        """Convert all trainable parts to fp16."""
        if self.trainable_v3:
            self.v3_model.convert_to_fp16()
        self.convert_new_layers_to_fp16()
    
    def convert_to_fp32(self):
        """Convert all trainable parts to fp32."""
        if self.trainable_v3:
            self.v3_model.convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)
        # self.out_layer = self.out_layer.float()
        # self.mesh_extractor = self.mesh_extractor.float()
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """Convert network output to 512 resolution meshes."""
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret
    
    def forward(self, x, sha256=None, **kwargs):
        # Forward through V3 model
        if self.trainable_v3:
            raise NotImplementedError
        else:
            with torch.no_grad():
                h = self.v3_model.forward_slat(x)
                for block in self.v3_model.upsample:
                    h, _ = block(h, enable_occ=False, is_training=False)
                h_fp32 = h.type(x.dtype)
                valid_occ = self.v3_model.pred_occ_dilate(self.v3_model.out_layer(h_fp32))
                new_coords, new_feats = find_features(valid_coords=valid_occ, h_coords=h.coords, h_feats = h.feats)
                coord_count = new_coords.shape[0]
                if not (0 < coord_count < 900000): 
                    ss_512 = kwargs.get('ss_512', 'unknown')
                    ss_sum = getattr(ss_512, 'sum', lambda: ss_512)()
                    logging.error(f"Pruning failed sha256 {sha256}, hcoord {h.coords.shape}, valid_occ {valid_occ.shape}, coord_count {coord_count}, ss_num {ss_sum}")
                    return None, []
                v3validfeat = SparseTensor(coords=new_coords, feats=new_feats)

        occupancy = []
        for i, block in enumerate(self.upsample):
            h_new, occ_prob = block(v3validfeat, enable_occ=True, is_training=self.training)
            occupancy.append(occ_prob)
            
            if i < len(self.upsample) - 1:
                del v3validfeat
                h = h_new
            else:
                h_512 = h_new
                del v3validfeat

        # Final output layer
        h_512 = h_512.type(x.dtype)
        h_512 = self.out_layer(h_512)
        
        # Convert to mesh representation at 512 resolution
        meshes = self.to_representation(h_512)
        
        if self.training:
            return meshes, occupancy
        else:
            return meshes, []
    
    @torch.no_grad()
    def decode_mesh(self, x: sp.SparseTensor, rank=0):
        ### rank is just a simple tag for use dilate or not, need rename later
        h = self.v3_model.forward_slat(x)
        for block in self.v3_model.upsample:
            h, _ = block(h, enable_occ=False, is_training=False)

        h_fp32 = h.type(x.dtype)
        if rank <= 0:
            valid_occ = self.v3_model.pred_occ(self.v3_model.out_layer(h_fp32))
        else:
            valid_occ = self.v3_model.pred_occ_dilate(self.v3_model.out_layer(h_fp32))

        new_coords, new_feats = find_features(valid_coords=valid_occ, h_coords=h.coords, h_feats = h.feats)
        v3validfeat = SparseTensor(coords=new_coords, feats=new_feats)

        for i, block in enumerate(self.upsample):
            h_new, _ = block(v3validfeat, enable_occ=False, is_training=False)

        h_512 = h_new.type(x.dtype)
        h_512 = self.out_layer(h_512)
        meshes = self.to_representation(h_512)
        return meshes

    @classmethod
    def from_v3_config(cls, v3_config: dict, **v3u_kwargs):
        """
        Create V3U model from V3 config dict.
        
        Args:
            v3_config: V3 configuration dictionary
            **v3u_kwargs: Additional V3U specific arguments (target_resolution, trainable_v3, preload_path)
        
        Example:
            v3_config = {
                "resolution": 64,
                "model_channels": 768,
                "latent_channels": 8,
                ...
            }
            model = SLatMeshDecoderV3U.from_v3_config(
                v3_config,
                target_resolution=512,
                trainable_v3=False,
                preload_path="path/to/v3.pth"
            )
        """
        # Merge V3 config with V3U specific arguments
        config = {**v3_config, **v3u_kwargs}
        return cls(**config)
