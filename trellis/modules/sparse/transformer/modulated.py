from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,

    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)


def dense_to_sparse_tokens(cond: torch.Tensor) -> 'SparseTensor':
    """
    cond: [B, T, C] → SparseTensor with coords[:,0]=batch_id, coords[:,1]=seq_idx
    The generated order is T tokens from batch0, then T tokens from batch1, ensuring a contiguous layout.
    """
    B, T, C = cond.shape
    device = cond.device
    batch_idx = torch.arange(B, device=device).repeat_interleave(T)      # [B*T]
    
    coords = torch.cat([batch_idx.unsqueeze(-1), torch.zeros((B*T,3), device = cond.device)], dim=1).to(torch.int32)    # [B*T, 2]
    feats  = cond.reshape(B*T, C)
    from ..basic import SparseTensor  # Location of your SparseTensor definition
    return SparseTensor(feats=feats, coords=coords)

from typing import List, Tuple
import torch

def sparse_cat_tokens_by_batch(inputs: List['SparseTensor']) -> Tuple['SparseTensor', list]:
    """
    Concatenate multiple SparseTensors along the token dimension within each batch while keeping batch IDs unchanged.
    Assume these inputs share the same batch size and batch ordering.
    Returns: (the concatenated SparseTensor, token counts of the first input in each batch, n0_per_batch)
    Used later to split the attention output back into per-batch segments.
    """
    assert len(inputs) >= 1
    B = inputs[0].shape[0]
    for t in inputs[1:]:
        assert t.shape[0] == B, "All inputs must share the same batch size"

    coords_all = []
    feats_all  = []
    n0_per_batch = []

    for b in range(B):
        # Extract and concatenate contiguous segments from each input batch while preserving batch-local contiguity
        coords_b_list, feats_b_list = [], []
        for j, t in enumerate(inputs):
            sl = t.layout[b]         # The contiguous slice of batch b within t
            coords_b = t.coords[sl]
            feats_b  = t.feats[sl]
            # Ensure the batch ID is b (usually already true, but enforced here for safety)
            coords_b = coords_b.clone()
            coords_b[:, 0] = b
            coords_b_list.append(coords_b)
            feats_b_list.append(feats_b)
            if j == 0:
                n0 = feats_b.shape[0]    # Record the number of tokens from the first input in this batch
        n0_per_batch.append(n0)
        coords_all.append(torch.cat(coords_b_list, dim=0))
        feats_all.append(torch.cat(feats_b_list,  dim=0))

    coords = torch.cat(coords_all, dim=0).contiguous()
    feats  = torch.cat(feats_all,  dim=0).contiguous()
    from ..basic import SparseTensor
    out = SparseTensor(coords=coords, feats=feats)  # The layout is rebuilt during construction to keep each batch contiguous
    return out, n0_per_batch

def split_joint_output_by_batch(joint: 'SparseTensor', n0_per_batch: list) -> Tuple['SparseTensor', torch.Tensor]:
    """
    joint: joint SparseTensor concatenated batch by batch (attention output)
    n0_per_batch: number of tokens from the first input in each batch (x tokens here)
    Returns: updated x_sparse (kept as SparseTensor) and updated cond_dense [B, T, C]
    """
    B = joint.shape[0]
    C = joint.feats.shape[-1]
    x_feats_list = []
    c_feats_list = []

    for b in range(B):
        sl = joint.layout[b]                   # The contiguous range of this batch in joint
        feats_b = joint.feats[sl]              # [N_b_total, C] = [N_x_b + T_b, C]
        n0 = n0_per_batch[b]
        x_feats_b = feats_b[:n0]               # The first n0 entries belong to x
        c_feats_b = feats_b[n0:]               # The remaining entries belong to cond
        x_feats_list.append(x_feats_b)
        c_feats_list.append(c_feats_b)

    # Reconstruct x as a SparseTensor by taking the first n0 entries of each batch directly from joint
    coords_x = []
    feats_x  = []
    coords_c = []
    feats_c  = []
    for b in range(B):
        sl = joint.layout[b]
        coords_b = joint.coords[sl]
        n0 = n0_per_batch[b]
        coords_x.append(coords_b[:n0])
        feats_x.append(joint.feats[sl][:n0])
        coords_c.append(coords_b[n0:])
        feats_c.append(joint.feats[sl][n0:])

    coords_x = torch.cat(coords_x, dim=0).contiguous()
    feats_x  = torch.cat(feats_x,  dim=0).contiguous()
    from ..basic import SparseTensor
    x_sparse = SparseTensor(coords=coords_x, feats=feats_x)

    # Restore cond to [B, T, C] (T_b may be equal or different; if different, keep a list or pad)
    # In your setup cond has a fixed length T, so reshape it directly:
    T = feats_c[0].shape[0]  # Assume each batch has the same cond length
    cond_dense = torch.stack([f.reshape(T, C) for f in feats_c], dim=0)  # [B, T, C]
    return x_sparse, cond_dense

class ModulatedSparseTransformerMMDiTBlock(nn.Module):
    """
    MMDiT: merge tokens from x (sparse) and cond (dense) within the same batch for self-attention, then split them back into two branches.
    AdaLN-zero: LN(affine=False) + (1 + scale) * x + shift, with a gate on the residual branch.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        require_cond_feedforward: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)

        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode="full",
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )

        # Sparse branch: use your existing Sparse FFN (pointwise equivalent)
        self.mlp_sparse = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        # Dense branch (cond): use an equivalent dense FFN (token-wise)

        self.require_cond_feedforward = require_cond_feedforward

        if require_cond_feedforward:
            hidden = int(mlp_ratio * channels)
            self.mlp_dense = nn.Sequential(
                nn.Linear(channels, hidden, bias=True),
                nn.SiLU(),
                nn.Linear(hidden, channels, bias=True),
            )

        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        # self.cond_adaLN_modulation = nn.Sequential(
        # )
            # AdaLN-zero: zero-initialize the last layer during model initialization (or zero it here)

    def _apply_adaln(self, x_feats: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # x_feats: [N, C] or [B, T, C]; shift/scale: [B, C]
        if x_feats.dim() == 2:
            # Sparse branch: broadcast [B, C] to each sparse batch segment
            # You can also use sparse_batch_broadcast; here is the simplest handwritten version:
            # (Using your existing sparse_batch_op would be more robust)
            raise NotImplementedError  # Handle this batch by batch below to avoid broadcasting here
        else:
            # Dense branch: broadcast directly across the T dimension
            return x_feats * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _forward(self, x: SparseTensor, mod: torch.Tensor, cond: torch.Tensor) -> Tuple[SparseTensor, torch.Tensor]:
        """
        x: SparseTensor (∑N, C)
        cond: [B, T, C]
        mod: [B, C] (timestep embedding; if share_mod=False, it is projected to 6 * C in this layer)
        """
        B = x.shape[0]
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)

        # shift_msa, scale_msa, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = self.cond_adaLN_modulation(mod).chunk(6, dim=1)

        
        # ====== Pre-norm + AdaLN-zero ======
        # Sparse branch: process batch by batch to avoid broadcasting issues
        x_norm = x.replace(self.norm1(x.feats))
        x_adaln = x_norm * (1 + scale_msa) + shift_msa

        # dense cond
        cond_norm = self.norm1(cond)
        cond_adaln = cond_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        # ====== Build a sparse representation for cond and perform self-attention on jointly concatenated per-batch tokens ======
        cond_sparse = dense_to_sparse_tokens(cond_adaln)           # [B*T, C] + coords

        joint, n0_per_batch = sparse_cat_tokens_by_batch([x_adaln, cond_sparse])  # Merge tokens batch by batch

        joint_out = self.attn(joint)  # self-attn over (x + cond) in each batch

        # ====== Split back into sparse x and dense cond batch by batch ======
        x_attn, cond_attn = split_joint_output_by_batch(joint_out, n0_per_batch)


        # Gate residual
        # ).unsqueeze(-1))
        x = x + x_attn * gate_msa
        cond = cond + (cond_attn * gate_msa.unsqueeze(1))

        # ====== Two MLP branches ======
        # Sparse: pre-norm + AdaLN
        x2 = x.replace(self.norm2(x.feats))
        x2 = x2 * (1 + scale_mlp) + shift_mlp
        x2 = self.mlp_sparse(x2)
        x = x + x2 * gate_mlp

        # Dense branch: in the last block no cond output is required, so disable this feedforward to avoid None gradients
        if self.require_cond_feedforward: 
            c2 = self.norm2(cond)
            c2 = c2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            c2 = self.mlp_dense(c2)
            cond = cond + c2 * gate_mlp.unsqueeze(1)

        return x, cond

    def forward(self, x: SparseTensor, mod: torch.Tensor, cond: torch.Tensor) -> Tuple[SparseTensor, torch.Tensor]:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, cond, use_reentrant=False)
        else:
            return self._forward(x, mod, cond)
        
