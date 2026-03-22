import torch
import torch.nn.functional as F
cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
        1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.int)
cube_neighbor = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
                2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.long, requires_grad=False)
     
def construct_dense_grid(res, device='cuda'):
    '''construct a dense grid based on resolution'''
    res_v = res + 1
    vertsid = torch.arange(res_v ** 3, device=device)
    coordsid = vertsid.reshape(res_v, res_v, res_v)[:res, :res, :res].flatten()
    cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
    cube_fx8 = (coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device))
    verts = torch.stack([vertsid // (res_v ** 2), (vertsid // res_v) % res_v, vertsid % res_v], dim=1)
    return verts, cube_fx8


def construct_voxel_grid(coords):
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes


def cubes_to_verts(num_verts, cubes, value, reduce='mean'):
    """
    Args:
        cubes [Vx8] verts index for each cube
        value [Vx8xM] value to be scattered
    Operation:
        reduced[cubes[i][j]][k] += value[i][k]
    """
    M = value.shape[2] # number of channels
    reduced = torch.zeros(num_verts, M, device=cubes.device)

    # print(f"cubes 范围: [{cubes.min()}, {cubes.max()}]")
    # print(f"num_verts: {num_verts}")
    # assert cubes.min() >= 0, f"发现负索引: {cubes.min()}"
    # assert cubes.max() < num_verts, f"索引越界: {cubes.max()} >= {num_verts}"
    # print(f"cubes.dtype: {cubes.dtype}")
    # print(f"🔍 快速检查 - cubes范围: [{cubes.min()}, {cubes.max()}], num_verts: {num_verts}")
    # if cubes.min() < 0 or cubes.max() >= num_verts:
    #     print(f"❌ 索引越界!")

    return torch.scatter_reduce(reduced, 0, 
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1), 
        value.flatten(0, 1), reduce=reduce, include_self=False)
    
def sparse_cube2verts(coords, feats, training=True):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    if training:
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    return new_coords, new_feats, con_loss
    

def get_dense_attrs(coords : torch.Tensor, feats : torch.Tensor, res : int, sdf_init=True):
    F = feats.shape[-1]
    dense_attrs = torch.zeros([res] * 3 + [F], device=feats.device)
    if sdf_init:
        dense_attrs[..., 0] = 1 # initial outside sdf value
    dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats
    return dense_attrs.reshape(-1, F)

def get_sparse_attrs(coords : torch.Tensor, feats : torch.Tensor, res : int, sdf_init=True):
    verts = coords
    verts, masks = torch.unique(verts, dim=0, return_inverse=True)
    feats_sparse = torch.zeros((len(verts), feats.shape[-1]), device=feats.device, dtype=feats.dtype)
    feats_sparse[masks] = feats
    return feats_sparse, verts


def get_defomed_verts(v_pos : torch.Tensor, deform : torch.Tensor, res):
    return v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)


def transform_vertices_to_occ_dilate8(verts: torch.Tensor, reso: int):
    '''
        This is a simple 1 -> 8 version
    '''
    # center = (verts.max(dim=0)[0] + verts.min(dim=0)[0]) / 2
    # verts_centered = verts - center

    # # 找到最大范围
    # max_range = (verts.max(dim=0)[0] - verts.min(dim=0)[0]).max()

    # # 缩放到[-0.5, 0.5]
    # scale = 1.0 / max_range if max_range > 0 else 1.0
    # verts = verts_centered * scale

    # coords = torch.floor((verts + 0.5) * reso - 0.5 + 1e-6).int() # flooring
    coords = torch.floor((verts + 0.5) * reso).int() # flooring
    coords_all = []
    for i in range(8):
        offset_coords = coords + torch.tensor(
            (i >> 2, (i >> 1) & 1, i & 1), 
            dtype=coords.dtype, 
            device=coords.device
        )
        coords_all.append(offset_coords.clamp(min=0, max=reso-1))
    # coords = torch.ceil((verts + 0.5) * reso - 0.5).int() # ceiling
    # unique_coords = torch.unique(coords, dim=0)
    unique_coords = torch.unique(torch.cat(coords_all), dim=0)
    return unique_coords

def transform_vertices_to_occ(verts: torch.Tensor, reso: int):
    coords = torch.floor((verts + 0.5) * reso).int()
    offsets = torch.tensor([
        [0, 0, 0],   
        [-1, 0, 0],  
        [1, 0, 0],   
        [0, -1, 0],  
        [0, 1, 0],   
        [0, 0, -1],  
        [0, 0, 1],   
    ], dtype=coords.dtype, device=coords.device)
    
    coords_dilated = coords.unsqueeze(1) + offsets.unsqueeze(0)  # [N, 7, 3]
    coords_dilated = coords_dilated.reshape(-1, 3)  # [N*7, 3]

    coords_dilated = coords_dilated.clamp(min=0, max=reso-1)
    unique_coords = torch.unique(coords_dilated, dim=0)
    
    return unique_coords

def dilate27_dense_pool(coords_xyz: torch.Tensor, reso: int, device=None) -> torch.Tensor:
    if coords_xyz.numel() == 0:
        return coords_xyz.new_zeros((0, 3), dtype=torch.int32)

    device = coords_xyz.device if device is None else device
    occ = torch.zeros((reso, reso, reso), dtype=torch.uint8, device=device)
    x, y, z = coords_xyz.unbind(dim=1)
    occ[z.long(), y.long(), x.long()] = 1  # [Z,Y,X]

    occ_f = occ.unsqueeze(0).unsqueeze(0).to(torch.float32)  # [1,1,D,H,W]
    dil = F.max_pool3d(occ_f, kernel_size=3, stride=1, padding=1)
    dil = (dil.squeeze(0).squeeze(0) > 0)  # bool

    idx_zyx = dil.nonzero(as_tuple=False).to(torch.int32)  # [K,3] in [z,y,x]
    out = torch.stack([idx_zyx[:, 2], idx_zyx[:, 1], idx_zyx[:, 0]], dim=1)  # [x,y,z]
    return out

def find_features(valid_coords, h_coords, h_feats):
    assert valid_coords.shape[-1] == h_coords.shape[-1]    
    both = torch.cat([valid_coords, h_coords], dim=0)  # [N+M, 4]
    uniq, inv = torch.unique(both, return_inverse=True, dim=0)

    inv_coords = inv[:h_coords.size(0)]  # [N]
    inv_valid = inv[valid_coords.size(0):]   # [M]
    
    # 构建 unique_id -> h_coords 行号的映射
    row_ids = torch.arange(valid_coords.size(0), device=valid_coords.device, dtype=torch.long)
    map_id2row = torch.full((uniq.size(0),), -1, device=valid_coords.device, dtype=torch.long)
    
    # 使用scatter处理重复（保留第一次出现的行号）
    map_id2row.scatter_(0, inv_coords.flip(0), row_ids.flip(0))
    
    # 查找valid在h_coords中的行号
    idx_in_coords = map_id2row[inv_valid]  # [M]
    
    # 创建匹配掩码
    mask = idx_in_coords != -1
    
    # 提取匹配的特征
    if mask.any():
        matched_feats = h_feats[idx_in_coords[mask]]  # [M_matched, C]
    else:
        matched_feats = torch.empty(0, h_feats.size(1), device=h_feats.device, dtype=h_feats.dtype)
    
    return matched_feats, mask, idx_in_coords