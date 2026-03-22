import numpy as np
import torch
import logging


def coords_to_voxel_ply(coordinates, voxel_size=1.0, output_file="voxels.ply"):
    """
    Convert spatial coordinates into tightly connected voxels and save them as a PLY file
    Cubes for adjacent coordinates are tightly connected with no gaps
    
    Parameters:
    coordinates: numpy array or tensor, shape (N, 3) - N 3D coordinate points
    voxel_size: float - edge length of each voxel
    output_file: str - output PLY filename
    """
    # Ensure the input is a NumPy array
    if hasattr(coordinates, 'numpy'):  # If it is a tensor
        coords = coordinates.numpy()
    else:
        coords = np.array(coordinates)
    
    # Convert coordinates to a set for fast lookup
    coord_set = set(map(tuple, coords))
    
    all_vertices = []
    all_faces = []
    vertex_dict = {}  # Store the mapping from vertex coordinates to indices to avoid duplicate vertices
    vertex_count = 0
    
    def get_vertex_index(vertex):
        """Get the vertex index, creating it if the vertex does not already exist"""
        nonlocal vertex_count
        vertex_tuple = tuple(vertex)
        if vertex_tuple not in vertex_dict:
            vertex_dict[vertex_tuple] = vertex_count
            all_vertices.append(vertex)
            vertex_count += 1
        return vertex_dict[vertex_tuple]
    
    # Generate a voxel for each coordinate
    for coord in coords:
        x, y, z = coord
        
        # Compute the 8 vertex coordinates of the current voxel
        # The voxel occupies the space from coord to coord + 1
        v000 = np.array([x, y, z]) * voxel_size
        v001 = np.array([x, y, z + 1]) * voxel_size
        v010 = np.array([x, y + 1, z]) * voxel_size
        v011 = np.array([x, y + 1, z + 1]) * voxel_size
        v100 = np.array([x + 1, y, z]) * voxel_size
        v101 = np.array([x + 1, y, z + 1]) * voxel_size
        v110 = np.array([x + 1, y + 1, z]) * voxel_size
        v111 = np.array([x + 1, y + 1, z + 1]) * voxel_size
        
        vertices = [v000, v001, v010, v011, v100, v101, v110, v111]
        
        # Get vertex indices
        v_indices = [get_vertex_index(v) for v in vertices]
        
        # Define the 6 faces and add a face only when there is no neighboring voxel
        # This avoids internal faces and keeps only the outer surface
        
        # Check whether the 6 neighboring positions contain voxels
        neighbors = {
            'left':  (x - 1, y, z),      # -X face
            'right': (x + 1, y, z),      # +X face  
            'back':  (x, y - 1, z),      # -Y face
            'front': (x, y + 1, z),      # +Y face
            'down':  (x, y, z - 1),      # -Z face
            'up':    (x, y, z + 1),      # +Z face
        }
        
        # Add only faces without adjacent voxels
        faces_to_add = []
        
        # -X face (left)
        if neighbors['left'] not in coord_set:
            faces_to_add.extend([
                [v_indices[0], v_indices[2], v_indices[3]],  # 0-2-3
                [v_indices[0], v_indices[3], v_indices[1]]   # 0-3-1
            ])
        
        # +X face (right) 
        if neighbors['right'] not in coord_set:
            faces_to_add.extend([
                [v_indices[4], v_indices[5], v_indices[7]],  # 4-5-7
                [v_indices[4], v_indices[7], v_indices[6]]   # 4-7-6
            ])
        
        # -Y face (back)
        if neighbors['back'] not in coord_set:
            faces_to_add.extend([
                [v_indices[0], v_indices[1], v_indices[5]],  # 0-1-5
                [v_indices[0], v_indices[5], v_indices[4]]   # 0-5-4
            ])
        
        # +Y face (front)
        if neighbors['front'] not in coord_set:
            faces_to_add.extend([
                [v_indices[2], v_indices[6], v_indices[7]],  # 2-6-7
                [v_indices[2], v_indices[7], v_indices[3]]   # 2-7-3
            ])
        
        # -Z face (down)
        if neighbors['down'] not in coord_set:
            faces_to_add.extend([
                [v_indices[0], v_indices[4], v_indices[6]],  # 0-4-6
                [v_indices[0], v_indices[6], v_indices[2]]   # 0-6-2
            ])
        
        # +Z face (up)
        if neighbors['up'] not in coord_set:
            faces_to_add.extend([
                [v_indices[1], v_indices[3], v_indices[7]],  # 1-3-7
                [v_indices[1], v_indices[7], v_indices[5]]   # 1-7-5
            ])
        
        all_faces.extend(faces_to_add)
    
    # Write the PLY file
    with open(output_file, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(all_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertex data
        for vertex in all_vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write face data
        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"成功生成PLY文件: {output_file}")
    print(f"包含 {len(coords)} 个体素, {len(all_vertices)} 个顶点, {len(all_faces)} 个三角面")


def find_features(valid_coords, h_coords, h_feats):
    """
    Find the intersection of valid_coords and h_coords and return the intersected coordinates and corresponding features
    
    Args:
        valid_coords: [N, D] Coordinates to look up (smaller set, already deduplicated)
        h_coords: [M, D] Large coordinate set (very large, deduplicated, and expected to contain valid_coords)
        h_feats: [M, C] Features corresponding to h_coords
    
    Returns:
        new_coords: [K, D] Intersected coordinates (K <= N), following the order in valid_coords
        new_feats: [K, C] Corresponding features
    """
    assert valid_coords.shape[-1] == h_coords.shape[-1]
    
    device = valid_coords.device
    N = valid_coords.size(0)
    M = h_coords.size(0)
    C = h_feats.size(1)
    
    # Concatenate the two coordinate sets
    both = torch.cat([h_coords, valid_coords], dim=0)  # [M+N, D]
    uniq, inv = torch.unique(both, return_inverse=True, dim=0)
    
    # Split the inverse indices
    inv_h = inv[:M]      # inverse indices for h_coords
    inv_valid = inv[M:]  # inverse indices for valid_coords
    
    h_row_ids = torch.arange(M, device=device, dtype=torch.long)
    map_id2h_row = torch.full((uniq.size(0),), -1, device=device, dtype=torch.long)
    map_id2h_row.scatter_(0, inv_h, h_row_ids)
    
    idx_in_h = map_id2h_row[inv_valid]  # [N]
    mask = idx_in_h != -1
    n_matched = mask.sum().item()
    
    if n_matched > 0:
        new_coords = valid_coords[mask]  # [K, D] Keep only the matching coordinates
        new_feats = h_feats[idx_in_h[mask]]  # [K, C] Corresponding features
    else:
        new_coords = torch.empty(0, valid_coords.size(1), device=device, dtype=valid_coords.dtype)
        new_feats = torch.empty(0, C, device=device, dtype=h_feats.dtype)
    
    return new_coords, new_feats


def test_find_features():
    """Test the find_features function"""
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    print("Test 1: 正常情况 - h_coords 包含所有 valid_coords")
    # h_coords is the large set
    h_coords = torch.tensor([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], 
        [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]
    ], dtype=torch.float32)
    h_feats = torch.arange(10).unsqueeze(1).expand(-1, 2).float() * 10  # [[0,0], [10,10], [20,20], ...]
    
    # valid_coords is the smaller set and is fully contained
    valid_coords = torch.tensor([[2, 2], [5, 5], [8, 8]], dtype=torch.float32)
    
    new_coords, new_feats = find_features(valid_coords, h_coords, h_feats)
    
    print(f"Valid coords: {valid_coords.tolist()}")
    print(f"New coords: {new_coords.tolist()}")
    print(f"New feats: {new_feats.tolist()}")
    
    expected_feats = torch.tensor([[20, 20], [50, 50], [80, 80]], dtype=torch.float32)
    assert torch.allclose(new_coords, valid_coords), "坐标应该保持不变"
    assert torch.allclose(new_feats, expected_feats), "特征应该正确匹配"
    print("✓ Test 1 passed\n")
    
    
    print("Test 2: 极端情况 - 有少量 valid_coords 不在 h_coords 中")
    h_coords = torch.tensor([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4]
    ], dtype=torch.float32)
    h_feats = torch.arange(5).unsqueeze(1).expand(-1, 2).float() * 10
    
    # valid_coords includes [5, 5], which is not in h_coords
    valid_coords = torch.tensor([[1, 1], [5, 5], [3, 3]], dtype=torch.float32)
    
    print("Expected warning about unmatched coordinates:")
    new_coords, new_feats = find_features(valid_coords, h_coords, h_feats)
    
    print(f"Valid coords: {valid_coords.tolist()}")
    print(f"New coords: {new_coords.tolist()}")
    print(f"New feats: {new_feats.tolist()}")
    
    expected_feats = torch.tensor([[10, 10], [0, 0], [30, 30]], dtype=torch.float32)  # the feature for [5,5] is 0
    assert torch.allclose(new_coords, valid_coords), "坐标应该保持不变"
    assert torch.allclose(new_feats, expected_feats), "匹配的特征正确，未匹配的为0"
    print("✓ Test 2 passed\n")
    
    
    print("Test 3: 大规模测试 - 模拟实际场景")
    # Create a large h_coords set (for example, 10,000 points)
    M = 10000
    h_coords = torch.randn(M, 4) * 100  # 4D coordinates
    h_feats = torch.randn(M, 256)  # 256D features
    
    # Select 100 of them as valid_coords
    N = 100
    selected_indices = torch.randperm(M)[:N]
    valid_coords = h_coords[selected_indices]
    
    new_coords, new_feats = find_features(valid_coords, h_coords, h_feats)
    
    print(f"h_coords shape: {h_coords.shape}")
    print(f"valid_coords shape: {valid_coords.shape}")
    print(f"new_coords shape: {new_coords.shape}")
    print(f"new_feats shape: {new_feats.shape}")
    
    # Verify that the returned features are correct
    expected_feats = h_feats[selected_indices]
    assert torch.allclose(new_coords, valid_coords), "坐标应该保持不变"
    assert torch.allclose(new_feats, expected_feats, atol=1e-5), "特征应该正确匹配"
    print("✓ Test 3 passed\n")
    
    
    print("Test 4: 边界情况 - valid_coords 为空")
    h_coords = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    h_feats = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32)
    valid_coords = torch.empty(0, 2, dtype=torch.float32)
    
    new_coords, new_feats = find_features(valid_coords, h_coords, h_feats)
    
    assert new_coords.shape == (0, 2), "空输入应该返回空输出"
    assert new_feats.shape == (0, 2), "空输入应该返回空输出"
    print("✓ Test 4 passed\n")
    
    
    print("Test 5: 顺序保持测试")
    h_coords = torch.tensor([
        [9, 9], [8, 8], [7, 7], [6, 6], [5, 5],
        [4, 4], [3, 3], [2, 2], [1, 1], [0, 0]
    ], dtype=torch.float32)  # h_coords is in reverse order
    h_feats = torch.arange(10).unsqueeze(1).expand(-1, 2).float() * 10
    
    # valid_coords in a specific order
    valid_coords = torch.tensor([[5, 5], [2, 2], [8, 8], [0, 0]], dtype=torch.float32)
    
    new_coords, new_feats = find_features(valid_coords, h_coords, h_feats)
    
    print(f"Valid coords order: {valid_coords.tolist()}")
    print(f"New coords order: {new_coords.tolist()}")
    print(f"New feats: {new_feats.tolist()}")
    
    # Verify that the order remains unchanged
    assert torch.allclose(new_coords, valid_coords), "应该保持valid_coords的原始顺序"
    # [5,5] is at index 4 in h_coords, [2,2] at index 7, [8,8] at index 1, and [0,0] at index 9
    expected_feats = torch.tensor([[40, 40], [70, 70], [10, 10], [90, 90]], dtype=torch.float32)
    assert torch.allclose(new_feats, expected_feats), "特征应该按照valid_coords的顺序返回"
    print("✓ Test 5 passed\n")
    
    print("所有测试通过！")


if __name__ == "__main__":
    test_find_features()

