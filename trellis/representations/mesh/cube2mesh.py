import numpy as np
import torch
import trimesh
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes_triposf.flexicubes import FlexiCubes


class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals   

    def contiguous(self):
        return self

    def save_obj(self, filepath, save_vertex_colors=False, save_normals=True):
        """
        Save the mesh as an OBJ file
        
        Args:
            filepath (str): Output file path, for example 'output.obj'
            save_vertex_colors (bool): Whether to save vertex colors when vertex_attrs contains color information
            save_normals (bool): Whether to save vertex normals
        """
        # Convert tensors to NumPy arrays
        vertices = self.vertices.detach().cpu().numpy() if torch.is_tensor(self.vertices) else self.vertices
        faces = self.faces.detach().cpu().numpy() if torch.is_tensor(self.faces) else self.faces

        # adjust rotation
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        
        # Compute vertex normals if needed
        vertex_normals = None
        if save_normals:
            vertex_normals = self.comput_v_normals(self.vertices, self.faces)
            vertex_normals = vertex_normals.detach().cpu().numpy() if torch.is_tensor(vertex_normals) else vertex_normals
        
        # Get vertex colors if available and requested
        vertex_colors = None
        if save_vertex_colors and self.vertex_attrs is not None:
            vertex_attrs = self.vertex_attrs.detach().cpu().numpy() if torch.is_tensor(self.vertex_attrs) else self.vertex_attrs
            # Assume the first 3 channels of vertex_attrs store RGB colors
            if vertex_attrs.shape[-1] >= 3:
                vertex_colors = vertex_attrs[..., :3]
                # Ensure color values stay within [0, 1]
                vertex_colors = np.clip(vertex_colors, 0, 1)
        
        # Write the OBJ file
        with open(filepath, 'w') as f:
            # Write the file header
            f.write("# OBJ file generated from MeshExtractResult\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n\n")
            
            # Write vertices
            for i, v in enumerate(vertices):
                if vertex_colors is not None:
                    # Vertex with color (v x y z r g b)
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {vertex_colors[i][0]:.6f} {vertex_colors[i][1]:.6f} {vertex_colors[i][2]:.6f}\n")
                else:
                    # Vertex with coordinates only
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # Write vertex normals
            if vertex_normals is not None:
                for vn in vertex_normals:
                    f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
                f.write("\n")
            
            # Write faces (OBJ indices start from 1)
            for face in faces:
                if vertex_normals is not None:
                    # Face with normals (f v//vn v//vn v//vn)
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    # Face with vertex indices only
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"Mesh saved to {filepath}")
        print(f"  - Vertices: {len(vertices)}")
        print(f"  - Faces: {len(faces)}")
        if vertex_colors is not None:
            print(f"  - With vertex colors")
        if vertex_normals is not None:
            print(f"  - With vertex normals")

    def save_glb(self, filepath, save_vertex_colors=False, save_normals=True):
        # Vertices and faces
        vertices = self.vertices.detach().cpu().numpy() if hasattr(self.vertices, "detach") else self.vertices
        faces = self.faces.detach().cpu().numpy() if hasattr(self.faces, "detach") else self.faces
        # rotate mesh (from z-up to y-up)
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        # Vertex colors
        vertex_colors = None
        if save_vertex_colors and self.vertex_attrs is not None:
            vertex_attrs = self.vertex_attrs.detach().cpu().numpy() if hasattr(self.vertex_attrs, "detach") else self.vertex_attrs
            if vertex_attrs.shape[-1] >= 3:
                # glTF requires uint8 values in the range [0, 255]
                vertex_colors = np.clip(vertex_attrs[..., :3] * 255, 0, 255).astype(np.uint8)
    
        # Create the mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
        # Attach colors
        if vertex_colors is not None:
            mesh.visual.vertex_colors = vertex_colors
    
        # Save as .glb
        mesh.export(filepath, file_type="glb")
        print(f"Mesh with {len(vertices)} vertices saved to {filepath}")


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        '''
        a model to generate a mesh from sparse features structures using flexicube
        '''
        super().__init__()
        self.device=device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        # verts, cube = construct_dense_grid(self.res, self.device)
        # self.reg_c = cube.to(self.device)
        # self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()
    
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start
        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        
        res_v = self.res + 1
        v_attrs_d, v_pos_dilate = get_sparse_attrs(v_pos, v_attrs, res=res_v, sdf_init=True)
        weights_d, coords_dilate = get_sparse_attrs(coords, weights, res=self.res, sdf_init=False)

        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        x_nx3 = get_defomed_verts(v_pos_dilate, deform_d, self.res)
        x_nx3 = torch.cat((x_nx3, torch.ones((1, 3), dtype=x_nx3.dtype, device=x_nx3.device) * 0.5))
        sdf_d = torch.cat((sdf_d, torch.ones((1), dtype=sdf_d.dtype, device=sdf_d.device)))
        
        mask_reg_c_sparse = (v_pos_dilate[..., 0] * res_v + v_pos_dilate[..., 1]) * res_v + v_pos_dilate[..., 2]
        reg_c_sparse = (coords_dilate[..., 0] * res_v + coords_dilate[..., 1]) * res_v + coords_dilate[..., 2]

        cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]            
        reg_c_value = (reg_c_sparse.unsqueeze(1) + cube_corners_bias.unsqueeze(0).cuda()).reshape(-1)
        reg_c = torch.searchsorted(mask_reg_c_sparse, reg_c_value)
        exact_match_mask = mask_reg_c_sparse[reg_c] == reg_c_value
        reg_c[exact_match_mask == 0] = len(mask_reg_c_sparse)
        reg_c = reg_c.reshape(-1, 8)

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            cube_index_map=coords_dilate,
            training=training)
        
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
    
    @torch.no_grad()
    def forward_flexicubes(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=False)
        
        res_v = self.res + 1
        v_attrs_d, v_pos_dilate = get_sparse_attrs(v_pos, v_attrs, res=res_v, sdf_init=True)
        weights_d, coords_dilate = get_sparse_attrs(coords, weights, res=self.res, sdf_init=False)

        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        x_nx3 = get_defomed_verts(v_pos_dilate, deform_d, self.res)
        x_nx3 = torch.cat((x_nx3, torch.ones((1, 3), dtype=x_nx3.dtype, device=x_nx3.device) * 0.5))
        sdf_d = torch.cat((sdf_d, torch.ones((1), dtype=sdf_d.dtype, device=sdf_d.device)))
        
        mask_reg_c_sparse = (v_pos_dilate[..., 0] * res_v + v_pos_dilate[..., 1]) * res_v + v_pos_dilate[..., 2]
        reg_c_sparse = (coords_dilate[..., 0] * res_v + coords_dilate[..., 1]) * res_v + coords_dilate[..., 2]

        cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]            
        reg_c_value = (reg_c_sparse.unsqueeze(1) + cube_corners_bias.unsqueeze(0).cuda()).reshape(-1)
        reg_c = torch.searchsorted(mask_reg_c_sparse, reg_c_value)
        exact_match_mask = mask_reg_c_sparse[reg_c] == reg_c_value
        reg_c[exact_match_mask == 0] = len(mask_reg_c_sparse)
        reg_c = reg_c.reshape(-1, 8)

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            cube_index_map=coords_dilate,
            training=training)

        occ = transform_vertices_to_occ(vertices, self.res)
        return occ

    @torch.no_grad()
    def extract_occ(self, cubefeats : SparseTensor, training=False):
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        
        res_v = self.res + 1
        v_attrs_d, v_pos_dilate = get_sparse_attrs(v_pos, v_attrs, res=res_v, sdf_init=True)
        weights_d, coords_dilate = get_sparse_attrs(coords, weights, res=self.res, sdf_init=False)

        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        x_nx3 = get_defomed_verts(v_pos_dilate, deform_d, self.res)
        x_nx3 = torch.cat((x_nx3, torch.ones((1, 3), dtype=x_nx3.dtype, device=x_nx3.device) * 0.5))
        sdf_d = torch.cat((sdf_d, torch.ones((1), dtype=sdf_d.dtype, device=sdf_d.device)))
        
        mask_reg_c_sparse = (v_pos_dilate[..., 0] * res_v + v_pos_dilate[..., 1]) * res_v + v_pos_dilate[..., 2]
        reg_c_sparse = (coords_dilate[..., 0] * res_v + coords_dilate[..., 1]) * res_v + coords_dilate[..., 2]

        cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]            
        reg_c_value = (reg_c_sparse.unsqueeze(1) + cube_corners_bias.unsqueeze(0).cuda()).reshape(-1)
        reg_c = torch.searchsorted(mask_reg_c_sparse, reg_c_value)
        exact_match_mask = mask_reg_c_sparse[reg_c] == reg_c_value
        reg_c[exact_match_mask == 0] = len(mask_reg_c_sparse)
        reg_c = reg_c.reshape(-1, 8)
        return self.mesh_extractor._identify_valid(sdf_d, reg_c)

    @torch.no_grad()
    def occ_based_pruning(self, cubefeats : SparseTensor): # training is always false
        valid_occ = self.extract_occ(cubefeats)
        new_feats = cubefeats.feats[valid_occ]
        new_coords = cubefeats.coords[valid_occ]
        return SparseTensor(
            coords=new_coords,
            feats=new_feats,
        )

    @torch.no_grad()
    def forward_flexicubes_with_dilate27(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=False)
        
        res_v = self.res + 1
        v_attrs_d, v_pos_dilate = get_sparse_attrs(v_pos, v_attrs, res=res_v, sdf_init=True)
        weights_d, coords_dilate = get_sparse_attrs(coords, weights, res=self.res, sdf_init=False)

        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        x_nx3 = get_defomed_verts(v_pos_dilate, deform_d, self.res)
        x_nx3 = torch.cat((x_nx3, torch.ones((1, 3), dtype=x_nx3.dtype, device=x_nx3.device) * 0.5))
        sdf_d = torch.cat((sdf_d, torch.ones((1), dtype=sdf_d.dtype, device=sdf_d.device)))
        
        mask_reg_c_sparse = (v_pos_dilate[..., 0] * res_v + v_pos_dilate[..., 1]) * res_v + v_pos_dilate[..., 2]
        reg_c_sparse = (coords_dilate[..., 0] * res_v + coords_dilate[..., 1]) * res_v + coords_dilate[..., 2]

        cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]            
        reg_c_value = (reg_c_sparse.unsqueeze(1) + cube_corners_bias.unsqueeze(0).cuda()).reshape(-1)
        reg_c = torch.searchsorted(mask_reg_c_sparse, reg_c_value)
        exact_match_mask = mask_reg_c_sparse[reg_c] == reg_c_value
        reg_c[exact_match_mask == 0] = len(mask_reg_c_sparse)
        reg_c = reg_c.reshape(-1, 8)

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            cube_index_map=coords_dilate,
            training=training)
        occ = dilate27_dense_pool(vertices, self.res, device=vertices.device)
        return occ