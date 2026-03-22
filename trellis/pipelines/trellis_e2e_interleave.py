from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
import os

from transformers import CLIPTextModel, AutoTokenizer, AutoProcessor



class TrellisE2EInterleaveResCondPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisE2EInterleaveResCondPipeline, TrellisE2EInterleaveResCondPipeline).from_pretrained(path)
        new_pipeline = TrellisE2EInterleaveResCondPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization'] if "slat_normalization" in args.keys() else None 
        if new_pipeline.slat_normalization is None:
            print("None in new_pipeline.slat_normalization!!!!!")
        new_pipeline._init_image_cond_model(args['image_cond_model'])
        new_pipeline._init_text_cond_model(args['txt_cond_model'])
        new_pipeline.resolution = new_pipeline.models['sparse_structure_flow_model'].resolution
        

        return new_pipeline

    def _get_runtime_modules(self) -> dict[str, nn.Module]:
        runtime_modules = super()._get_runtime_modules()
        if hasattr(self, 'image_cond_model') and isinstance(self.image_cond_model, dict) and 'model' in self.image_cond_model:
            runtime_modules['image_cond_model'] = self.image_cond_model['model']
        if hasattr(self, 'text_cond_model') and isinstance(self.text_cond_model, dict) and 'model' in self.text_cond_model:
            runtime_modules['text_cond_model'] = self.text_cond_model['model']
        return runtime_modules
    
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        if "dinov3" in name:
            # facebook/dinov3-vith16plus-pretrain-lvd1689m

            self.dino_github = './external/dinov3'
            self.dino_weight = "./external/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
            self.image_resolution = 592
        elif "dinov2" in name:
            self.dino_github = './external/facebookresearch_dinov2_main'
            self.image_resolution = 518
        if "dinov2" in name:
            dinov2_model = torch.hub.load(self.dino_github, name,source="local",  pretrained=True)
        else:
            dinov2_model = torch.hub.load(self.dino_github, name,source="local", weights=self.dino_weight, pretrained=True)
        # self.models['image_cond_model'] = dinov2_model

        dinov2_model = dinov2_model.eval().to(self.execution_device)
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # self.image_cond_model_transform = transform
        self.image_cond_model = {
            'model':dinov2_model,
            'transform': transform
        }

    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.to(self.execution_device)
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    def _alpha_crop_resize_like_old(self, im: Image.Image, out_size: int) -> Image.Image:
        # 1) Ensure RGBA (the original implementation assumes an alpha channel)
        if im.mode != "RGBA":
            im = im.convert("RGBA")

        alpha = np.array(im.getchannel(3))
        nz = np.nonzero(alpha)
        if len(nz[0]) == 0:
            im = im.resize((out_size, out_size), Image.Resampling.LANCZOS)
            return im.convert("RGB")

        x0, y0 = int(np.min(nz[1])), int(np.min(nz[0]))
        x1, y1 = int(np.max(nz[1])), int(np.max(nz[0]))
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        hsize = max(x1 - x0, y1 - y0) / 2.0
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio

        bx0 = int(round(center_x - aug_hsize))
        by0 = int(round(center_y - aug_hsize))
        bx1 = int(round(center_x + aug_hsize))
        by1 = int(round(center_y + aug_hsize))
        bx0 = max(0, bx0); by0 = max(0, by0)
        bx1 = min(im.width, bx1); by1 = min(im.height, by1)

        im = im.crop((bx0, by0, bx1, by1)).resize((out_size, out_size), Image.Resampling.LANCZOS)

        arr = np.array(im).astype(np.float32) / 255.0        # [H,W,4]
        rgb = arr[:, :, :3]
        a   = arr[:, :, 3:4]
        rgb = rgb * a                                       
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(rgb_uint8, mode="RGB")
        
    def preprocess_image(self, input: Image.Image, res=518, save_cond_path = None) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]

        bbox = np.argwhere(alpha > 0. * 255)
        
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((res, res), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        real_output = output[:, :, :3] * output[:, :, 3:4]
        real_output = Image.fromarray((real_output * 255).astype(np.uint8))

        if save_cond_path:
            output_white_bg = output[:, :, :3] * output[:, :, 3:4] + (1-output[:, :, 3:4])
            output_white_bg = Image.fromarray((output_white_bg * 255).astype(np.uint8))
            
            output_white_bg.save(save_cond_path)
        return real_output
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]], image_res=518) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((image_res, image_res), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model['transform'](image).to(self.device)
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']# [1,1374,1024]
        patchtokens = F.layer_norm(features, features.shape[-1:])

        return patchtokens

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and isinstance(text[0], str), "TextConditionedMixin only supports list of strings as cond"

        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].to(self.device)
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state

        return embeddings
        

    def get_cond(self, text: List[str], image: List[List[Image.Image]]) -> dict:
        """
        Get the conditioning data.
        Must have images, text is optional
        """

        txt = []
        input_images = []
        for cap, imgs in zip(text, image):
            input_images += imgs
            if cap:
                txt.append(cap)
            else:
                txt.append('')

        txt_cond = self.encode_text(txt)
        if len(input_images):
            image_cond = self.encode_image(input_images, 592)
        else:
            image_cond = torch.zeros([1, 1374, 1280], device=self.device)
        
        cond = {
            'image_cond':image_cond, 
            'txt_cond':txt_cond
        }

        neg_cond = {
            'image_cond':torch.zeros_like(cond['image_cond']), 
            'txt_cond': self.text_cond_model['null_cond'].to(self.device).repeat(cond['image_cond'].shape[0], 1, 1)
        }
        
        
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        sparse_structure: bool=False,
        convert_to_dense: bool=True
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        
        if sparse_structure:
            noise_dense = torch.ones(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
            nonzero_mask = noise_dense.to(torch.bool)  # Get the boolean mask of nonzero elements
            indices = torch.nonzero(nonzero_mask[:,0,...])  # Transpose to match the sparse_coo_tensor format
            coords = indices.to(torch.int32)  # Extract (batch, x, y, z)
            feats = torch.randn(coords.shape[0],flow_model.in_channels).to(torch.float32)
            noise = sp.SparseTensor(feats=feats, coords=coords.to(torch.int32)).to(self.device)
        else:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)

        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        if sparse_structure:
            if convert_to_dense:
                ss_feats = torch.zeros(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
                coords = z_s.coords
                mask = torch.zeros(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
                ss_feats[0,:,coords[:,1],coords[:,2],coords[:,3]] = z_s.feats.permute(1,0)
                mask[0,:,coords[:,1],coords[:,2],coords[:,3]] = True
                return ss_feats
        
        # Decode occupancy latent

        return z_s

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            slat_trellis = self.models['slat_decoder_gs'].compute_slats(slat)
            ret['mesh'] = self.models['slat_decoder_mesh'].decode_mesh(slat_trellis)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        return ret
    

    @torch.no_grad()
    def run(
        self,
        text: str,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        sample: bool = False,
        sparse_structure=False,
        convert_to_dense=True,
        save_cond_path=None
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(im, save_cond_path = save_cond_path) for im in images]

        cond_module_names = {'text_cond_model'}
        if len(images) > 0:
            cond_module_names.add('image_cond_model')
        with self.use_runtime_modules(cond_module_names):
            cond = self.get_cond([text], [images])

        torch.manual_seed(seed)
        with self.use_runtime_modules({'sparse_structure_flow_model'}):
            slat = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params,
                    sparse_structure=sparse_structure,convert_to_dense=convert_to_dense)

        del cond
        if self.offload_enabled and torch.cuda.is_available():
            torch.cuda.empty_cache()

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std.view(*std.shape,1,1,1).repeat(1,1,self.resolution,self.resolution,self.resolution) + mean.view(*mean.shape,1,1,1).repeat(1,1,self.resolution,self.resolution,self.resolution)
        if sample:
            return slat

        decoder_module_names = {'slat_decoder_gs'}
        if 'mesh' in formats:
            decoder_module_names.add('slat_decoder_mesh')
        with self.use_runtime_modules(decoder_module_names):
            return self.decode_slat(slat, formats)
