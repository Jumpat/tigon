from typing import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch

from transformers import AutoTokenizer, CLIPTextModel, AutoProcessor, CLIPModel, AutoProcessor

from ....utils import dist_utils

from PIL import Image

import torch.nn.functional as F
from torchvision import transforms
import numpy as np

class InterleaveConditionedDINOCLIPMixin:
    """
    Mixin for text-conditioned models.
    
    Args:
        text_cond_model: The text conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'openai/clip-vit-large-patch14', txt_cond_model: str = 'openai/clip-vit-large-patch14', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.text_cond_model_name = txt_cond_model
        self.use_qwen = 'qwen' in txt_cond_model.lower()

        self.text_cond_model = None     # the model is init lazily
        self.image_cond_model = None     # the model is init lazily

        # self.use_qwen = 'qwen' in text_cond_model.lower()
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input images (color, shape, size, texture, objects) if provided and consider the user's possible text instruction to generate a coherent 3D object that meets the user's requirements.<|im_end|>\n<|im_start|>user\n<|vision_start|>{}<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 52

        self.prompt_template_encode_no_img = "<|im_start|>system\nDescribe the key features of the input images (color, shape, size, texture, objects) if provided and consider the user's possible text instruction to generate a coherent 3D object that meets the user's requirements.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # self.prompt_template_encode_start_idx = {
        #     "text":35,
        #     "image":,
        #     "interleave":
        # }

    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            if "dinov3" in self.image_cond_model_name:
                # facebook/dinov3-vith16plus-pretrain-lvd1689m

                self.dino_github = '/opt/huawei/explorer-env/dataset/trellis_ckpt/cache/torch/hub/facebookresearch_dinov3_main'
                self.dino_weight = "/opt/huawei/explorer-env/dataset/trellis_ckpt/cache/torch/hub/checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
                self.image_resolution = 592
            elif "dinov2" in self.image_cond_model_name:
                self.dino_github = '/opt/huawei/explorer-env/dataset/trellis_ckpt/cache/torch/hub/facebookresearch_dinov2_main'
                self.image_resolution = 518
            if "dinov2" in self.image_cond_model_name:
                dinov2_model = torch.hub.load(self.dino_github, self.image_cond_model_name,source="local",  pretrained=True)
            else:
                dinov2_model = torch.hub.load(self.dino_github, self.image_cond_model_name,source="local", weights=self.dino_weight, pretrained=True)
            # self.models['image_cond_model'] = dinov2_model
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # self.image_cond_model_transform = transform

            dinov2_model.eval().cuda()
            self.image_cond_model = {
                'model': dinov2_model,
                'transform': transform,
            }

    def _init_text_cond_model(self):
        """
        Initialize the text conditioning model.
        """
        # load model

        with dist_utils.local_master_first():
            model = CLIPTextModel.from_pretrained(self.text_cond_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.text_cond_model_name)

            clip_vl_model = CLIPModel.from_pretrained(self.text_cond_model_name)
            processor = AutoProcessor.from_pretrained(self.text_cond_model_name)
        model.eval()
        model = model.cuda()
        clip_vl_model.eval()
        clip_vl_model = clip_vl_model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
            'clip_model': clip_vl_model,
            'processor': processor,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])
        
    def _init_interleave_cond_model(self):
        """
        Initialize the text/image conditioning model.
        """
        self._init_image_cond_model()
        self._init_text_cond_model()

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and isinstance(text[0], str), "TextConditionedMixin only supports list of strings as cond"
        if self.text_cond_model is None:
            self._init_text_cond_model()
        if not self.use_qwen:
            encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
            tokens = encoding['input_ids'].cuda()
            embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        else:
            text = [self.qwen_prompt_template.format(e) for e in text]

            encoding = self.text_cond_model['tokenizer'](text, max_length=400 + self.prompt_template_encode_start_idx, padding='max_length', truncation=True, return_tensors='pt', padding_side='left')
            tokens = encoding.input_ids.cuda()
            attention_mask = encoding.attention_mask.cuda()
            encoder_hidden_states = self.text_cond_model['model'](
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            embeddings = encoder_hidden_states.hidden_states[-1][:,self.prompt_template_encode_start_idx:,:]
        
        return embeddings
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]], image_res=592) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """

        if self.image_cond_model is None:
            self._init_image_cond_model()
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image) or  all(isinstance(i, torch.Tensor) for i in image) , "Image list should be list of PIL images or torch.Tensor"
            if isinstance(image[0], Image.Image) :
                image = [i.resize((self.image_resolution, self.image_resolution), Image.LANCZOS) for i in image]
                image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
                image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
                image = torch.stack(image).to(self.device)
            else:
                image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model['transform'](image).to(self.device)
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']# [1,1374,1024]
        patchtokens = F.layer_norm(features, features.shape[-1:])

        return patchtokens    

    @torch.no_grad()
    def get_image_text_similarity(self, txt, images):
        txt_inputs = self.text_cond_model['processor'](text = txt, return_tensors='pt', padding=True).to(self.device)
        txt_features = self.text_cond_model['clip_model'].get_text_features(**txt_inputs)

        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        
        img_inputs =  self.text_cond_model['processor'](images=images, return_tensors="pt", do_rescale=False) .to(self.device)
        img_feats = self.text_cond_model['clip_model'].get_image_features(**img_inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                
        sims = (img_feats * txt_features).sum(dim=-1)  # [B]
        return sims
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        Must have images, text is optional
        """

        txt = []
        input_images = []
        for cap, imgs in cond:
            input_images += imgs
            if cap:
                txt.append(cap)
            else:
                txt.append('')

        
        # NLC
        txt_cond = self.encode_text(txt)
        image_cond = self.encode_image(input_images)

        
        cond = {
            'image_cond':image_cond, 
            'txt_cond':txt_cond
        }

        kwargs['neg_cond'] = {
            'image_cond':torch.zeros_like(cond['image_cond']), 
            'txt_cond': self.text_cond_model['null_cond'].repeat(cond['image_cond'].shape[0], 1, 1)
            # 'txt_cond': torch.cat([self.text_cond_model['null_cond'].repeat(cond['image_cond'].shape[0], 1, 1), 0.5 * torch.ones([cond['image_cond'].shape[0], 1,self.text_cond_model['null_cond'].shape[-1] ], device = cond['image_cond'].device)], dim = 1)
        }
        
        
        # kwargs['neg_cond'] = self.cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.get_cond(cond)
        # kwargs['neg_cond'] = self.cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        # kwargs['neg_cond'] = torch.zeros_like(image_cond)

        kwargs['neg_cond'] = {
            'image_cond':torch.zeros_like(cond['image_cond']), 
            'txt_cond': self.text_cond_model['null_cond'].repeat(cond['image_cond'].shape[0], 1, 1)
        }

        cond = super().get_inference_cond(cond, **kwargs)
        return cond


