from typing import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from ....utils import dist_utils


class InterleaveConditionedMixin:
    """
    Mixin for text-conditioned models.
    
    Args:
        text_cond_model: The text conditioning model.
    """
    def __init__(self, *args, cond_model: str = 'openai/clip-vit-large-patch14', **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_model_name = cond_model
        self.cond_model = None     # the model is init lazily
        # self.use_qwen = 'qwen' in text_cond_model.lower()
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input images (color, shape, size, texture, objects) if provided and consider the user's possible text instruction to generate a coherent 3D object that meets the user's requirements.<|im_end|>\n<|im_start|>user\n<|vision_start|>{}<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 52

        self.prompt_template_encode_no_img = "<|im_start|>system\nDescribe the key features of the input images (color, shape, size, texture, objects) if provided and consider the user's possible text instruction to generate a coherent 3D object that meets the user's requirements.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # self.prompt_template_encode_start_idx = {
        #     "text":35,
        #     "image":,
        #     "interleave":
        # }
        
    def _init_interleave_cond_model(self):
        """
        Initialize the text/image conditioning model.
        """

        with dist_utils.local_master_first():

            processor = AutoProcessor.from_pretrained(self.cond_model_name)

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.cond_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                local_files_only=True
            )
            
        model.eval()
        model = model.cuda()

        self.cond_model = {
            'model': model,
            'processor': processor,
        }
        self.cond_model['null_cond'] = self.encode_cond([([], '')])

            
    @torch.no_grad()
    def encode_cond(self, input_cond: List) -> torch.Tensor:
        """
        Encode the text.
        """

        if self.cond_model is None:
            self._init_interleave_cond_model()
        txt = []
        input_images = []
        for cap, imgs in input_cond:
            input_images += imgs
            if len(imgs):
                txt.append(self.prompt_template_encode.format(len(imgs)*"<|image_pad|>",cap))
            else:
                txt.append(self.prompt_template_encode_no_img.format(cap))
        if len(input_images):
            model_inputs = self.cond_model['processor'](
                text=txt,
                images=input_images,
                max_length=768 + self.prompt_template_encode_start_idx,
                padding='max_length',
                return_tensors="pt",
                padding_side='left',
                truncation=True,
            ).to('cuda')

            encoder_hidden_states = self.cond_model['model'](
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )
        else:
            model_inputs = self.cond_model['processor'](
                text=txt,
                max_length=768 + self.prompt_template_encode_start_idx,
                padding='max_length',
                return_tensors="pt",
                padding_side='left',
                truncation=True,
            ).to('cuda')

            encoder_hidden_states = self.cond_model['model'](
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_hidden_states=True,
            )


        embeddings = encoder_hidden_states.hidden_states[-1][:,self.prompt_template_encode_start_idx:,:]
        
        return embeddings
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_cond(cond)
        kwargs['neg_cond'] = self.cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_cond(cond)
        kwargs['neg_cond'] = self.cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond