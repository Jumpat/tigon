from typing import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from transformers import AutoTokenizer, CLIPTextModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor

from ....utils import dist_utils


class TextConditionedMixin:
    """
    Mixin for text-conditioned models.
    
    Args:
        text_cond_model: The text conditioning model.
    """
    def __init__(self, *args, text_cond_model: str = 'openai/clip-vit-large-patch14', **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cond_model_name = text_cond_model
        self.text_cond_model = None     # the model is init lazily
        self.use_qwen = 'qwen' in text_cond_model.lower()
        self.qwen_prompt_template = "<|im_start|>system\nDescribe the 3D object by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 35
        
    def _init_text_cond_model(self):
        """
        Initialize the text conditioning model.
        """
        # load model
        if 'clip' in self.text_cond_model_name:
            with dist_utils.local_master_first():
                model = CLIPTextModel.from_pretrained(self.text_cond_model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.text_cond_model_name)
            model.eval()
            model = model.cuda()
            self.text_cond_model = {
                'model': model,
                'tokenizer': tokenizer,
            }
            self.text_cond_model['null_cond'] = self.encode_text([''])
        elif 'qwen' in self.text_cond_model_name.lower():

            with dist_utils.local_master_first():

                processor = AutoProcessor.from_pretrained(self.text_cond_model_name)
                tokenizer = processor.tokenizer

                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.text_cond_model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    local_files_only=True
                )
                
            model.eval()
            model = model.cuda()
            self.text_cond_model = {
                'model': model,
                'tokenizer': tokenizer,
            }
            self.text_cond_model['null_cond'] = self.encode_text([''])

            
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
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_text(cond)
        kwargs['neg_cond'] = self.text_cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_text(cond)
        kwargs['neg_cond'] = self.text_cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond
