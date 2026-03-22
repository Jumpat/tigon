import torch
import numpy as np
from ....utils.general_utils import dict_foreach
from ....pipelines import samplers


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance" 

        if self.p_uncond > 0:
            # randomly drop the class label
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                elif isinstance(cond, dict):
                    return cond['image_cond'].shape[0]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
                
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)
            
            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                    return torch.where(mask, neg_cond, cond)
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                elif isinstance(cond, dict):
                    return {k: select(cond[k], neg_cond[k], mask) for k in cond}
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
            
            mask = list(np.random.rand(B) < self.p_uncond)

            cond = select(cond, neg_cond, mask)

        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}
    
    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)



class InterleaveClassifierFreeGuidanceMixin:
    def __init__(self, *args, image_p_uncond: float = 0.5, text_p_uncond: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        # self.p_uncond = p_uncond

        self.image_p_uncond = image_p_uncond
        self.text_p_uncond = text_p_uncond


    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance" 


        if self.image_p_uncond > 0:
            # randomly drop the class label
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                elif isinstance(cond, dict):
                    return cond['image_cond'].shape[0]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
                
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)

            
            image_mask = list(np.random.rand(B) < self.image_p_uncond)
            text_mask = list(np.random.rand(B) < self.text_p_uncond)

            def select(cond, neg_cond, i_mask, t_mask, key = None):
                if isinstance(cond, torch.Tensor):
                    assert key
                    mask = i_mask if 'image' in key else t_mask
                    mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                    return torch.where(mask, neg_cond, cond)
                elif isinstance(cond, list):
                    assert key
                    mask = i_mask if 'image' in key else t_mask
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                elif isinstance(cond, dict):
                    return {k: select(cond[k], neg_cond[k], i_mask, t_mask, k) for k in cond}
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
            
            cond = select(cond, neg_cond, image_mask, text_mask)

        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}
    
    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)
