from typing import *
import torch

class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        if type(cfg_strength) == int or type(cfg_strength) == float:

            return (1 + cfg_strength) * pred - cfg_strength * neg_pred

        elif type(cfg_strength) == List:
            assert len(cfg_strength) == pred.shape[0]
            cfg_strength = torch.tensor(cfg_strength).to(pred.device)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred

        elif type(cfg_strength) == torch.Tensor:
            assert len(cfg_strength) == pred.shape[0]

            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        
        else:
            raise NotImplementedError