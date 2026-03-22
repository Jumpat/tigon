from typing import *


class AlternateGuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """
    
    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if getattr(self, 'alternate', None) is None:
            self.alternate = True

        model.alternate = self.alternate
            
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            self.alternate = not self.alternate
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            self.alternate = not self.alternate
            return super()._inference_model(model, x_t, t, cond, **kwargs)
