import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        processor_path = "./pickscore_ckpt/CLIP-model/"
        model_path = "./pickscore_ckpt/pickscore/"
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, prompt, images, num_views=4):
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
            do_rescale=False,
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        # Get embeddings, [BK * 4, 3, H, W] -> [BK * 4, dim] -> [BK, 4, dim]
        image_embs = self.model.get_image_features(**image_inputs)

        image_embs = image_embs.view(-1, num_views, image_embs.size(-1))

        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)


        # [BK, dim] -> [BK, 1, dim]
        text_embs = self.model.get_text_features(**text_inputs)

        text_embs = text_embs.unsqueeze(1)

        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs * image_embs).sum(dim = -1)
        # Normalize to [0, 1]
        scores = scores / 26
        # [BK, 4]
        return scores