"""Bayes-PFL (Qu et al., CVPR 2025), zero-shot inference with the authors' released weights.

Image-level score as in the authors' test.py: the abnormal text probability averaged over the prompt bank and the
Monte Carlo samples of the prompt flows (stage 2).
"""
import contextlib
import json
import types

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from baselines.common import (choose_device, download_openai_clip, prepare_repo, redirect_cuda, select_checkpoint,
                              use_package)


class BayesPFL:
    def __init__(self, checkpoint, checkpoints, repo_url, repo_dir, class_name="object", device="auto",
                 image_size=518, features_list=(6, 12, 18, 24), prompt_context_len=5, prompt_num=3,
                 prompt_state_len=5, sample_num=10, num_flows=10, seed=333, **_):
        repo = prepare_repo(repo_url, repo_dir)
        use_package("models", repo / "models")  # no __init__.py, and this project has its own `models`
        with contextlib.chdir(repo):  # the tokenizer opens its vocabulary relative to the repo
            from models.model_CLIP import Load_CLIP, tokenize
            from models.VPB import Context_Prompting, TextEncoder

        torch.manual_seed(seed)
        self.device = choose_device(device)
        self.class_name = class_name
        with open(repo / "open_clip_local" / "model_configs" / "ViT-L-14-336.json") as f:
            cfg = json.load(f)
        self.args = args = types.SimpleNamespace(
            image_size=image_size, features_list=list(features_list), prompt_context_len=prompt_context_len,
            prompt_num=prompt_num, prompt_state_len=prompt_state_len, sample_num=sample_num, num_flows=num_flows,
            vision_width=cfg["vision_cfg"]["width"], text_width=cfg["text_cfg"]["width"], embed_dim=cfg["embed_dim"],
        )
        redirect_cuda(self.device)  # the flows and the vision transformer allocate with .cuda() / .to("cuda")

        clip_path = download_openai_clip("ViT-L/14@336px", repo.parent / "clip_cache")
        self.clip, _, _ = Load_CLIP(image_size, str(clip_path), device=self.device)
        self.clip.to(self.device).eval()
        self.tokenizer = tokenize
        self.text_encoder = TextEncoder(self.clip, args)
        self.model = Context_Prompting(args=args).to(self.device).eval()

        path = repo / "bayes_weight" / f"train_{checkpoint}.pth"
        if not path.is_file():
            import gdown
            path.parent.mkdir(exist_ok=True)
            gdown.download(id=select_checkpoint(checkpoint, checkpoints, "Bayes-PFL"), output=str(path), quiet=False)
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["MyModel"], strict=True)

        self.preprocess = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def _score_one(self, image: Image.Image) -> float:
        # Batch size 1: the authors' score pairs normal/abnormal logits along the batch dimension.
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features, _, patch_tokens = self.clip.encode_image(x, self.args.features_list)
        text, _ = self.model.forward_ensemble(self.text_encoder, image_features, patch_tokens, [self.class_name],
                                              self.device, self.tokenizer, mode="test")
        pro_img, _ = self.model(text, image_features, patch_tokens, stage=2, mode="test")
        pro_img = pro_img.squeeze(2)
        r = self.args.prompt_num * self.args.sample_num
        probs = torch.stack([pro_img[0, :r], pro_img[0, r:]], dim=1).softmax(-1)[:, 1]
        return float(probs.mean())

    def score(self, images) -> np.ndarray:
        return np.asarray([self._score_one(image) for image in images])
