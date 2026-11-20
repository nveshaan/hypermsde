"""AA-CLIP (Ma et al., CVPR 2025), zero-shot inference with adapters trained by scripts/train_aaclip.py.

The authors release no weights; train them with their train.py on VisA (full shot, their scripts.sh).
Image-level score as in the authors' test.py for the Medical domain: the maximum of the anomaly map (patch
similarities summed over the adapted layers, Gaussian-blurred and upsampled).
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from baselines.common import choose_device, download_openai_clip, prepare_repo, use_package


def prepare(repo_url: str, repo_dir: str) -> Path:
    """Clone the repo, put the OpenAI CLIP weights where it expects them and stub its stray `import ipdb`."""
    repo = prepare_repo(repo_url, repo_dir)
    use_package("model", repo / "model")  # no __init__.py
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
    weights = repo / "model" / "ViT-L-14-336px.pt"
    if not weights.exists():
        weights.symlink_to(download_openai_clip("ViT-L/14@336px", repo.parent / "clip_cache"))
    return repo


class AACLIP:
    # The adapters change both towers: the encoding of an image is its score. The adapter file (trained locally,
    # possibly again into the same folder) is part of the key.
    encode_keys = ("checkpoint", "class_name", "image_size", "text_adapt_weight", "image_adapt_weight",
                   "text_adapt_until", "image_adapt_until", "relu")

    def __init__(self, checkpoint, repo_url, repo_dir, class_name="object", device="auto", image_size=518,
                 text_adapt_weight=0.1, image_adapt_weight=0.1, text_adapt_until=3, image_adapt_until=6,
                 relu=False, batch_size=8, seed=111, **_):
        repo = prepare(repo_url, repo_dir)
        from dataset.constants import CLASS_NAMES, REAL_NAMES
        from forward_utils import calculate_similarity_map, get_adapted_text_embedding
        from model.adapter import AdaptedCLIP
        from model.clip import create_model

        torch.manual_seed(seed)
        self.device = choose_device(device)
        self.image_size = image_size
        self.batch_size = batch_size
        self.similarity_map = calculate_similarity_map
        path = Path(checkpoint)
        path = path if path.is_absolute() else repo / path
        image_files = sorted(path.glob("image_adapter_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if not image_files:
            raise FileNotFoundError(f"AA-CLIP: no image_adapter_*.pth in {path}; train with scripts/train_aaclip.py")

        clip_model = create_model(model_name="ViT-L-14-336", img_size=image_size, device=self.device,
                                  pretrained="openai", require_pretrained=True).eval()
        self.model = AdaptedCLIP(clip_model=clip_model, text_adapt_weight=text_adapt_weight,
                                 image_adapt_weight=image_adapt_weight, text_adapt_until=text_adapt_until,
                                 image_adapt_until=image_adapt_until, relu=relu).to(self.device).eval()
        text_file = path / "text_adapter.pth"
        if text_file.is_file():
            self.model.text_adapter.load_state_dict(torch.load(text_file, map_location="cpu")["text_adapter"])
        # The last epoch's image adapter (test.py evaluates every saved epoch).
        self.fingerprint_extra = {"adapter": image_files[-1].name, "trained": image_files[-1].stat().st_mtime_ns}
        self.model.image_adapter.load_state_dict(torch.load(image_files[-1], map_location="cpu")["image_adapter"])

        # Prompted with the modality's class name, registered as a one-class dataset of the authors' prompt code.
        CLASS_NAMES["delta"], REAL_NAMES["delta"] = [class_name], {class_name: class_name}
        with torch.no_grad():
            text_model = self.model if text_file.is_file() else clip_model
            self.text = get_adapted_text_embedding(text_model, "delta", self.device)[class_name]

        self.preprocess = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def encode(self, images) -> list:
        items = []
        for i in range(0, len(images), self.batch_size):
            x = torch.stack([self.preprocess(img) for img in images[i:i + self.batch_size]]).to(self.device)
            patch_features, _ = self.model(x)
            maps = [self.similarity_map(f, self.text, self.image_size, test=True, domain="Medical")
                    for f in patch_features]
            maps = torch.cat(maps, dim=1).sum(1)
            items.extend(maps.flatten(1).max(1).values.float().cpu().numpy().reshape(-1, 1))
        return items

    @staticmethod
    def score_encoded(items) -> np.ndarray:
        return np.concatenate(items)
