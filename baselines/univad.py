"""UniVAD (Gu et al., CVPR 2025), training-free few-shot inference with the authors' code.

As in the authors' pipeline, every image (support and query) is first segmented by GroundingDINO + SAM-HQ
(their segment_components.py, prompted per modality as in configs/class_histogram), and the K support images are
the references of UniVAD.setup. The masks are cached by image content, so they are computed once per image.
Image-level score: the model's pred_score.

Deviation: pydensecrf does not build on Python 3.14, so the dense CRF that refines the part heatmaps of the
MULTI-component branch returns its unary probabilities (the CRF's starting point) instead.
"""
import contextlib
import functools
import hashlib
import subprocess
import sys
import types
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from baselines.common import choose_device, prepare_repo, redirect_cuda

CHECKPOINTS = {
    "sam_hq_vit_h.pth": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
    "groundingdino_swint_ogc.pth":
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
}


def unary_only(image_tensor, output_logits):
    """Stand-in for utils.crf.dense_crf: the softmax probabilities the CRF starts from, at the image size."""
    warnings.warn("UniVAD: pydensecrf unavailable, CRF refinement skipped", stacklevel=2)
    H, W = image_tensor.shape[-2:]
    logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
    return F.softmax(logits.squeeze(0), dim=0).cpu().numpy()


def restore_extended_attention_mask() -> None:
    """GroundingDINO's BERT wrapper binds three BertModel mask helpers that transformers 5 dropped."""
    from transformers import BertModel
    if hasattr(BertModel, "get_head_mask"):
        return

    def get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
        dtype = dtype or self.dtype
        extended = attention_mask[:, None, None, :] if attention_mask.dim() == 2 else attention_mask[:, None, :, :]
        return (1.0 - extended.to(dtype)) * torch.finfo(dtype).min

    def invert_attention_mask(self, encoder_attention_mask):
        extended = encoder_attention_mask[:, None, None, :] if encoder_attention_mask.dim() == 2 \
            else encoder_attention_mask[:, None, :, :]
        return (1.0 - extended.to(self.dtype)) * torch.finfo(self.dtype).min

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError("head_mask is not used by GroundingDINO")
        return [None] * num_hidden_layers

    BertModel.get_extended_attention_mask = get_extended_attention_mask
    BertModel.invert_attention_mask = invert_attention_mask
    BertModel.get_head_mask = get_head_mask


@contextlib.contextmanager
def cuda_alias(device: str):
    """The authors' code picks `"cuda" if torch.cuda.is_available() else "cpu"` in places, which on a Mac would put
    those parts on the CPU next to MPS tensors. Report CUDA as available (redirect_cuda maps it to `device`)."""
    if device.startswith("cuda"):
        yield
        return
    available, load = torch.cuda.is_available, torch.load

    def load_to_device(*args, map_location=None, **kwargs):
        if str(map_location).startswith("cuda"):
            map_location = device
        return load(*args, map_location=map_location, **kwargs)

    torch.cuda.is_available, torch.load = (lambda: True), load_to_device
    try:
        yield
    finally:
        torch.cuda.is_available, torch.load = available, load


class ChunkedFunctional:
    """torch.nn.functional whose cosine_similarity runs in chunks: the authors' patch-to-reference comparison
    broadcasts to 2**32 elements, which MPS cannot index."""

    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8, chunk=64):
        if max(x1.shape[0], x2.shape[0]) <= chunk or x1.shape[0] != max(x1.shape[0], x2.shape[0]):
            return F.cosine_similarity(x1, x2, dim=dim, eps=eps)
        return torch.cat([F.cosine_similarity(x1[i:i + chunk], x2, dim=dim, eps=eps)
                          for i in range(0, x1.shape[0], chunk)])


class UniVAD:
    requires_support = True

    def __init__(self, repo_url, repo_dir, grounding, modality="chest", device="auto", image_size=448, seed=0, **_):
        repo = prepare_repo(repo_url, repo_dir)
        if not (repo / "models" / "dinov2" / "hubconf.py").exists():
            subprocess.run(["git", "-C", str(repo), "submodule", "update", "--init", "--depth", "1", "models/dinov2"],
                           check=True)
        for name, url in CHECKPOINTS.items():
            path = repo / "pretrained_ckpts" / name
            if not path.exists():
                torch.hub.download_url_to_file(url, str(path))
        # This project's `models` and `utils` would shadow the repo's.
        for key in [k for k in sys.modules if k.split(".")[0] in ("models", "utils")]:
            del sys.modules[key]
        sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
        # GroundingDINO imported as a package, without the CUDA extension of its setup.py (pure-PyTorch fallback).
        sys.path.insert(0, str(repo / "models" / "GroundingDINO"))
        for name in ("pydensecrf", "pydensecrf.densecrf", "pydensecrf.utils"):
            sys.modules.setdefault(name, types.ModuleType(name))

        self.device = choose_device(device)
        redirect_cuda(self.device)  # the authors' code calls .cuda() and .to("cuda") throughout
        self.repo, self.image_size = repo, image_size
        self.grounding = dict(grounding[modality])
        self.category = f"delta_{modality}"
        self.data_dir = repo / "data" / "delta" / modality
        self.data_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        restore_extended_attention_mask()
        with contextlib.chdir(repo), cuda_alias(self.device):
            import UniVAD as module
            import models.component_segmentaion as segmentation
            module.dense_crf = unary_only
            module.F = ChunkedFunctional()
            # Load GroundingDINO and SAM-HQ once instead of on every call of grounding_segmentation.
            segmentation.load_model = functools.cache(segmentation.load_model)
            sam = functools.cache(segmentation.sam_hq_model_registry["vit_h"])
            segmentation.sam_hq_model_registry = {**segmentation.sam_hq_model_registry, "vit_h": sam}
            self.segment = segmentation.grounding_segmentation
            self.model = module.UniVAD(image_size=image_size).to(self.device)
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def _paths(self, images) -> list[str]:
        """Each image saved under the repo's data/ (named by content) and segmented if not done yet. The returned
        paths are relative to the repo, where UniVAD looks for ./masks/<path after /data/>/grounding_mask.png."""
        paths, todo = [], []
        for image in images:
            name = hashlib.sha1(np.asarray(image).tobytes()).hexdigest()
            path = f"./data/delta/{self.data_dir.name}/{name}.png"
            if not (self.repo / "masks" / "delta" / self.data_dir.name / name / "grounding_mask.png").exists():
                image.save(self.repo / path)
                todo.append(path)
            paths.append(path)
        if todo:
            with contextlib.chdir(self.repo), cuda_alias(self.device):
                self.segment(todo, "./masks", self.grounding)
        return paths

    def fit(self, images) -> None:
        paths = self._paths(images)
        samples = torch.stack([self.transform(image) for image in images]).to(self.device)
        with contextlib.chdir(self.repo), cuda_alias(self.device):
            self.model.setup({"few_shot_samples": samples, "dataset_category": self.category, "image_path": paths})
        self._concatenate_leftovers()

    def _concatenate_leftovers(self) -> None:
        """UniVAD.setup concatenates each part's support features only for the part indices of the last support
        image, so a part seen only in earlier images stays a list of per-image tensors and breaks scoring. Finish
        that concatenation (an empty list means the part never occurred and is kept, as the scorer skips it)."""
        model = self.model
        for idx, feats in enumerate(model.normal_dino_part_patch_features):
            if isinstance(feats, list) and feats:
                model.normal_dino_part_patch_features[idx] = torch.cat(feats, dim=0)
        for layer in model.normal_clip_part_patch_features:
            for idx, feats in enumerate(layer):
                if isinstance(feats, list) and feats:
                    layer[idx] = torch.cat(feats, dim=0)

    @torch.no_grad()
    def score(self, images) -> np.ndarray:
        paths = self._paths(images)
        scores = []
        with contextlib.chdir(self.repo), cuda_alias(self.device):
            for image, path in zip(images, paths):
                x = self.transform(image).unsqueeze(0).to(self.device)
                scores.append(float(self.model(x, path)["pred_score"]))
        return np.asarray(scores)
