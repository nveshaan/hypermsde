"""MediCLIP (Zhang et al., MICCAI 2024): CLIP adapted to the K normal support images with the authors' anomaly
synthesis tasks, prompts, adapters and training loop (train.py), then scored as in their validate().

Image-level score: the maximum of the anomaly map. The authors keep the epoch with the best test AUROC; without
test labels the last epoch is kept. imgaug (used for one random rotation in the Perlin patch maker) does not run
on NumPy 2 and is replaced by the same rotation (bilinear, zero fill, same size); the removed alias
np.float is restored for their code.
"""
import contextlib
import math
import random
import sys
import types

import numpy as np
import PIL
import torch
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader

from baselines.common import choose_device, prepare_repo


class _Affine:
    """imgaug.augmenters.Affine(rotate=(low, high)): rotation about the centre, bilinear, constant 0 fill."""

    def __init__(self, rotate):
        self.rotate = rotate

    def __call__(self, image):
        angle = np.random.uniform(*self.rotate)
        return ndimage.rotate(image, angle, reshape=False, order=1, mode="constant", cval=0.0)


class MediCLIP:
    requires_support = True
    # The frozen CLIP's patch tokens of an image are the only inference on the images to score; the necker, adapter
    # and prompts trained on the support read them.
    encode_keys = ("config",)
    fingerprint_extra = {}

    def __init__(self, repo_url, repo_dir, config="config/chexpert.yaml", device="auto", epochs=None, seed=100,
                 **_):
        repo = prepare_repo(repo_url, repo_dir)
        # The repo's own open_clip fork, models, datasets and utils shadow this project's and pip's.
        for key in [k for k in sys.modules if k.split(".")[0] in ("open_clip", "models", "datasets", "utils")]:
            del sys.modules[key]
        if "float" not in np.__dict__:  # the synthesis code uses np.float, removed in NumPy 2
            np.float = float
        augmenters = types.ModuleType("imgaug.augmenters")
        augmenters.Affine = _Affine
        sys.modules["imgaug"] = types.ModuleType("imgaug")
        sys.modules["imgaug.augmenters"] = augmenters
        with contextlib.chdir(repo):
            import open_clip
            from datasets.dataset import TrainDataset
            from models.Adapter import Adapter
            from models.CoOp import PromptMaker
            from models.MapMaker import MapMaker
            from models.Necker import Necker
            from utils.losses import BinaryDiceLoss, FocalLoss
        self.modules = types.SimpleNamespace(TrainDataset=TrainDataset, Adapter=Adapter, PromptMaker=PromptMaker,
                                             MapMaker=MapMaker, Necker=Necker, FocalLoss=FocalLoss,
                                             BinaryDiceLoss=BinaryDiceLoss)
        with open(repo / config) as f:
            self.cfg = types.SimpleNamespace(**yaml.safe_load(f))
        if epochs is not None:
            self.cfg.epoch = epochs
        self.seed = seed
        self.device = choose_device(device)
        self.fingerprint_extra = {"model": self.cfg.model_name, "image_size": self.cfg.image_size,
                                  "layers_out": list(self.cfg.layers_out)}

        self.clip, self.preprocess, model_cfg = open_clip.create_model_and_transforms(
            self.cfg.model_name, self.cfg.image_size, device=self.device)
        for param in self.clip.parameters():
            param.requires_grad_(False)
        # make_vision_takens_info of the authors' train.py
        with torch.no_grad():
            size = model_cfg["vision_cfg"]["image_size"]
            _, tokens = self.clip.encode_image(torch.ones((1, 3, size, size), device=self.device), self.cfg.layers_out)
        if len(tokens[0].shape) == 3:
            self.clip.token_size = [int(math.sqrt(t.shape[1] - 1)) for t in tokens]
            self.clip.token_c = [t.shape[-1] for t in tokens]
        else:
            self.clip.token_size = [t.shape[2] for t in tokens]
            self.clip.token_c = [t.shape[1] for t in tokens]
        self.clip.embed_dim = model_cfg["embed_dim"]
        self.embed_dim = model_cfg["embed_dim"]

    def _support_dataset(self, images):
        """The authors' TrainDataset over in-memory images (it reads `<source>/images/<filename>` otherwise)."""
        size = self.cfg.image_size
        gray = [np.array(img.resize((size, size), PIL.Image.Resampling.BILINEAR).convert("L")).astype(np.uint8)
                for img in images]

        class Support(self.modules.TrainDataset):
            def get_image_data(self):
                return [{"filename": str(i)} for i in range(len(gray))]

            def read_image(self, path):
                return gray[int(path.rsplit("/", 1)[-1])]

        return Support(args=self.cfg, source="", preprocess=self.preprocess)

    @torch.no_grad()
    def encode(self, images) -> list:
        """The tokens of the CLIP layers `layers_out` of every image, as a tuple of arrays."""
        size, items = self.cfg.image_size, []
        for i in range(0, len(images), self.cfg.batch_size):
            x = torch.stack([
                self.preprocess(img.convert("RGB").resize((size, size), PIL.Image.Resampling.BILINEAR))
                for img in images[i:i + self.cfg.batch_size]
            ]).to(self.device)
            _, tokens = self.clip.encode_image(x, out_layers=self.cfg.layers_out)
            tokens = [t.float().cpu().numpy() for t in tokens]
            items.extend(tuple(t[b] for t in tokens) for b in range(len(x)))
        return items

    def fit(self, images, encoded=None) -> "Fitted":
        m, cfg = self.modules, self.cfg
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        necker = m.Necker(clip_model=self.clip).to(self.device)
        adapter = m.Adapter(clip_model=self.clip, target=self.embed_dim).to(self.device)
        prompt_maker = m.PromptMaker(prompts=cfg.prompts, clip_model=self.clip, n_ctx=cfg.n_learnable_token,
                                     CSC=cfg.CSC, class_token_position=cfg.class_token_positions).to(self.device)
        map_maker = m.MapMaker(image_size=cfg.image_size).to(self.device)
        optimizer = torch.optim.Adam([
            {"params": prompt_maker.prompt_learner.parameters(), "lr": 0.001},
            {"params": adapter.parameters(), "lr": 0.001},
        ], lr=0.001, betas=(0.5, 0.999))
        loader = DataLoader(self._support_dataset(images), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        focal, dice = m.FocalLoss(), m.BinaryDiceLoss()

        # train_one_epoch of the authors' train.py, for every epoch
        adapter.train()
        prompt_maker.train()
        for _ in range(cfg.epoch):
            for batch in loader:
                # The synthesised masks are float64, which MPS does not support.
                x, mask = batch["image"].to(self.device), batch["mask"].float().to(self.device)
                with torch.no_grad():
                    _, tokens = self.clip.encode_image(x, out_layers=cfg.layers_out)
                    features = necker(tokens)
                adapted = adapter(features)
                anomaly_map = map_maker(adapted, prompt_maker(adapted))
                loss = focal(anomaly_map, mask) + dice(anomaly_map[:, 1, :, :], mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        adapter.eval()
        prompt_maker.eval()
        return Fitted(self, necker, adapter, prompt_maker, map_maker)


class Fitted:
    """MediCLIP trained on one support: the scorer of the encoded images."""

    def __init__(self, method, necker, adapter, prompt_maker, map_maker):
        self.method, self.necker, self.adapter = method, necker, adapter
        self.prompt_maker, self.map_maker = prompt_maker, map_maker

    @torch.no_grad()
    def score_encoded(self, items) -> np.ndarray:
        method, scores = self.method, []
        for i in range(0, len(items), method.cfg.batch_size):
            batch = items[i:i + method.cfg.batch_size]
            tokens = [torch.from_numpy(np.stack([item[layer] for item in batch])).to(method.device)
                      for layer in range(len(batch[0]))]
            adapted = self.adapter(self.necker(tokens))
            anomaly_map = self.map_maker(adapted, self.prompt_maker(adapted))[:, 1]
            scores.append(anomaly_map.flatten(1).max(1).values.float().cpu().numpy())
        return np.concatenate(scores)
