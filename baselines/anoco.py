"""ANoCo (Seo et al., CVPR 2026, arXiv 2605.28428), reimplemented from the paper and its supplement (no code is
released). Training-free: every query patch is pulled towards anchor-consistent normal patches of the K support
images by an anchored graph-Laplacian energy, solved in closed form, and the required drift is the anomaly score.

Paper settings: DINOv3-L/16, output of transformer block 18, 768 x 768 input, ImageNet normalisation, lambda = 1
(around the best value of their Fig. S1), anomaly map smoothed by a Gaussian (kernel 7, sigma 0.8), image score =
max. Not reproduced: the test-time augmentation of Sec. S1, whose aggregation ("entropy-based confidence
weighting") is not specified; the paper also reports running without it.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms.functional import gaussian_blur

from baselines.common import choose_device


class ANoCo:
    requires_support = True

    def __init__(self, backbone="facebook/dinov3-vitl16-pretrain-lvd1689m", layer=18, image_size=768, lam=1.0,
                 blur_kernel=7, blur_sigma=0.8, query_chunk=256, device="auto", **_):
        from transformers import AutoModel

        self.device = choose_device(device)
        self.model = AutoModel.from_pretrained(backbone).to(self.device).eval()
        self.layer, self.image_size, self.lam = layer, image_size, lam
        self.blur_kernel, self.blur_sigma = blur_kernel, blur_sigma
        self.query_chunk = query_chunk
        self.skip = 1 + getattr(self.model.config, "num_register_tokens", 0)  # CLS and register tokens
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.reference = None

    @torch.no_grad()
    def _patches(self, image) -> torch.Tensor:
        """(N, d) patch features: output of block `layer` (hidden_states[0] are the embeddings)."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        hidden = self.model(pixel_values=x, output_hidden_states=True).hidden_states[self.layer]
        return hidden[0, self.skip:].float()

    def fit(self, images) -> None:
        self.reference = torch.cat([self._patches(image) for image in images])  # F_r, (N_r, d)
        self.reference_unit = F.normalize(self.reference, dim=1)
        self.reference_norm = self.reference.norm(dim=1)

    @torch.no_grad()
    def _energy(self, query: torch.Tensor) -> torch.Tensor:
        """Non-conformity E_i (Eq. 10) of every query patch."""
        energies = []
        for q in query.split(self.query_chunk):
            s = F.normalize(q, dim=1) @ self.reference_unit.T                        # s_ij, (n, N_r)
            # Sec. 3.2: references sorted by s_ij; the first is the anchor j*(i). Keep the longest prefix with
            # a_ij = cos(f_j*, f_j) > s_ij* (Eq. 1).
            s_sorted, order = s.sort(dim=1, descending=True)
            anchor = order[:, 0]
            a_sorted = (self.reference_unit[anchor] @ self.reference_unit.T).gather(1, order)
            keep = torch.cumprod((a_sorted > s_sorted[:, :1]).to(s.dtype), dim=1)   # 1 on the prefix, then 0
            # Eq. 2: w_ij = s_ij * alpha_ij, alpha_ij = 2 |f_i| |f_j| / (|f_i|^2 + |f_j|^2); 0 outside N(i).
            qn = q.norm(dim=1, keepdim=True)
            rn = self.reference_norm[order]
            w = keep * s_sorted * (2 * qn * rn / (qn ** 2 + rn ** 2))
            # Eq. 9: f~_i = (lambda f_i - (L_qr F_r)_i) / (d_i + lambda), with L_qr = -W and d_i = sum_j w_ij.
            n_keep = int(keep.sum(1).max())
            w, idx = w[:, :n_keep], order[:, :n_keep]
            aggregated = torch.einsum("nk,nkd->nd", w, self.reference[idx])
            moved = (self.lam * q + aggregated) / (w.sum(1, keepdim=True) + self.lam)
            # Eq. 10: E_i = |f~_i - f_i|^2 (1 - cos(f~_i, f_i))
            energies.append(((moved - q) ** 2).sum(1) * (1 - F.cosine_similarity(moved, q, dim=1)))
        return torch.cat(energies)

    def score(self, images) -> np.ndarray:
        scores = []
        for image in images:
            energy = self._energy(self._patches(image))
            side = int(round(len(energy) ** 0.5))
            anomaly_map = F.interpolate(energy.reshape(1, 1, side, side), size=self.image_size, mode="bilinear",
                                        align_corners=False)
            anomaly_map = gaussian_blur(anomaly_map, self.blur_kernel, self.blur_sigma)
            scores.append(float(anomaly_map.max()))
        return np.asarray(scores)
