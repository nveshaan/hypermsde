"""DPMM (Schulthess & Konukoglu, MICCAI 2025): a Dirichlet process mixture fit to DINOv2 patch embeddings of the
normal support images, with the authors' DPMM implementation and training schedule.

The authors fit on the full normal training set and keep the epoch with the lowest validation loss; here it is fit
on the K support images and the last epoch is kept (no validation set in the few-shot protocol). Their code scores
pixels only. The patch score is theirs (cosine distance to the nearest component mean with weight > 1e-6); the
image-level score is that of AnomalyDINO, which their code builds on: the mean of the top 1% patch scores.
"""
import importlib.util
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from baselines.common import choose_device, prepare_repo


def memory_efficient(base):
    """The authors' DPMM with the same math for diagonal covariances, without their (N, K, D) and (N, D, D)
    intermediates (gigabytes for a few thousand patches)."""
    from torch.special import digamma

    class Mixture(base):
        def get_weighted_log_prob(self, data, weight_threshold=0.0):
            # log(pi_k) + log N(x | mu_k, diag(var_k)), expanded into matrix products.
            x = data[:, 0, :]
            log_pi = self.calculate_log_pi()
            mask = torch.exp(log_pi) > weight_threshold
            mean, var = self.mean[mask], self.cov[mask].diagonal(dim1=1, dim2=2)
            precision = 1 / var
            mahalanobis = (x ** 2) @ precision.T - 2 * x @ (mean * precision).T + (mean ** 2 * precision).sum(1)
            log_prob = -0.5 * (mahalanobis + torch.log(var).sum(1) + self.D * np.log(2 * np.pi))
            out = torch.full((x.shape[0], self.K), -torch.inf, device=self.device)
            out[:, mask] = log_pi[mask] + log_prob
            return out

        def m_step(self, data, resp):
            # The authors' m_step; only the covariance statistic is computed per component instead of through
            # the (N, 1, D, D) outer products.
            n_samples = data.shape[0]
            x = data[:, 0, :]
            step_size = self.get_step_size()
            self.resp_stat *= (1 - step_size)
            self.mean_stat *= (1 - step_size)
            self.cov_stat *= (1 - step_size)
            self.resp_stat += step_size * resp.mean(0)
            self.mean_stat += step_size * (resp.T @ x) / n_samples
            idxs = torch.arange(self.D)
            self.cov_stat[:, idxs, idxs] += step_size * (resp.T @ x ** 2) / n_samples  # only the diagonal is read

            self.mean = self.mean_stat / (self.resp_stat[:, None] + self.eps)
            if self.cov_type == "diag":
                self.cov = self.estimate_covariance_diagonal()
            else:
                self.cov = self.estimate_covariance_spherical()
            self.compute_covariance_cholesky()

            double_sum = torch.cumsum(self.resp_stat.flip(0), 0).flip(0)
            double_sum[0] = 0
            double_sum = double_sum.roll(-1)
            self.v = self.resp_stat / (self.resp_stat + (self.alpha - 1) / n_samples + double_sum + self.eps)
            if not self.alpha_fixed:
                self.alpha = (self.K - 1) / (
                    digamma(n_samples * (self.resp_stat + double_sum) + self.alpha + 1) -
                    digamma(n_samples * double_sum + self.alpha)
                )[:-1].sum().item()

    return Mixture


class DPMM:
    requires_support = True
    # The patch tokens of an image are the only model inference; the mixture is fit on the support's tokens.
    encode_keys = ("backbone", "resolution")
    fingerprint_extra = {}

    def __init__(self, repo_url, repo_dir, device="auto", backbone="dinov2_vits14", resolution=448,
                 max_num_components=500, update_rate=0.2, schedule="ema", covariance_type="diag",
                 covariance_regularization=1e-6, epochs=40, batch_size=12, weight_threshold=1e-6, top_fraction=0.01,
                 seed=0, **_):
        repo = prepare_repo(repo_url, repo_dir)
        # Only the mixture itself: the rest of their src/ imports anomalib, torch_pca and wandb.
        spec = importlib.util.spec_from_file_location("dpmm_authors", repo / "src" / "DirichletProcessMixture" / "dpmm.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if covariance_type == "full":
            raise ValueError("DPMM: only diag/spherical covariances are supported (memory-efficient updates)")
        self.mixture_cls = memory_efficient(module.DPMM)

        self.device = choose_device(device)
        self.mixture_kwargs = dict(K=max_num_components, update_rate=update_rate, schedule=schedule,
                                   cov_type=covariance_type, reg_covar=covariance_regularization)
        self.epochs, self.batch_size = epochs, batch_size
        self.weight_threshold, self.top_fraction = weight_threshold, top_fraction
        self.seed = seed
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone).to(self.device).eval()
        self.transform = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def _features(self, image) -> torch.Tensor:
        """Unit-norm last-layer patch tokens (S, D), the image cropped to a multiple of the patch size."""
        x = self.transform(image).unsqueeze(0)
        p = self.backbone.patch_size
        x = x[:, :, :x.shape[2] - x.shape[2] % p, :x.shape[3] - x.shape[3] % p].to(self.device)
        tokens = self.backbone.get_intermediate_layers(x)[0]
        return F.normalize(tokens, dim=2)[0].float().cpu()

    def encode(self, images) -> list:
        return [self._features(image).numpy() for image in images]

    def fit(self, images, encoded) -> "Fitted":
        torch.manual_seed(self.seed)
        features = [torch.from_numpy(item) for item in encoded]
        dim = features[0].shape[1]
        mixture = self.mixture_cls(D=dim, device="cpu", **self.mixture_kwargs)
        order = list(range(len(features)))
        rng = random.Random(self.seed)
        for epoch in range(self.epochs):
            rng.shuffle(order)  # their DataLoader shuffles every epoch
            for start in range(0, len(order), self.batch_size):
                batch = torch.cat([features[i] for i in order[start:start + self.batch_size]])
                if epoch == 0 and start == 0:
                    mixture.initialize(batch)
                mixture.step(batch)
        return Fitted(mixture, self.weight_threshold, self.top_fraction)


class Fitted:
    """DPMM fitted on one support: the scorer of the encoded images."""

    def __init__(self, mixture, weight_threshold, top_fraction):
        self.mixture, self.weight_threshold, self.top_fraction = mixture, weight_threshold, top_fraction

    def score_encoded(self, items) -> np.ndarray:
        scores = []
        for item in items:
            distance = self.mixture.cosine_distance_to_nearest_cluster(torch.from_numpy(item),
                                                                       weight_threshold=self.weight_threshold)
            top = max(1, int(len(distance) * self.top_fraction))
            scores.append(float(distance.topk(top).values.mean()))
        return np.asarray(scores)
