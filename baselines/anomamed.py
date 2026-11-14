"""AnomaMed (Niaz et al., IEEE Access 2026), reimplemented from the paper (no code is released), Algorithm 1:

1. AABM: a pre-trained DDPM perturbs each normal support image (forward diffusion to step t, then the reverse
   process with its variance scaled by 1 + s) into an anomaly source P, blended into the image through a Perlin
   mask M restricted to the foreground (adaptive threshold): A = (1 - M) I + (1 - delta) M I + delta M P.
2. Prompts [V1..V12][CLS], CLS in {healthy, disease}; the learnable context tokens are refined by multi-head
   self-attention before CLIP's text encoder.
3. Fusion modules Psi_j (1x1 conv, 3x3 conv, ReLU) on the patch tokens of CLIP ViT-L/14 layers 12, 20, 24;
   softmax over cosine similarity / tau with the normal and abnormal prompt features (Eq. 4), upsampled and averaged.
4. Focal([S_n, S_a], Y) + Dice(S_a, Y), 200 epochs, batch 8, 224 x 224, seed 100 (Table 1).
Image-level score: max(S_a).

Settings the paper does not give are in configs/baselines/anomamed.yaml, marked "not in paper".
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from baselines.common import choose_device

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def perlin_2d(shape, res, rng):
    """Perlin noise of `shape` with `res` periods per axis (as in DRAEM's rand_perlin_2d_np)."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * math.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile(slice1, slice2):
        return np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)

    def dot(grad, shift):
        return (np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]), -1)
                * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile([0, -1], [1, None]), [0, -1])
    n11 = dot(tile([1, None], [1, None]), [-1, -1])
    t = 6 * grid ** 5 - 15 * grid ** 4 + 10 * grid ** 3
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return math.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def otsu(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return float(np.argmax(between))


def focal_loss(probs, target, gamma=2.0, eps=1e-6):
    """Focal loss on the two-channel probability map [S_n, S_a] against the binary mask."""
    p_true = torch.where(target.bool(), probs[:, 1], probs[:, 0]).clamp(eps, 1.0)
    return (-(1 - p_true) ** gamma * p_true.log()).mean()


def dice_loss(prob, target, eps=1.0):
    intersection = (prob * target).flatten(1).sum(1)
    union = prob.flatten(1).sum(1) + target.flatten(1).sum(1)
    return (1 - (2 * intersection + eps) / (union + eps)).mean()


class PromptLearner(nn.Module):
    """[V1..VM][CLS] prompts whose context tokens are refined by multi-head self-attention (Sec. III-C)."""

    def __init__(self, clip_model, class_names, n_ctx, n_heads):
        super().__init__()
        import clip

        width = clip_model.ln_final.weight.shape[0]
        self.ctx = nn.Parameter(torch.randn(n_ctx, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, n_heads, batch_first=True)
        prefix = " ".join(["X"] * n_ctx)
        self.register_buffer("tokens", clip.tokenize([f"{prefix} {name}." for name in class_names]))
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokens).float()
        self.register_buffer("prefix", embedding[:, :1])          # SOS
        self.register_buffer("suffix", embedding[:, 1 + n_ctx:])  # CLS, '.', EOS, padding
        self.clip = [clip_model]  # not a submodule: frozen and not saved

    def forward(self):
        clip_model = self.clip[0]
        ctx = self.ctx.unsqueeze(0)
        refined, _ = self.attention(ctx, ctx, ctx)  # A = softmax(QK^T / sqrt(d)) V
        n = len(self.tokens)
        x = torch.cat([self.prefix, refined.expand(n, -1, -1), self.suffix], dim=1)  # p' = [prefix, A, suffix]
        x = x + clip_model.positional_embedding.float()
        x = clip_model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = clip_model.ln_final(x).float()
        x = x[torch.arange(n), self.tokens.argmax(dim=-1)] @ clip_model.text_projection.float()
        return F.normalize(x, dim=-1)  # (2, C): A_n, A_a


class AnomaMed:
    requires_support = True

    def __init__(self, clip_model="ViT-L/14", layers=(12, 20, 24), image_size=224, n_ctx=12, n_heads=8,
                 class_names=("healthy", "disease"), temperature=0.07, epochs=200, batch_size=8, lr=1e-3,
                 normal_fraction=0.5, ddpm="google/ddpm-celebahq-256", ddpm_t=250, strength=(0.5, 2.0),
                 opacity=(0.5, 1.0), perlin_threshold=0.5, perlin_max_scale=6, seed=100, device="auto", **_):
        import clip

        self.device = choose_device(device)
        self.clip, _ = clip.load(clip_model, device=self.device, jit=False)
        self.clip = self.clip.float().eval()
        for param in self.clip.parameters():
            param.requires_grad_(False)
        self.layers, self.image_size = list(layers), image_size
        self.n_ctx, self.n_heads, self.class_names = n_ctx, n_heads, list(class_names)
        self.temperature, self.epochs, self.batch_size, self.lr = temperature, epochs, batch_size, lr
        self.normal_fraction = normal_fraction
        self.ddpm_name, self.ddpm_t, self.strength, self.opacity = ddpm, ddpm_t, tuple(strength), tuple(opacity)
        self.perlin_threshold, self.perlin_max_scale = perlin_threshold, perlin_max_scale
        self.seed = seed
        self.normalize = T.Normalize(CLIP_MEAN, CLIP_STD)

        self._tokens = {}
        for layer in self.layers:  # outputs of the residual blocks, LND
            self.clip.visual.transformer.resblocks[layer - 1].register_forward_hook(
                lambda _m, _i, out, layer=layer: self._tokens.__setitem__(layer, out))

    # ---- AABM ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _ddpm_sources(self, images, generator):
        """Anomaly source P per image: forward diffusion to step t, reverse with variance (1 + s) Sigma."""
        from diffusers import DDPMPipeline

        pipe = DDPMPipeline.from_pretrained(self.ddpm_name)
        unet, scheduler = pipe.unet.to(self.device).eval(), pipe.scheduler
        size = unet.config.sample_size
        alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        betas = scheduler.betas.to(self.device)
        sources = []
        for image in images:
            x0 = T.functional.to_tensor(image.convert("RGB").resize((size, size), Image.BICUBIC))
            x0 = (x0 * 2 - 1).unsqueeze(0).to(self.device)
            s = float(torch.empty(1).uniform_(*self.strength, generator=generator))
            t = self.ddpm_t
            noise = torch.randn(x0.shape, generator=generator).to(self.device)
            x = alphas_cumprod[t].sqrt() * x0 + (1 - alphas_cumprod[t]).sqrt() * noise
            for step in range(t, -1, -1):
                eps = unet(x, step).sample
                alpha, abar = 1 - betas[step], alphas_cumprod[step]
                mean = (x - betas[step] / (1 - abar).sqrt() * eps) / alpha.sqrt()
                if step == 0:
                    x = mean
                    break
                abar_prev = alphas_cumprod[step - 1]
                variance = betas[step] * (1 - abar_prev) / (1 - abar)  # posterior variance (fixed_small)
                z = torch.randn(x.shape, generator=generator).to(self.device)
                x = mean + ((1 + s) * variance).sqrt() * z
            source = ((x.clamp(-1, 1) + 1) / 2)[0].cpu()
            sources.append(T.functional.resize(source, [self.image_size, self.image_size], antialias=True))
        del pipe, unet
        return sources

    def _blend(self, image, source, rng):
        """A = (1 - M) I + (1 - delta) M I + delta M P with a Perlin mask on the foreground; returns (A, M)."""
        size = self.image_size
        scale = (2 ** rng.integers(0, self.perlin_max_scale), 2 ** rng.integers(0, self.perlin_max_scale))
        gray = (image.mean(0).numpy() * 255).astype(np.uint8)
        foreground = gray > otsu(gray)  # adaptive threshold: anomalies only on the anatomy
        for _ in range(20):
            noise = perlin_2d((size, size), scale, rng)
            mask = (noise > self.perlin_threshold) & foreground
            if mask.any():
                break
        mask = torch.from_numpy(mask.astype(np.float32))
        delta = rng.uniform(*self.opacity)
        blended = (1 - mask) * image + (1 - delta) * mask * image + delta * mask * source
        return blended, mask

    # ---- model ---------------------------------------------------------------------------------------------
    def _maps(self, x):
        """S_n, S_a (B, 2, H, W): Eq. 4 per layer, upsampled and averaged."""
        with torch.no_grad():
            self.clip.encode_image(self.normalize(x))
        text = self.prompts()  # (2, C)
        maps = []
        for layer, fuse in zip(self.layers, self.fusion):
            tokens = self._tokens[layer].permute(1, 0, 2)[:, 1:].float()  # NLD without CLS
            side = int(math.sqrt(tokens.shape[1]))
            z = fuse(tokens.transpose(1, 2).reshape(len(x), -1, side, side))
            z = F.normalize(z, dim=1)
            logits = torch.einsum("bchw,kc->bkhw", z, text) / self.temperature
            maps.append(F.interpolate(logits.softmax(1), size=self.image_size, mode="bilinear", align_corners=True))
        return torch.stack(maps).mean(0)

    def fit(self, images) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        width = self.clip.visual.transformer.width
        embed = self.clip.text_projection.shape[1]
        self.fusion = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, embed, 1), nn.Conv2d(embed, embed, 3, padding=1), nn.ReLU())
            for _ in self.layers
        ]).to(self.device)
        self.prompts = PromptLearner(self.clip, self.class_names, self.n_ctx, self.n_heads).to(self.device)
        optimizer = torch.optim.Adam(list(self.fusion.parameters()) + list(self.prompts.parameters()), lr=self.lr)

        normals = [T.functional.to_tensor(img.convert("RGB").resize((self.image_size, self.image_size),
                                                                     Image.BICUBIC)) for img in images]
        sources = self._ddpm_sources(images, generator)
        self.fusion.train()
        self.prompts.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(normals))
            for start in range(0, len(order), self.batch_size):
                batch, masks = [], []
                for i in order[start:start + self.batch_size]:
                    if rng.random() < self.normal_fraction:
                        batch.append(normals[i])
                        masks.append(torch.zeros(self.image_size, self.image_size))
                    else:
                        blended, mask = self._blend(normals[i], sources[i], rng)
                        batch.append(blended)
                        masks.append(mask)
                x, y = torch.stack(batch).to(self.device), torch.stack(masks).to(self.device)
                maps = self._maps(x)
                loss = focal_loss(maps, y) + dice_loss(maps[:, 1], y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.fusion.eval()
        self.prompts.eval()

    @torch.no_grad()
    def score(self, images) -> np.ndarray:
        scores = []
        for i in range(0, len(images), self.batch_size):
            x = torch.stack([T.functional.to_tensor(img.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BICUBIC)) for img in images[i:i + self.batch_size]])
            scores.append(self._maps(x.to(self.device))[:, 1].flatten(1).max(1).values.cpu().numpy())
        return np.concatenate(scores)
