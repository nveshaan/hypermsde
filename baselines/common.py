"""Shared helpers for the inference-only baselines: device choice, vendored repos and checkpoints."""
import gc
import os
import subprocess
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def choose_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def free_memory(device: str) -> None:
    """Return the memory of models that were just deleted to the device."""
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device.startswith("mps"):
        torch.mps.empty_cache()


def prepare_repo(url: str, repo_dir: str) -> Path:
    """Clone the authors' repo into `repo_dir` if missing and put it first on `sys.path`."""
    if "pkg_resources" not in sys.modules:  # the repos do `from pkg_resources import packaging`, gone from new setuptools
        import packaging.version
        sys.modules["pkg_resources"] = types.SimpleNamespace(packaging=packaging)
    path = Path(repo_dir)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if not path.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def use_package(name: str, path: Path) -> None:
    """Make `import name` resolve to the folder `path` of a vendored repo. Needed when the repo's folder has no
    __init__.py (a namespace package) or shares its name with a package of this project (models, datasets)."""
    for key in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
        del sys.modules[key]
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def redirect_cuda(device: str) -> None:
    """Send the hardcoded `.cuda()` / `.to("cuda")` calls of a vendored repo to `device` when it is not CUDA."""
    if device.startswith("cuda") or getattr(torch.Tensor, "_cuda_redirect", False):
        return
    to = torch.Tensor.to
    # MPS has no float64: tensors sent there from a "cuda" call become float32.
    narrow = (lambda t: t.float() if t.dtype == torch.float64 else t) if device.startswith("mps") else (lambda t: t)

    def redirected_to(self, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).startswith("cuda"):
            args = (device, *args[1:])
            self = narrow(self)
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = device
            self = narrow(self)
        return to(self, *args, **kwargs)

    torch.Tensor.to = redirected_to
    torch.Tensor.cuda = lambda self, *args, **kwargs: to(narrow(self), device)
    torch.Tensor._cuda_redirect = True


def download_openai_clip(name: str, root: Path) -> Path:
    """Path of the OpenAI CLIP checkpoint `name` (e.g. ViT-L/14@336px), downloaded into `root` if missing."""
    import clip
    return Path(clip.clip._download(clip.clip._MODELS[name], str(root)))


def select_checkpoint(checkpoint: str, checkpoints: dict, method: str):
    """The entry of `checkpoints` named `checkpoint`; fails clearly if the authors released none."""
    if checkpoint not in checkpoints:
        raise ValueError(f"{method}: unknown checkpoint {checkpoint!r}, choose from {list(checkpoints)}")
    if checkpoints[checkpoint] is None:
        available = [key for key, value in checkpoints.items() if value is not None]
        raise ValueError(f"{method}: no released {checkpoint!r} checkpoint, available: {available}")
    return checkpoints[checkpoint]
