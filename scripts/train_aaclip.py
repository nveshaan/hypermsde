"""Train AA-CLIP's adapters with the authors' train.py (their scripts.sh: full shot on VisA), for baselines=aaclip.

The authors release no weights. This runs their training unchanged, except that the VisA root is set from the
command line (their constants hardcode a path) and the device is the best available one (their code picks CUDA
or CPU only, so MPS is used on a Mac).

Example: python scripts/train_aaclip.py --visa /path/to/VisA_20220922
Extra arguments go to the authors' train.py, e.g. --image_epoch 10. Checkpoints land in the directory that
configs/baselines/aaclip.yaml points `checkpoint` at (default .third_party/AA-CLIP/ckpt/visa_full).
"""
import argparse
import contextlib
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from baselines.aaclip import prepare  # noqa: E402
from baselines.common import choose_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visa", required=True, help="VisA root (the folder holding candle/, capsules/, ...)")
    parser.add_argument("--device", default="auto")
    args, extra = parser.parse_known_args()

    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "baselines" / "aaclip.yaml")
    repo = prepare(cfg.repo_url, OmegaConf.to_container(cfg, resolve=True)["repo_dir"])
    save_path = Path(cfg.checkpoint)
    save_path = save_path if save_path.is_absolute() else repo / save_path

    from dataset import constants
    constants.DATA_PATH["VisA"] = str(Path(args.visa).resolve())  # the same dict the dataset code reads

    import train
    device = choose_device(args.device)
    # train.main does torch.device("cuda:0" if torch.cuda.is_available() else "cpu"): hand it `device` instead.
    proxy = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
    proxy.device = lambda *_args, **_kwargs: torch.device(device)
    train.torch = proxy
    if not device.startswith("cuda"):
        # Their setup_seed demands deterministic algorithms, which MPS lacks for index_put (used in backward): warn.
        deterministic = torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms = lambda mode, *, warn_only=False: deterministic(mode, warn_only=True)

    sys.argv = ["train.py", "--training_mode", "full_shot", "--dataset", "VisA", "--save_path", str(save_path),
                *extra]
    with contextlib.chdir(repo):  # metadata paths are relative to the repo
        train.main()


if __name__ == "__main__":
    main()
