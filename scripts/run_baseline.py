"""Run an inference-only anomaly detection baseline on the raw images of a dataset and log the results to MLflow.

Example: python scripts/run_baseline.py baselines=adaclip modality=chest dataset=VinCXR
Few-shot: python scripts/run_baseline.py baselines=anoco modality=chest support=4 support_seed=0
Many supports in one pass (each image is encoded once): ... supports=[1,2,4,8,16] support_seeds=[0,1,2,3,4]
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Hydra 1.3 passes a lazy help object to argparse. Python 3.14 tightened
# argparse's help validation to require a string.
if sys.version_info >= (3, 14):
    _argparse_check_help = argparse.ArgumentParser._check_help

    def _hydra_argparse_check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return _argparse_check_help(self, action)

    argparse.ArgumentParser._check_help = _hydra_argparse_check_help

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.cache import EncodingCache, Encodings  # noqa: E402
from datasets.embeds import _balanced_cap_indices  # noqa: E402
from datasets.raw import MODALITY_DATASETS, RawDataset  # noqa: E402
from run_hypermsde import dataset_names, detection_metrics, select_support  # noqa: E402


def raw_dataset(cfg: DictConfig, name: str) -> RawDataset:
    """The raw images of `name` in the order the embeddings were generated in (see generate_embeddings.py)."""
    raw_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "datasets" / f"raw_{cfg.modality}.yaml")
    root = Path(cfg.data_root)
    raw_cfg.root = str(root if root.is_absolute() else PROJECT_ROOT / root)
    resolved = OmegaConf.to_container(OmegaConf.create({"data": {cfg.modality: raw_cfg}}), resolve=True)
    config = resolved["data"][cfg.modality]
    kwargs = {"seed": int(config.get("seed", 42))}
    for key in ("sample_strategy", "image_extensions"):
        if config.get(key) is not None:
            kwargs[key] = config[key]
    cls = MODALITY_DATASETS.get(cfg.modality.lower(), RawDataset)
    return cls(config["sources"][name], transform=None, **kwargs)


class Run:
    """One (K, support seed) evaluation of a dataset: its support, its evaluated samples and their scores."""

    def __init__(self, k, seed, keep, labels_keep):
        self.k, self.seed = k, seed
        self.support_index = None   # indices into the dataset
        pos = np.arange(len(keep))
        if k > 0:
            chosen, rest = select_support(labels_keep, k, seed)
            if chosen is None:
                self.skip = f"support={k} needs more normal samples (has {(labels_keep == 0).sum()})"
                return
            # Normal samples drawn as support are removed from the evaluated samples, as in run_hypermsde.py.
            self.support_index, pos = keep[chosen], rest
        self.pos, self.labels = pos, labels_keep[pos]
        self.y = (self.labels != 0).astype(int)
        n, n_anomalies = len(self.y), int(self.y.sum())
        self.skip = (f"needs both normal and anomalous samples (n={n}, anomalies={n_anomalies})"
                     if n_anomalies in (0, n) else None)
        self.mask = np.zeros(len(keep), dtype=bool)
        self.mask[pos] = True
        self.score = np.full(len(keep), np.nan)
        self.head = None


def encoding_settings(cfg: DictConfig, method, overrides: dict) -> dict:
    """The settings of the baseline that change its encodings (the part of its config named by `encode_keys`)."""
    params = {**OmegaConf.to_container(cfg.baselines, resolve=True), **overrides}
    return {**{key: params[key] for key in method.encode_keys if key in params}, **method.fingerprint_extra}


def run_dataset(cfg: DictConfig, name: str) -> None:
    root = Path(cfg.data_root)
    root = root if root.is_absolute() else PROJECT_ROOT / root
    all_labels = np.load(root / cfg.modality / name / "labels.npy")
    raw = raw_dataset(cfg, name)
    if not np.array_equal(np.asarray(raw.labels), all_labels):
        raise RuntimeError(f"{name}: raw images and labels.npy are out of sync ({len(raw)} vs {len(all_labels)})")
    # Same samples as run_hypermsde.py (SingleEmbedsDataset applies the cap exactly like this).
    keep = (_balanced_cap_indices(all_labels, cfg.cap, cfg.seed)
            if cfg.cap is not None and len(all_labels) > cfg.cap else np.arange(len(all_labels)))
    labels_keep = all_labels[keep]

    # Every (K, seed) listed is evaluated in one pass over the images: each is encoded once, whatever their number.
    ks = [int(k) for k in (cfg.supports if cfg.supports is not None else [cfg.support])]
    seeds = [int(s) for s in (cfg.support_seeds if cfg.support_seeds is not None else [cfg.support_seed])]
    if len(set(k > 0 for k in ks)) > 1:
        raise ValueError("supports mixes zero-shot (0) and few-shot (> 0) runs")
    runs = [Run(k, s, keep, labels_keep) for k in ks for s in (seeds if ks[0] > 0 else seeds[:1])]
    for run in runs:
        if run.skip:
            print(f"[skip] {name}: K={run.k} seed={run.seed}: {run.skip}")
    runs = [run for run in runs if not run.skip]
    if not runs:
        return

    overrides = {"class_name": cfg.class_names[cfg.modality]} if "class_name" in cfg.baselines else {}
    if "modality" in cfg.baselines:
        overrides["modality"] = cfg.modality
    method = hydra.utils.instantiate(cfg.baselines, **overrides)
    few_shot = getattr(method, "requires_support", False)
    if few_shot != (ks[0] > 0):
        raise ValueError(f"{cfg.baselines.name} is a {'few' if few_shot else 'zero'}-shot baseline, "
                         f"run it with {'support > 0' if few_shot else 'support=0'}")

    def load_images(indices):
        return [Image.open(raw.paths[i]).convert("RGB") for i in indices]

    cache = None
    if cfg.cache_dir is not None and cfg.baselines.get("cache_encodings", False):
        cache_root = Path(cfg.cache_dir)
        cache = EncodingCache(cache_root if cache_root.is_absolute() else PROJECT_ROOT / cache_root,
                              cfg.baselines.name, encoding_settings(cfg, method, overrides), cfg.modality, name)
    encodings = Encodings(method.encode, load_images, cache)

    def progress(stage):   # on stderr; "<stage> ..." so that the sweep's log filter (lines starting with a modality) skips it
        return tqdm(total=len(needed), desc=f"{stage} {cfg.baselines.name} {cfg.modality}/{name}", unit="img",
                    file=sys.stderr, mininterval=15, leave=False)

    needed = np.flatnonzero(np.any([run.mask for run in runs], axis=0))
    if few_shot:
        # The support images are encoded once, pinned in memory, and not encoded again when they are scored for
        # the runs they are not support of.
        encodings.pin(sorted({int(i) for run in runs for i in run.support_index}))
    if getattr(method, "encode_first", False):
        # Baselines with a separate model for the encoding: every image is encoded before the models of the next
        # stage are loaded, and the encoder is released, so the models never share the memory.
        with progress("encoding") as bar:
            for start in range(0, len(needed), cfg.encode_chunk):
                block = needed[start:start + cfg.encode_chunk]
                encodings.pin(keep[block])
                bar.update(len(block))
        method.release_encoder()
    if few_shot:
        fit_many = getattr(method, "fit_many", None) or (
            lambda images, encoded: [method.fit(i, e) for i, e in zip(images, encoded)])
        heads = fit_many([load_images(run.support_index) for run in runs],
                         [encodings.get(run.support_index) for run in runs])
        for run, head in zip(runs, heads):
            run.head = head
    else:
        runs[0].head = method

    with progress("scoring") as bar:
        for start in range(0, len(needed), cfg.encode_chunk):
            block = needed[start:start + cfg.encode_chunk]
            items = encodings.get(keep[block])
            for run in runs:
                chosen = np.flatnonzero(run.mask[block])
                if len(chosen):
                    run.score[block[chosen]] = run.head.score_encoded([items[j] for j in chosen])
            bar.update(len(block))
    for run in runs:
        log_run(cfg, name, keep, run)


def log_run(cfg: DictConfig, name: str, keep: np.ndarray, run: Run) -> None:
    labels, y, score = run.labels, run.y, run.score[run.pos]
    n, n_anomalies = len(y), int(y.sum())
    metrics = detection_metrics(y, score)
    subtypes = [int(c) for c in np.unique(labels) if c != 0]
    subtype_counts = {c: int((labels == c).sum()) for c in subtypes} if len(subtypes) > 1 else {}
    for c in subtype_counts:
        sub = (labels == 0) | (labels == c)
        metrics.update(detection_metrics((labels[sub] == c).astype(int), score[sub], prefix=f"subtype_{c}/"))

    label = cfg.baselines.name + ("" if cfg.baselines.get("checkpoint") is None else f"-{cfg.baselines.checkpoint}")
    if run.support_index is not None:
        label += f"-K{run.k}-s{run.seed}"
    run_cfg = OmegaConf.merge(cfg, {"support": run.k, "support_seed": run.seed, "supports": None,
                                    "support_seeds": None})
    with mlflow.start_run(run_name=f"{cfg.modality}-{name}-{label}"):
        mlflow.log_params({
            "dataset": name,
            "modality": cfg.modality,
            "method": cfg.baselines.name,
            "seed": cfg.seed,
            "support": run.k,
            "support_seed": run.seed,
            "n_samples": n,
            "n_anomalies": n_anomalies,
            **{f"n_subtype_{c}": count for c, count in subtype_counts.items()},
        })
        if cfg.baselines.get("checkpoint") is not None:
            mlflow.set_tag("checkpoint", cfg.baselines.checkpoint)
        mlflow.log_metrics(metrics, step=0)
        mlflow.log_dict(OmegaConf.to_container(run_cfg, resolve=True), "configs/config.yaml")
        mlflow.log_dict(OmegaConf.to_container(cfg.baselines, resolve=True), "configs/baseline.yaml")
        # Per-sample scores, with each sample's label and its index into the dataset's labels.npy (and, with
        # support, the support samples' indices).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.npz"
            extra = {"support_index": run.support_index} if run.support_index is not None else {}
            np.savez(path, score=score, label=labels, index=keep[run.pos], **extra)
            mlflow.log_artifact(str(path), "scores")
    print(f"{cfg.modality}/{name}/{label}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "/" not in k))
    for c, count in subtype_counts.items():
        print(f"  subtype {c} (n={count}): auroc={metrics[f'subtype_{c}/auroc']:.4f} "
              f"auroc_flipped={metrics[f'subtype_{c}/auroc_flipped']:.4f}")


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    tracking_uri = cfg.mlflow.tracking_uri
    if tracking_uri.startswith("sqlite:///") and not Path(tracking_uri[10:]).is_absolute():
        db_path = PROJECT_ROOT / tracking_uri[10:]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(cfg.mlflow.experiment_name) is None:
        artifact_root = PROJECT_ROOT / "experiments" / "artifacts"
        mlflow.create_experiment(cfg.mlflow.experiment_name, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    torch.manual_seed(cfg.seed)
    for name in dataset_names(cfg):
        run_dataset(cfg, name)


if __name__ == "__main__":
    main()
