"""Run a standard anomaly detector (cosine softmax with text, kNN, LOF, PCA) on the embeddings of a dataset and log
the results to MLflow. Same samples, support and metrics as scripts/run_msde.py, for the encoder comparison.

Example: python scripts/run_detector.py detector=lof modality=chest encoder=BiomedCLIP dataset=VinCXR support=4
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

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
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.embeds import SingleEmbedsDataset  # noqa: E402
from run_msde import dataset_names, detection_metrics, select_support  # noqa: E402


def run_dataset(cfg: DictConfig, name: str) -> None:
    root = Path(cfg.data_root)
    root = root if root.is_absolute() else PROJECT_ROOT / root
    data = SingleEmbedsDataset(
        root / cfg.modality / name, encoder=cfg.encoder, cap=cfg.cap, seed=cfg.seed, as_tensor=False
    )
    X, labels, indices = np.asarray(data.embeddings, dtype=np.float64), np.asarray(data.labels), data.indices
    support, support_index = None, None
    if cfg.support > 0:
        chosen, rest = select_support(labels, cfg.support, cfg.support_seed)
        if chosen is None:
            print(f"[skip] {name}: support={cfg.support} needs more normal samples (has {(labels == 0).sum()})")
            return
        # Normal samples drawn as support are removed from the evaluated points, as in run_msde.py.
        support, support_index = X[chosen], indices[chosen]
        X, labels, indices = X[rest], labels[rest], indices[rest]
    y = (labels != 0).astype(int)
    n, n_anomalies = len(y), int(y.sum())
    if n_anomalies == 0 or n_anomalies == n:
        print(f"[skip] {name}: needs both normal and anomalous samples (n={n}, anomalies={n_anomalies})")
        return

    kwargs = {}
    if cfg.detector.get("text"):
        text = np.load(root / cfg.modality / f"{cfg.encoder}_text.npz")
        kwargs["text"] = (text["normal"], text["abnormal"])
    score = np.asarray(hydra.utils.instantiate(cfg.detector.fn)(X, support, **kwargs), dtype=np.float64)

    metrics = detection_metrics(y, score)
    subtypes = [int(c) for c in np.unique(labels) if c != 0]
    subtype_counts = {c: int((labels == c).sum()) for c in subtypes} if len(subtypes) > 1 else {}
    for c in subtype_counts:
        keep = (labels == 0) | (labels == c)
        metrics.update(detection_metrics((labels[keep] == c).astype(int), score[keep], prefix=f"subtype_{c}/"))

    label = f"{cfg.encoder}-{cfg.detector.name}" + (f"-K{cfg.support}-s{cfg.support_seed}" if cfg.support else "")
    with mlflow.start_run(run_name=f"{cfg.modality}-{name}-{label}"):
        mlflow.log_params({
            "dataset": name,
            "modality": cfg.modality,
            "encoder": cfg.encoder,
            "detector": cfg.detector.name,
            "seed": cfg.seed,
            "support": cfg.support,
            "support_seed": cfg.support_seed,
            "n_samples": n,
            "n_anomalies": n_anomalies,
            **{f"n_subtype_{c}": count for c, count in subtype_counts.items()},
            **{f"detector.{key}": value for key, value in OmegaConf.to_container(cfg.detector.fn).items()
               if not key.startswith("_")},
        })
        mlflow.log_metrics(metrics, step=0)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "configs/config.yaml")
        # Per-sample scores, with each sample's label and its index into the dataset's labels.npy (and, with
        # support, the support samples' indices).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.npz"
            extra = {"support_index": support_index} if support_index is not None else {}
            np.savez(path, score=score, label=labels, index=indices, **extra)
            mlflow.log_artifact(str(path), "scores")
    print(f"{cfg.modality}/{name}/{label}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "/" not in k))
    for c, count in subtype_counts.items():
        print(f"  subtype {c} (n={count}): auroc={metrics[f'subtype_{c}/auroc']:.4f} "
              f"auroc_flipped={metrics[f'subtype_{c}/auroc_flipped']:.4f}")


@hydra.main(version_base=None, config_path="../configs", config_name="detector")
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
    for name in dataset_names(cfg):
        run_dataset(cfg, name)


if __name__ == "__main__":
    main()
