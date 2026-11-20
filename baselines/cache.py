"""Per-image model outputs ("encodings") of the baselines, so a model is never run twice on the same image.

A baseline splits its work into ``encode(images)`` (the model inference on an image: independent of the support
and of anything trained on it) and a cheap head on the encodings (``score_encoded``). The runner keeps the
encodings of the support images in memory and, for baselines whose encodings are small, on disk too: an
`EncodingCache` holds one file per image, in a folder named by the baseline and a hash of every setting that
changes its encodings, so a rerun, another checkpoint or another (K, seed) reuses them.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def fingerprint(method: str, settings: dict) -> str:
    """Short hash of the settings that change a baseline's encodings."""
    blob = json.dumps({"method": method, **settings}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


class EncodingCache:
    """Encodings on disk under ``root/<method>-<fingerprint>/<modality>/<dataset>/<index>.npz``, keyed by the
    image's index in the dataset (the order of its labels.npy)."""

    def __init__(self, root, method: str, settings: dict, modality: str, dataset: str):
        key = fingerprint(method, settings)
        self.dir = Path(root) / f"{method}-{key}" / modality / dataset
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir.parent.parent / "settings.json"
        if not meta.exists():
            meta.write_text(json.dumps({"method": method, **settings}, indent=1, sort_keys=True, default=str))

    def _file(self, index: int) -> Path:
        return self.dir / f"{int(index)}.npz"

    def get(self, index: int):
        path = self._file(index)
        if not path.is_file():
            return None
        try:
            with np.load(path) as f:
                arrays = tuple(f[f"a{i}"] for i in range(int(f["n"])))
                return arrays[0] if bool(f["single"]) else arrays
        except Exception:   # a file cut short by an interrupted run is recomputed
            return None

    def put(self, index: int, item) -> None:
        single = isinstance(item, np.ndarray)
        arrays = (item,) if single else tuple(item)
        path = self._file(index)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, n=len(arrays), single=single, **{f"a{i}": a for i, a in enumerate(arrays)})
        os.replace(tmp, path)   # atomic: a reader never sees a half-written file


class Encodings:
    """The encodings of one dataset's images: in memory for the ones the runner pins (the support images), on disk
    when a cache is given, and computed by ``encode`` for the rest. ``encode(indices)`` runs the model on exactly
    the images that are in neither place."""

    def __init__(self, encode, load_images, cache: EncodingCache | None = None):
        self._encode, self._load_images, self.cache = encode, load_images, cache
        self.memory: dict[int, object] = {}

    def pin(self, indices) -> None:
        """Keep the encodings of `indices` in memory from now on."""
        self.get(indices, keep=True)

    def get(self, indices, keep: bool = False) -> list:
        indices = [int(i) for i in indices]
        found = {}
        for i in indices:
            item = self.memory.get(i)
            if item is None and self.cache is not None:
                item = self.cache.get(i)
            if item is not None:
                found[i] = item
        todo = [i for i in dict.fromkeys(indices) if i not in found]
        if todo:
            for i, item in zip(todo, self._encode(self._load_images(todo))):
                found[i] = item
                if self.cache is not None:
                    self.cache.put(i, item)
        if keep:
            self.memory.update({i: found[i] for i in indices})
        return [found[i] for i in indices]
