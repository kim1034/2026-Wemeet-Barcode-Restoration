"""Reuse the exact split; optional content-checked CPU rendering cache."""

import fcntl
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.controlled_geometry.common import digest, object_digest, read_json, write_json
from scripts.train_geometry import IMAGENET_MEAN, IMAGENET_STD
from wemeet.data.synthesis import build, recipe_from_dict


def load_split(path):
    split = read_json(path)
    if set(split) != {"train", "validation"}:
        raise ValueError("Expected train and validation only")
    ids = []
    recipes = []
    for subset in ("train", "validation"):
        if not split[subset]:
            raise ValueError("Empty split")
        for record in split[subset]:
            if record["bucket"] not in ("L", "M", "H") or record["band"] not in (
                "target",
                "hard",
                "first_ok",
            ):
                raise ValueError("Unexpected training bucket/band")
            if record["recipe"]["symbology"].lower() != "code128":
                raise ValueError("Only Code128 is allowed")
            ids.append(record["id"])
        recipes.append({object_digest(r["recipe"]) for r in split[subset]})
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate or overlapping sample IDs")
    if recipes[0] & recipes[1]:
        raise ValueError("Exact recipes overlap train and validation")
    return split


def preprocess(gray):
    resized = cv2.resize(gray, (384, 160), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255).unsqueeze(0).repeat(3, 1, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class RenderCache:
    def __init__(self, root=None, fingerprint=None):
        self.root = Path(root) if root else None
        self.fingerprint = fingerprint or "uncached"
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def get(self, record):
        if not self.root:
            return self._get(record)
        key = object_digest({"renderer": self.fingerprint, "recipe": record["recipe"]})
        with (self.root / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._get(record)

    def _get(self, record):
        key = object_digest({"renderer": self.fingerprint, "recipe": record["recipe"]})
        path = self.root / f"{key}.npz" if self.root else None
        if path and path.exists():
            metadata = read_json(path.with_suffix(".json"))
            if metadata["key"] != key or digest(path) != metadata["sha256"]:
                raise ValueError("Render cache integrity mismatch")
            with np.load(path, allow_pickle=False) as value:
                result = {name: value[name].copy() for name in ("gray", "src", "dst", "flat_hw")}
        else:
            sample = build(recipe_from_dict(record["recipe"]))
            result = {
                "gray": sample.obs,
                "src": sample.src_norm,
                "dst": sample.dst_norm,
                "flat_hw": np.asarray([sample.h_flat, sample.w_flat], dtype=np.int64),
            }
            if path:
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **result)
                temp = path.with_name(f".{key}.{os.getpid()}.tmp")
                temp.write_bytes(buffer.getvalue())
                write_json(path.with_suffix(".json"), {"key": key, "sha256": digest(temp)})
                temp.replace(path)
        if result["gray"].dtype != np.uint8 or result["gray"].ndim != 2:
            raise ValueError("Invalid rendered image")
        if result["src"].shape != (48, 2) or not np.isfinite(result["src"]).all():
            raise ValueError("Invalid source coordinates")
        return result


class GeometryDataset(Dataset):
    def __init__(self, records, cache=None):
        self.records = records
        self.cache = cache or RenderCache()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sample = self.cache.get(record)
        h, w = sample["gray"].shape
        return (
            preprocess(sample["gray"]),
            torch.from_numpy(sample["src"].astype(np.float32)),
            torch.tensor([w, h], dtype=torch.float32),
            record["id"],
        )


def validation_samples(records, cache=None):
    cache = cache or RenderCache()
    for record in records:
        item = cache.get(record)
        row = {**record["recipe"], "band": record["band"], "bucket": record["bucket"]}
        row.update(h_flat=int(item["flat_hw"][0]), w_flat=int(item["flat_hw"][1]))
        yield f"validation/{record['id']}", row, item["gray"], item["src"], item["dst"]


def benchmark_samples(root):
    import json

    for bucket in ("low", "mid", "high"):
        directory = Path(root) / "eval" / bucket
        rows = [
            json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()
        ]
        if len(rows) != 500:
            raise ValueError("Benchmark must have 500 samples per bucket")
        for row in rows:
            gray = cv2.imread(str(directory / row["file"]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError("Unreadable benchmark image")
            with np.load(directory / row["npz"], allow_pickle=False) as values:
                src, dst = values["src_norm"].copy(), values["dst_norm"].copy()
            yield f"benchmark/{bucket}/{row['file']}", row, gray, src, dst
