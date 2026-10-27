"""Frozen protocol, provenance and deterministic checkpoint utilities."""

import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "stage2-backbone-controlled-v2"
DATA_REVISION = "a833a3a5944148cb9815ccbd1f05aced50f6c464"
PRESETS = {
    "R": {"lr": 1e-4, "batch_size": 8, "epochs": 50, "smooth_l1_beta": 0.02},
    "M": {"lr": 3e-4, "batch_size": 64, "epochs": 100, "smooth_l1_beta": 1 / 154},
}
PACKAGES = ("torch", "torchvision", "numpy", "opencv-python", "zxing-cpp", "pyzbar", "wandb")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def source_hashes():
    paths = list((ROOT / "scripts/controlled_geometry").glob("*.py"))
    paths += [ROOT / "scripts/train_geometry.py", ROOT / "scripts/evaluate_stage2_v1.py"]
    paths += [ROOT / "scripts/label_recipes.py"]
    for name in ("ai/geometry", "data", "sw/rectify", "sw/decoding", "sw/pipeline"):
        paths += list((ROOT / "wemeet" / name).rglob("*.py"))
    paths += [ROOT / "wemeet/schemas.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def configure(seed=42, threads=2, device="cpu"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    # PyTorch 2.8 has no deterministic CUDA AdaptiveAvgPool2d backward.
    # Keep the native common head and disclose warnings instead of failing every
    # GPU training run. CPU checks remain strict; CUDA bitwise replay is not claimed.
    torch.use_deterministic_algorithms(True, warn_only=str(device).startswith("cuda"))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def capture_rng(generator, device="cpu"):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "loader": generator.get_state(),
        "cuda": [torch.cuda.get_rng_state(device)] if str(device).startswith("cuda") else [],
    }


def restore_rng(state, generator, device="cpu"):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if state["cuda"]:
        if not str(device).startswith("cuda"):
            raise ValueError("CUDA checkpoint requires CUDA resume")
        torch.cuda.set_rng_state(state["cuda"][0].cpu(), device)


def validate_lock(lock_path, verify_data=True):
    lock = read_json(lock_path)
    if lock["protocol"] != PROTOCOL:
        raise ValueError("Unexpected protocol")
    if lock["packages"] != package_versions():
        raise ValueError("Package versions changed after preparation")
    if lock["sources"] != source_hashes():
        raise ValueError("Source changed after preparation; create a new prepared protocol")
    paths = {lock["split_manifest"]: lock["split_sha256"]}
    paths[lock["d0_manifest"]] = lock["d0_manifest_sha256"]
    paths[lock["source_archive"]] = lock["source_archive_sha256"]
    paths.update({v["path"]: v["sha256"] for v in lock["pretrained"].values()})
    if verify_data:
        paths.update(lock["dataset_files"])
    for path, expected in paths.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return lock


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        str((value.shape, value.dtype.str)).encode() + value.tobytes()
    ).hexdigest()


def selection_key(summary, epoch):
    if summary["geometry"]["invalid_field_count"]:
        raise ValueError("Invalid geometry cannot select a checkpoint")
    stats = summary["paths"]["crop_policy_model"]
    if stats["RR_H"]["pct"] is None:
        raise ValueError("No validation H; cannot apply locked decode selection")
    return (
        -stats["RR_H"]["pct"],
        stats["FDR_H"]["pct"],
        summary["geometry"]["metrics"]["epe_mean_px"],
        epoch,
    )
