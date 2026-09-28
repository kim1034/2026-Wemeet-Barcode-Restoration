"""Package and optionally publish the ResNet18-C3 checkpoint to Hugging Face.

The repository keeps only this script and the model configuration. The binary
weights stay in the ``123metro/barcode-weights`` model repository. The upload
contains both the full reproducible checkpoint and a smaller deployment
``state_dict`` used by ``wemeet.ai.geometry``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_CHECKPOINT = Path("runs/restoration-resnet18-c3-v1/best.pt")
DEFAULT_STAGING = Path("downloads/geometry/resnet18-c3")
DEFAULT_REPO = "123metro/barcode-weights"
HF_SUBPATH = "geometry/resnet18-c3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="upload the prepared folder to Hugging Face after packaging",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _model_card(config: dict) -> str:
    metrics = config["metrics"]
    return f"""---
library_name: pytorch
tags:
- barcode
- geometry
- resnet18
- tps
---

# Wemeet ResNet18-C3 geometry model

This folder contains the Stage 2 geometry model for the Wemeet barcode
restoration pipeline. It predicts 48 normalized source control points for a
fixed 16×3 destination grid. It does not generate pixels; the SW TPS
rectifier consumes the coordinates.

## Files

- `model_state_dict.pt`: deployment state dictionary used by
  `wemeet.ai.geometry`.
- `best.pt`: full training checkpoint, including optimizer state and training
  metadata.
- `config.json`: architecture, preprocessing, dataset revision, and metrics.

## Input contract

- input: BGR `uint8` detected crop, converted to grayscale
- resize: 160×384 (height×width), `INTER_AREA`
- grayscale is repeated to three channels
- ImageNet mean/std normalization
- output: `(48, 2)` normalized `(x, y)` source points

## Training result

- checkpoint epoch: {config["checkpoint_epoch"]}
- validation mean control-point error: {metrics["control_point_mean_px"]:.4f} px
- validation control-point P95: {metrics["control_point_p95_px"]:.4f} px
- held-out exact-decode rescue rate: {metrics["rescue_rate"]:.4%} (synthetic target-only)
- held-out exact-decode misread rate: {metrics["misread_rate"]:.4%}

The held-out decode result is synthetic data only. Real detector crops and
photographed invoices still require a separate evaluation.
"""


def package(checkpoint: Path, staging_dir: Path) -> Path:
    import torch

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError("체크포인트에서 model state_dict를 찾지 못했습니다")

    staging_dir = staging_dir.expanduser().resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    state_path = staging_dir / "model_state_dict.pt"
    best_path = staging_dir / "best.pt"
    config_path = staging_dir / "config.json"
    card_path = staging_dir / "README.md"
    torch.save(state_dict, state_path)
    shutil.copy2(checkpoint, best_path)

    config = _jsonable(payload.get("config", {}))
    # The training checkpoint contains absolute paths from one GPU host. Keep
    # the reproducibility labels but never publish an internal filesystem path
    # in a public model repository.
    if "data_root" in config:
        config["data_root"] = "synthetic/v1"
    if "dataset_root" in config:
        config["dataset_root"] = "synthetic/v1"
    if "output_dir" in config:
        config["output_dir"] = "runs/restoration-resnet18-c3-v1"
    val = _jsonable(payload.get("val", {}))
    metadata = {
        "model_type": "ResNet18-C3",
        "architecture": "ResNet18 through layer3 + 1x1 adapter + 16x3 geometry head",
        "checkpoint_epoch": payload.get("epoch"),
        "input_hw": [160, 384],
        "control_grid": [16, 3],
        "output_shape": [48, 2],
        "preprocess": {
            "grayscale_to_three_channels": True,
            "resize_interpolation": "cv2.INTER_AREA",
            "imagenet_normalization": True,
        },
        "dataset_revision": "a833a3a5944148cb9815ccbd1f05aced50f6c464",
        "config": config,
        "metrics": {
            "control_point_mean_px": val.get("control_point_mean_px"),
            "control_point_rmse_px": val.get("control_point_rmse_px"),
            "control_point_p95_px": val.get("control_point_p95_px"),
            "rescue_rate": 1067 / 1500,
            "misread_rate": 7 / 1500,
        },
        "files": {
            "model_state_dict.pt_sha256": _sha256(state_path),
            "best.pt_sha256": _sha256(best_path),
        },
    }
    config_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    card_path.write_text(_model_card(metadata), encoding="utf-8")
    print(json.dumps({"staging_dir": str(staging_dir), **metadata["files"]}, indent=2))
    return staging_dir


def upload(staging_dir: Path, repo_id: str, revision: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(staging_dir),
        path_in_repo=HF_SUBPATH,
        revision=revision,
        commit_message="Publish ResNet18-C3 geometry checkpoint",
    )
    print(f"uploaded: https://huggingface.co/{repo_id}/tree/{revision}/{HF_SUBPATH}")


def main() -> None:
    args = _parser().parse_args()
    staging_dir = package(args.checkpoint, args.staging_dir)
    if args.upload:
        upload(staging_dir, args.repo_id, args.revision)
    else:
        print("패키징만 완료했습니다. 업로드하려면 --upload와 HF Write 인증을 사용하세요.")


if __name__ == "__main__":
    main()
