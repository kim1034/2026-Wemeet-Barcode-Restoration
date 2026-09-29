"""Stage 2 geometry inference backed by the ResNet18-C3 checkpoint.

The public contract is deliberately small: :func:`estimate_geometry` receives
the ``DetectedBarcode`` produced by Stage 1 and returns only normalized control
points. The model is loaded lazily on the first call so importing the package
does not download weights or require a CUDA installation. A local checkpoint
or the Hugging Face Hub can provide the weights.

The model does not predict uncertainty. ``GeometryField.confidence`` is
therefore an availability/validity flag (``1.0`` for a finite prediction),
not a calibrated probability. Pipeline confidence gating remains useful for
future calibrated checkpoints without pretending that this checkpoint has an
uncertainty head.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from wemeet.schemas import DetectedBarcode, GeometryField

DEFAULT_WEIGHTS = Path("downloads/geometry/resnet18-c3/model_state_dict.pt")
DEFAULT_HF_REPO = "123metro/barcode-weights"
DEFAULT_HF_FILENAME = "geometry/resnet18-c3/model_state_dict.pt"
DEFAULT_HF_REVISION = "main"
INPUT_HEIGHT = 160
INPUT_WIDTH = 384

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def resolve_weights(weights: str | Path | None = None) -> Path:
    """Resolve a local checkpoint or download the published HF checkpoint.

    Resolution order is explicit argument, ``WEMEET_GEOMETRY_WEIGHTS``, the
    conventional ``downloads/`` path, then Hugging Face Hub. The downloaded
    file is cached by ``huggingface_hub`` and is not copied into the repository.
    """
    candidates = (weights, os.environ.get("WEMEET_GEOMETRY_WEIGHTS"), DEFAULT_WEIGHTS)
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path.resolve()

    repo = os.environ.get("WEMEET_GEOMETRY_HF_REPO", DEFAULT_HF_REPO)
    filename = os.environ.get("WEMEET_GEOMETRY_HF_FILE", DEFAULT_HF_FILENAME)
    revision = os.environ.get("WEMEET_GEOMETRY_HF_REVISION", DEFAULT_HF_REVISION)
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision)).resolve()
    except Exception as exc:  # network, auth, or an unpublished checkpoint
        raise FileNotFoundError(
            f"기하 추정 가중치를 찾지 못했습니다 ({repo}/{filename}@{revision}: {exc}).\n"
            "다음 중 하나를 준비하세요.\n"
            f"  · {DEFAULT_WEIGHTS} 에 파일을 두기\n"
            "  · WEMEET_GEOMETRY_WEIGHTS 로 로컬 파일 지정\n"
            f"  · HF 저장소에 {filename} 업로드하기"
        ) from exc


def _select_device(device: str | None = None):
    import torch

    requested = device or os.environ.get("WEMEET_GEOMETRY_DEVICE")
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _load_payload(path: Path):
    import torch

    # The published deployment file is a plain state_dict. ``weights_only``
    # avoids arbitrary pickle objects while retaining compatibility with older
    # torch versions that do not expose the keyword.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache(maxsize=2)
def _load_model_cached(path: str, device_name: str):
    from wemeet.ai.geometry.model import ResNet18C3

    model = ResNet18C3(pretrained=False)
    payload = _load_payload(Path(path))
    if isinstance(payload, Mapping) and "model" in payload:
        payload = payload["model"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"가중치 파일 형식이 state_dict가 아닙니다: {path}")
    model.load_state_dict(payload, strict=True)
    model.to(device_name)
    model.eval()
    return model


def clear_model_cache() -> None:
    """Drop the process-local model cache (useful after changing env paths)."""
    _load_model_cached.cache_clear()


def load_model(
    weights: str | Path | None = None,
    device: str | None = None,
):
    """Load and cache ResNet18-C3 for inference."""
    selected = _select_device(device)
    path = resolve_weights(weights)
    return _load_model_cached(str(path), str(selected))


def _prepare_input(target: DetectedBarcode, device):
    import cv2
    import torch

    gray = cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(resized.astype(np.float32) / 255.0)
    image = image.unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(_STD, dtype=image.dtype).view(3, 1, 1)
    return ((image - mean) / std).unsqueeze(0).to(device)


def estimate_geometry(target: DetectedBarcode) -> GeometryField:
    """Predict normalized source control points for one detected crop.

    The checkpoint predicts ``(48, 2)`` source points for the fixed ``16×3``
    destination grid. Coordinates stay in the original crop's normalized
    coordinate system even though the model input is resized to ``160×384``.
    """
    import torch

    device = _select_device()
    model = load_model(device=str(device))
    with torch.inference_mode():
        predicted = model(_prepare_input(target, device))[0].detach().cpu().numpy()

    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != (48, 2) or not np.isfinite(predicted).all():
        raise ValueError(f"ResNet18-C3 출력이 비정상입니다: shape={predicted.shape}")
    if predicted.min() < -0.5 or predicted.max() > 1.5:
        raise ValueError("ResNet18-C3 출력이 허용된 정규화 좌표 범위를 벗어났습니다")

    from wemeet.ai.geometry.model import destination_grid

    destination = destination_grid().detach().cpu().numpy().astype(np.float64)
    return GeometryField(
        control_points_dst_norm=destination,
        control_points_src_norm=predicted,
        method="tps",
        # This model has no uncertainty head; finite output means available.
        confidence=1.0,
    )


__all__ = [
    "DEFAULT_HF_FILENAME",
    "DEFAULT_HF_REPO",
    "DEFAULT_WEIGHTS",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "clear_model_cache",
    "estimate_geometry",
    "load_model",
    "resolve_weights",
]
