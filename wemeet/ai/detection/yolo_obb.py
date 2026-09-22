"""YOLO11n-OBB 백엔드.

**ultralytics 는 pyproject 의 필수 의존성이 아니다.** AGPL-3.0 이라 기획서
8.2 의 상용화 목표와 충돌할 수 있고, 아직 `docs/decisions/0002` 가 미결정이다
(2026-09-22 기준). 그래서 이 파일은 함수 안에서 늦게 import 한다 —
ultralytics 가 없는 환경에서도 `wemeet` 전체가 import 되고 CI 가 돈다.

쓰려면 학습·추론 환경에만 따로 깐다::

    uv pip install ultralytics

가중치(`*.pt`)는 .gitignore 로 커밋이 막혀 있다. HF Hub 규약은
docs/external/huggingface.md 를 따른다 (`123metro/barcode-weights`,
`detection/v2-yolo-obb/best.pt`). 경로를 이 순서로 찾는다.

1. 인자로 준 경로
2. 환경변수 ``WEMEET_DETECTION_WEIGHTS``
3. ``downloads/detection/v2-yolo-obb/best.pt`` (`hf download` 기본 위치)
4. Hugging Face Hub — 위 규약 경로. 환경변수로 덮어쓸 수 있다
"""

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_WEIGHTS = Path("downloads/detection/v2-yolo-obb/best.pt")
DEFAULT_HF_REPO = "123metro/barcode-weights"
DEFAULT_HF_FILENAME = "detection/v2-yolo-obb/best.pt"
#: 학습 입력 크기. 데이터셋이 1280x720 이라 960 이면 화질 손실이 거의 없다.
DEFAULT_IMGSZ = 960


def resolve_weights(weights: str | Path | None = None) -> Path:
    """가중치 파일 경로를 찾는다. 없으면 어디에 두면 되는지 알려준다."""
    for candidate in (weights, os.environ.get("WEMEET_DETECTION_WEIGHTS"), DEFAULT_WEIGHTS):
        if candidate and Path(candidate).exists():
            return Path(candidate)

    repo = os.environ.get("WEMEET_DETECTION_HF_REPO", DEFAULT_HF_REPO)
    filename = os.environ.get("WEMEET_DETECTION_HF_FILE", DEFAULT_HF_FILENAME)
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename=filename))
    except Exception as exc:  # 네트워크·권한·미업로드 모두 여기로 온다
        raise FileNotFoundError(
            f"탐지 가중치를 찾지 못했습니다 ({repo}/{filename}: {exc}).\n"
            "가중치는 git 에 커밋하지 않습니다 (.gitignore, docs/external/huggingface.md).\n"
            f"  · hf download {repo} {filename} --local-dir downloads 로 받거나\n"
            "  · WEMEET_DETECTION_WEIGHTS 로 로컬 경로를 직접 주세요."
        ) from exc


@lru_cache(maxsize=2)
def load_model(weights: str | None = None):
    """YOLO 모델을 한 번만 올려 재사용한다 (매 프레임 로드하면 초당 몇 장도 못 낸다)."""
    from ultralytics import YOLO

    return YOLO(str(resolve_weights(weights)))
