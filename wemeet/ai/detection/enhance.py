"""검출기 입력 전용 대비 향상(CLAHE).

모델 가중치와 무관한 순수 함수만 둔다(`geometry.py`와 같은 이유).

**이 함수의 결과는 오직 검출 모델 입력으로만 쓴다.** 실제 크롭은 이 결과가
아니라 원본 이미지에서 뜬다 — Stage 2(`wemeet/ai/geometry`)가 학습한 크롭은
CLAHE를 전제하지 않으므로, 크롭 자체를 CLAHE 처리된 픽셀로 넘기면 Stage 2의
학습/실전 도메인이 어긋난다. 또한 CLAHE는 이미 하얗게 날아간(포화) 반사
픽셀은 복원하지 못하므로 Stage 0(베스트 프레임 선택)·Stage 4a(반사 대응)를
대체하지 않는다 — 저대비·저조도 상황에서 검출기가 바코드 경계를 더 잘 보도록
돕는 보완 수단일 뿐이다.

`docs/experiments/2026-09-22-stage1-detection`의 측정치(mAP50 0.973 등)는
CLAHE 적용 전 원본 이미지 기준이다. CLAHE를 넣은 뒤의 재측정은 아직 없다.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["DEFAULT_CLIP_LIMIT", "DEFAULT_TILE_GRID_SIZE", "enhance_for_detection"]

DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = (8, 8)


def enhance_for_detection(
    image_bgr: np.ndarray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
) -> np.ndarray:
    """LAB의 L(밝기) 채널에만 CLAHE를 적용해 색은 보존한 채 대비만 올린다."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
