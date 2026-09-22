"""Stage 0 (준비 단계) — 여러 프레임 중 가장 잘 읽힐 만한 한 장을 고른다.

**wemeet 공식 4단계 파이프라인(docs/architecture.md)에는 포함되지 않는다.**
이 프로젝트의 공식 계약(`wemeet/schemas.py`, `wemeet.sw.pipeline.run`)은 사진
한 장(`image_bgr`)을 입력으로 가정한다. 카메라가 같은 바코드를 여러 프레임
연속으로 찍는 경우, 그중 반사가 적고 선명한 한 장을 이 단계에서 고른 뒤
**image_bgr 한 장으로 줄여서** `pipeline.run()`에 넘기면 된다.

    from wemeet.ai.best_frame import select_best_frame
    from wemeet.sw.pipeline import run

    best = select_best_frame(frames)   # frames: list[np.ndarray], BGR uint8
    result = run(best.image_bgr)

`wemeet/schemas.py`는 건드리지 않는다 — 이 모듈의 반환형(`BestFrameResult`)은
AI·SW 양쪽이 승인해야 하는 파트 간 공용 계약이 아니라, 이 모듈 안에서만 쓰는
지역 타입이다. 트래킹(`TrackedFrames`) 설계가 아직 확정되지 않은 상태에서도
스키마 변경 논의 없이 붙였다 뗄 수 있게 하려는 의도다.

반사 판정: 단순히 "밝기 250 초과 픽셀 비율"만 쓰면 안 된다 — 바코드는 원래
검은 막대 + 흰 배경(quiet zone)이라 반사가 전혀 없어도 절반 가까이가 밝은
픽셀이다. 그래서 "밝다"에 "그 주변(막대 폭 정도 반경)에 아무 무늬(경계)도
없다"는 조건을 더한다: 흰 배경은 밝아도 바로 옆 검은 막대와의 경계 때문에
근처 Laplacian이 크고, 진짜 반사로 뭉개진 자리는 밝은 채로 주변 막대 경계까지
같이 사라져서 그 반경 안에 Laplacian이 전부 작다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

Bbox = tuple[int, int, int, int]

__all__ = [
    "BestFrameResult",
    "compute_glare_ratio",
    "compute_sharpness_score",
    "select_best_frame",
]

GLARE_VALUE_THRESHOLD = 250  # HSV V채널이 이 값을 넘으면 '밝다'
GLARE_FLATNESS_THRESHOLD = (
    15.0  # 이 값보다 (주변 반경 내) Laplacian 크기가 작으면 '무늬 없음(뭉개짐)'
)
GLARE_NEIGHBORHOOD_PX = 15  # '주변'의 반경(px) — 대략 바코드 막대 폭 스케일
DEFAULT_GLARE_RATIO_THRESHOLD = 0.4  # 모든 프레임의 glare_ratio가 이 이상이면 저품질 fallback


@dataclass
class BestFrameResult:
    """`select_best_frame()` 반환값. `wemeet.schemas`의 공용 계약이 아니다."""

    image_bgr: np.ndarray  # 선택된 프레임 원본. BGR uint8, (H, W, 3)
    frame_index: int  # frames 내 선택 위치 (재현/디버깅용)
    glare_ratio: float  # [0, 1], 낮을수록 좋음
    sharpness_score: float  # 정규화 전 raw Laplacian 분산, 높을수록 좋음
    quality_score: float  # [0, 1], 최종 선택 기준
    low_quality_fallback: bool  # 전체가 저품질(반사 심함)이라 차선책으로 고른 경우 True
    stage_latency_ms: float


def _crop_roi(frame: np.ndarray, bbox: Bbox | None) -> np.ndarray:
    """bbox(x, y, w, h)로 ROI를 잘라낸다. bbox가 없으면 전체 프레임을 쓴다.

    프레임 경계 밖이면 안쪽으로 clamp하고, clamp 후 영역이 비면(bbox가 완전히
    프레임 밖이면) 전체 프레임으로 대체한다.
    """
    if bbox is None:
        return frame
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    x0 = max(0, min(x, width))
    y0 = max(0, min(y, height))
    x1 = max(x0, min(x + w, width))
    y1 = max(y0, min(y + h, height))
    roi = frame[y0:y1, x0:x1]
    return roi if roi.size > 0 else frame


def compute_glare_ratio(
    roi: np.ndarray,
    flatness_threshold: float = GLARE_FLATNESS_THRESHOLD,
    neighborhood_px: int = GLARE_NEIGHBORHOOD_PX,
) -> float:
    """진짜 반사로 날아간 픽셀의 비율([0, 1]). 낮을수록 좋음(반사 적음).

    "밝다(V > 250)"만으로 판정하면 바코드의 정상적인 흰 배경까지 반사로
    오판한다. 그래서 "밝다" AND "주변 neighborhood_px 반경 안에 무늬(경계)가
    전혀 없다"를 같이 요구한다 — 흰 배경은 근처 검은 막대와의 경계 때문에
    이 반경 안 어딘가에 Laplacian이 크게 남아있지만, 진짜 반사는 그 반경
    전체가 뭉개져서 Laplacian이 다 작다.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2] > GLARE_VALUE_THRESHOLD

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edge_strength = np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32)
    # 3x3 Laplacian은 경계 바로 위 1px만 잡으므로, 그걸 neighborhood_px만큼
    # 팽창(dilate)시켜 "이 반경 안 어딘가에 경계가 있는지"로 넓혀서 본다.
    kernel = np.ones((neighborhood_px, neighborhood_px), np.uint8)
    nearby_edge_strength = cv2.dilate(edge_strength, kernel)
    flat = nearby_edge_strength < flatness_threshold

    glare_mask = bright & flat
    return float(np.mean(glare_mask))


def compute_sharpness_score(roi: np.ndarray) -> float:
    """그레이스케일 Laplacian 분산(정규화 전 raw 값). 높을수록 좋음(더 선명함)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _min_max_normalize(values: list[float]) -> list[float]:
    """[0, 1] min-max 정규화. 전부 같은 값이면(분모 0) 전부 1.0으로 취급해 0으로 나누지 않는다."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def select_best_frame(
    frames: list[np.ndarray],
    bboxes: list[Bbox | None] | None = None,
    glare_ratio_threshold: float = DEFAULT_GLARE_RATIO_THRESHOLD,
) -> BestFrameResult:
    """frames 중 quality_score(=정규화된 선명도 × (1 - 반사비율))가 가장 높은 프레임을 고른다.

    `bboxes`를 주면(트래커가 있다면) 그 ROI만 보고 판정하고, 주지 않으면
    (`None` 또는 생략) 프레임 전체를 본다.

    모든 프레임의 glare_ratio가 `glare_ratio_threshold` 이상이면(전체가 반사로
    뒤덮인 경우) quality_score 대신 glare_ratio가 최소인 프레임을 고르고
    `low_quality_fallback=True`로 표시한다.
    """
    if not frames:
        raise ValueError("frames가 비어 있습니다.")
    if bboxes is not None and len(bboxes) != len(frames):
        raise ValueError("bboxes를 주려면 frames와 길이가 같아야 합니다.")

    start = time.perf_counter()

    resolved_bboxes: list[Bbox | None] = bboxes if bboxes is not None else [None] * len(frames)
    rois = [_crop_roi(frame, bbox) for frame, bbox in zip(frames, resolved_bboxes)]
    glare_ratios = [compute_glare_ratio(roi) for roi in rois]
    sharpness_scores = [compute_sharpness_score(roi) for roi in rois]
    normalized_sharpness = _min_max_normalize(sharpness_scores)

    quality_scores = [
        norm_sharp * (1.0 - glare) for norm_sharp, glare in zip(normalized_sharpness, glare_ratios)
    ]

    low_quality_fallback = all(g >= glare_ratio_threshold for g in glare_ratios)

    if low_quality_fallback:
        best_index = min(range(len(glare_ratios)), key=lambda i: glare_ratios[i])
    else:
        best_index = max(range(len(quality_scores)), key=lambda i: quality_scores[i])

    stage_latency_ms = (time.perf_counter() - start) * 1000.0

    return BestFrameResult(
        image_bgr=frames[best_index],
        frame_index=best_index,
        glare_ratio=glare_ratios[best_index],
        sharpness_score=sharpness_scores[best_index],
        quality_score=quality_scores[best_index],
        low_quality_fallback=low_quality_fallback,
        stage_latency_ms=stage_latency_ms,
    )
