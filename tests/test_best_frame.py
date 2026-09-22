"""wemeet.ai.best_frame 단위 테스트 — 합성 이미지만 사용, 모델 불필요."""

from __future__ import annotations

import numpy as np
import pytest

from wemeet.ai.best_frame import (
    compute_glare_ratio,
    compute_sharpness_score,
    select_best_frame,
)


def _flat_frame(value: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _checkerboard_frame(size: int = 64, tile: int = 4) -> np.ndarray:
    # 80/200 값만 써서(255 근처 없음) 반사 없이도 Laplacian 분산이 큰 이미지를 만든다.
    grid = (np.indices((size, size)).sum(axis=0) // tile) % 2
    gray = np.where(grid == 1, 200, 80).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _partial_glare_frame(bright_ratio: float, size: int = 64) -> np.ndarray:
    """상단 bright_ratio만큼을 반사(255)로, 나머지는 어둡게(100) 채운 프레임.
    glare_ratio가 대략 bright_ratio가 되도록 만들어 fallback 로직을 결정론적으로 테스트한다."""
    frame = np.full((size, size, 3), 100, dtype=np.uint8)
    bright_rows = int(size * bright_ratio)
    frame[:bright_rows, :, :] = 255
    return frame


def _barcode_stripe_frame(size: int = 96, bar_width: int = 6) -> np.ndarray:
    """검은 막대(30)와 흰 배경(255)이 번갈아 있는 합성 바코드 패턴. 반사는 전혀 없다."""
    cols = np.arange(size)
    is_white_bar = (cols // bar_width) % 2 == 0
    row = np.where(is_white_bar, 255, 30).astype(np.uint8)
    gray = np.tile(row, (size, 1))
    return np.stack([gray, gray, gray], axis=-1)


def _bright_patch_frame(size: int = 96, patch_size: int = 40, background: int = 60) -> np.ndarray:
    """어두운 배경(60, 반사 없음) 한가운데에 무늬 없는 큰 순백색(255) 사각형(진짜 반사)."""
    frame = np.full((size, size, 3), background, dtype=np.uint8)
    start = (size - patch_size) // 2
    frame[start : start + patch_size, start : start + patch_size, :] = 255
    return frame


def test_compute_glare_ratio_flat_bright_is_high():
    assert compute_glare_ratio(_flat_frame(255)) == pytest.approx(1.0)


def test_compute_glare_ratio_flat_dark_is_zero():
    assert compute_glare_ratio(_flat_frame(100)) == pytest.approx(0.0)


def test_compute_glare_ratio_ignores_legit_barcode_white_background():
    # 바코드 절반이 흰 배경(255)이라 단순 '밝기 비율'이면 glare_ratio가 크게 잡히지만,
    # 검은 막대와 붙어있는(무늬가 있는) 흰 부분이라 진짜 반사가 아니어야 한다.
    stripes = _barcode_stripe_frame()
    assert compute_glare_ratio(stripes) < 0.15


def test_compute_glare_ratio_detects_large_blown_out_patch():
    # 주변에 아무 무늬도 없이 큰 면적이 통째로 하얗게 뭉개진 경우는 진짜 반사로 잡혀야 한다.
    stripes = _barcode_stripe_frame()
    patch = _bright_patch_frame()
    assert compute_glare_ratio(patch) > compute_glare_ratio(stripes)
    assert compute_glare_ratio(patch) > 0.03


def test_compute_sharpness_score_checkerboard_beats_flat():
    assert compute_sharpness_score(_checkerboard_frame()) > compute_sharpness_score(
        _flat_frame(150)
    )


def test_select_best_frame_prefers_sharp_no_glare():
    blurry_glare = _flat_frame(255)  # 반사 심하고(글레어) 완전히 평탄함(선명도 0)
    mid = _flat_frame(150)  # 반사는 없지만 역시 평탄함(선명도 0)
    sharp_clean = _checkerboard_frame()  # 반사 없고 선명함

    best = select_best_frame([blurry_glare, mid, sharp_clean])

    assert best.frame_index == 2
    assert best.low_quality_fallback is False
    assert 0.0 <= best.quality_score <= 1.0
    assert best.image_bgr is sharp_clean


def test_select_best_frame_falls_back_when_all_glare():
    # 세 프레임 다 임계값(0.4)을 넘는 glare_ratio를 갖되, 서로 다르게 만들어
    # '그중 최솟값을 고른다'는 fallback 로직을 결정론적으로 검증한다.
    frames = [
        _partial_glare_frame(0.9),
        _partial_glare_frame(0.7),  # glare_ratio가 가장 낮음 -> 선택돼야 함
        _partial_glare_frame(0.8),
    ]

    best = select_best_frame(frames, glare_ratio_threshold=0.4)

    assert best.low_quality_fallback is True
    assert best.frame_index == 1


def test_select_best_frame_empty_frames_raises():
    with pytest.raises(ValueError):
        select_best_frame([])


def test_select_best_frame_bbox_length_mismatch_raises():
    with pytest.raises(ValueError):
        select_best_frame([_checkerboard_frame(), _checkerboard_frame()], bboxes=[(0, 0, 10, 10)])


def test_select_best_frame_bbox_out_of_bounds_does_not_crash():
    frame = _checkerboard_frame()

    best = select_best_frame([frame], bboxes=[(1000, 1000, 50, 50)])  # bbox가 프레임 완전히 밖

    assert best.frame_index == 0


def test_select_best_frame_without_bboxes_uses_whole_frame():
    blurry_glare = _flat_frame(255)
    sharp_clean = _checkerboard_frame()

    best = select_best_frame([blurry_glare, sharp_clean])  # bboxes 생략

    assert best.frame_index == 1
