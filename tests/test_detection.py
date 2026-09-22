"""Stage 1 탐지의 기하 계산 테스트.

가중치 파일 없이 도는 것만 둔다. `*.pt` 는 .gitignore 대상이라 CI 러너에
존재하지 않으므로, 모델을 부르는 테스트를 넣으면 CI 에서만 깨진다.

여기서 잡으려는 실수는 하나다 — **회전 부호**. 부호를 틀리면 크롭이 반대로
기울고, 2단계 기하 추정이 통째로 어긋난다. 약한 기울기에서는 우연히 읽히기도
해서 눈으로는 놓치기 쉽다.
"""

import cv2
import numpy as np
import pytest

from wemeet.ai.detection.geometry import (
    RATIO_5X3,
    RATIO_6X4,
    crop_upright,
    decide_ratio,
    fit_ratio,
    upright,
)


def _scene_with_bar(angle_deg: float, size: int = 400) -> tuple[np.ndarray, tuple[float, ...]]:
    """중앙에 가로 막대 하나를 그린 이미지와 그 박스를 만든다.

    반환하는 각도는 이미지 좌표계 라디안이다(화면에서 시계 방향이 양수).
    """
    image = np.zeros((size, size, 3), np.uint8)
    cx = cy = size / 2
    w, h = 200.0, 120.0
    box = cv2.boxPoints(((cx, cy), (w, h), angle_deg))
    cv2.fillConvexPoly(image, box.astype(np.int32), (255, 255, 255))
    # 위쪽 절반만 회색으로 칠해 위아래를 구분한다 — 180도 뒤집힘도 잡으려는 것
    top = cv2.boxPoints(((cx, cy - h / 4), (w, h / 2), angle_deg))
    cv2.fillConvexPoly(image, top.astype(np.int32), (128, 128, 128))
    return image, (cx, cy, w, h, np.deg2rad(angle_deg))


# ── 규격 판정 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("w", "h", "expected"),
    [
        (500, 300, RATIO_5X3),  # 정확히 5:3
        (600, 400, RATIO_6X4),  # 정확히 6:4
        (500, 305, RATIO_5X3),  # 조금 어긋나도 가까운 쪽으로
        (600, 390, RATIO_6X4),
    ],
)
def test_박스_모양으로_규격을_판정한다(w, h, expected):
    ratio, _ = decide_ratio(w, h)
    assert ratio == pytest.approx(expected)


def test_짧은_쪽만_늘려_비율을_맞춘다():
    """줄이면 막대가 잘린다. 늘리면 여백이 붙을 뿐이다."""
    for w, h in [(500, 330), (500, 280), (600, 400)]:
        fw, fh = fit_ratio(w, h, RATIO_5X3)
        assert fw >= w - 1e-6 and fh >= h - 1e-6  # 어느 쪽도 줄지 않는다
        assert fw / fh == pytest.approx(RATIO_5X3)


def test_세로로_선_박스는_눕혀서_판정한다():
    """모델이 같은 박스를 90도 돌려 내도 5:3 이 3:5 로 읽히면 안 된다."""
    _, _, w, h, angle = upright(0, 0, 300, 500, 0.0)
    assert (w, h) == (500, 300)
    assert angle == pytest.approx(np.pi / 2)
    assert decide_ratio(w, h)[0] == pytest.approx(RATIO_5X3)


# ── 회전 보정 ──────────────────────────────────────────────────────


@pytest.mark.parametrize("angle_deg", [-40.0, -15.0, 0.0, 15.0, 40.0, 75.0])
def test_기울어진_박스를_수평으로_세운다(angle_deg):
    image, (cx, cy, w, h, angle_rad) = _scene_with_bar(angle_deg)
    crop, _ = crop_upright(image, cx, cy, w, h, angle_rad, pad=0.0)

    # 세운 뒤에는 밝은 영역의 위쪽 경계가 가로로 평평해야 한다.
    mask = (crop.max(axis=2) > 60).astype(np.uint8)
    cols = [np.flatnonzero(mask[:, x]) for x in range(mask.shape[1])]
    tops = [c[0] for c in cols if len(c)]
    assert len(tops) > mask.shape[1] * 0.8  # 대부분의 열에 물체가 있다
    assert np.std(tops) < 2.0  # 윗변이 평평하다 = 수평이다


@pytest.mark.parametrize("angle_deg", [-30.0, 0.0, 30.0])
def test_보정한_각도를_되돌리면_원래_기울기가_된다(angle_deg):
    """angle_deg_ccw 계약 검증 — 반시계 양수, 도 단위."""
    image, (cx, cy, w, h, angle_rad) = _scene_with_bar(angle_deg)
    _, reported = crop_upright(image, cx, cy, w, h, angle_rad)
    assert reported == pytest.approx(angle_deg, abs=1e-6)


def test_위아래가_뒤집히지_않는다():
    """크롭 위쪽 절반이 원본에서도 위쪽이어야 한다 (회색 띠로 확인)."""
    image, (cx, cy, w, h, angle_rad) = _scene_with_bar(25.0)
    crop, _ = crop_upright(image, cx, cy, w, h, angle_rad, pad=0.0)
    upper = crop[: crop.shape[0] // 2].mean()
    lower = crop[crop.shape[0] // 2 :].mean()
    assert upper < lower  # 위쪽이 회색(128), 아래쪽이 흰색(255)


# ── 크롭 모양 ──────────────────────────────────────────────────────


@pytest.mark.parametrize("ratio", [RATIO_5X3, RATIO_6X4])
def test_크롭은_규격_비율과_10퍼센트_여백을_지킨다(ratio):
    image, (cx, cy, _, _, angle_rad) = _scene_with_bar(12.0)
    w, h = 200.0, 200.0 / ratio
    crop, _ = crop_upright(image, cx, cy, w, h, angle_rad, pad=0.10)

    out_h, out_w = crop.shape[:2]
    assert out_w / out_h == pytest.approx(ratio, rel=0.01)  # 여백을 붙여도 비율 유지
    assert out_w == pytest.approx(w * 1.2, rel=0.01)  # 양쪽 10%씩


def test_크롭은_계약이_요구하는_형식이다():
    image, (cx, cy, w, h, angle_rad) = _scene_with_bar(10.0)
    crop, _ = crop_upright(image, cx, cy, w, h, angle_rad)
    assert crop.dtype == np.uint8
    assert crop.ndim == 3 and crop.shape[2] == 3
