"""`detect_obb()` 와 `obb_corners()` 테스트.

`detect_obb()` 는 실제 YOLO 대신 `load_model` 을 가짜로 바꿔 검증한다 — CI 에는
가중치가 없다 (`tests/test_detection_enhance.py` 와 같은 이유).
"""

from __future__ import annotations

import numpy as np
import pytest

import wemeet.ai.detection as detection
from wemeet.ai.detection.geometry import RATIO_5X3, obb_corners


class _Row:
    """torch 텐서 흉내 — `.cpu().numpy()` 체인만 지원한다."""

    def __init__(self, values: list[float]) -> None:
        self._array = np.asarray(values, dtype=np.float64)

    def cpu(self) -> _Row:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _Obb:
    def __init__(self, boxes: list[list[float]], confs: list[float]) -> None:
        self.conf = np.asarray(confs)
        self.xywhr = [_Row(b) for b in boxes]

    def __len__(self) -> int:
        return len(self.xywhr)


class _Result:
    def __init__(self, obb: _Obb | None) -> None:
        self.obb = obb


class _Model:
    def __init__(self, obb: _Obb | None) -> None:
        self._obb = obb

    def predict(self, _image: np.ndarray, **_kwargs: object) -> list:
        return [_Result(self._obb)]


def _use(monkeypatch, obb: _Obb | None) -> None:
    monkeypatch.setattr(detection, "load_model", lambda: _Model(obb))


# ── obb_corners ───────────────────────────────────────────────────


def test_수평_박스의_꼭짓점은_중심에서_반폭_반높이다():
    corners = obb_corners(100, 50, 40, 20, 0.0)
    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32
    assert sorted(map(tuple, corners.tolist())) == [(80, 40), (80, 60), (120, 40), (120, 60)]


@pytest.mark.parametrize("angle_deg", [-30, 15, 60])
def test_회전해도_변의_길이와_중심이_유지된다(angle_deg):
    corners = obb_corners(200, 120, 90, 30, np.radians(angle_deg))
    assert np.allclose(corners.mean(axis=0), (200, 120), atol=1e-4)
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    assert np.allclose(sorted(sides), [30, 30, 90, 90], atol=1e-3)


# ── detect_obb ────────────────────────────────────────────────────


def test_바코드가_없으면_None(monkeypatch):
    _use(monkeypatch, None)
    assert detection.detect_obb(np.zeros((10, 10, 3), np.uint8)) is None


def test_비정상_입력이면_None(monkeypatch):
    _use(monkeypatch, None)
    assert detection.detect_obb(np.zeros((10, 10), np.uint8)) is None


def test_확신도가_가장_높은_박스를_고른다(monkeypatch):
    _use(monkeypatch, _Obb([[10, 10, 50, 30, 0.0], [100, 80, 50, 30, 0.0]], [0.4, 0.9]))
    found = detection.detect_obb(np.zeros((200, 200, 3), np.uint8))
    assert found is not None
    assert (found.cx, found.cy) == (100, 80)
    assert found.confidence == pytest.approx(0.9)


def test_세로로_나온_박스도_긴_변이_가로가_되고_꼭짓점은_그대로다(monkeypatch):
    # 모델이 같은 박스를 90도 돌려 내도 화면의 사각형은 같아야 한다
    _use(monkeypatch, _Obb([[100, 80, 30, 50, 0.0]], [0.8]))
    found = detection.detect_obb(np.zeros((200, 200, 3), np.uint8))
    assert found is not None
    assert found.w >= found.h
    assert found.ratio == RATIO_5X3
    expected = obb_corners(100, 80, 30, 50, 0.0)
    for p in expected:
        assert np.min(np.linalg.norm(found.corners_px - p, axis=1)) < 1e-3


def test_detect_는_detect_obb_와_같은_확신도를_낸다(monkeypatch):
    _use(monkeypatch, _Obb([[100, 100, 60, 36, 0.0]], [0.7]))
    image = np.zeros((200, 200, 3), np.uint8)
    found = detection.detect_obb(image)
    detected = detection.detect(image)
    assert found is not None and detected is not None
    assert detected.confidence == found.confidence
