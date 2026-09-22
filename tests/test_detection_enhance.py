"""Stage 1 CLAHE 전처리 테스트.

`enhance_for_detection` 자체는 가중치 없이 도는 순수 함수 테스트다.
`detect()` 쪽은 실제 YOLO 모델 대신 `load_model`을 가짜로 바꿔치기해서,
"모델 입력에는 CLAHE가 적용되지만 실제 크롭은 원본에서 뜬다"는 계약만
검증한다 — 모델을 직접 부르는 테스트를 넣으면 CI에서 가중치가 없어 깨진다
(`tests/test_detection.py` 와 같은 이유).
"""

from __future__ import annotations

import cv2
import numpy as np

import wemeet.ai.detection as detection
from wemeet.ai.detection.enhance import enhance_for_detection

# ── enhance_for_detection ─────────────────────────────────────────


def test_enhance_for_detection_preserves_shape_and_dtype():
    image = np.random.default_rng(0).integers(0, 256, size=(50, 80, 3), dtype=np.uint8)
    enhanced = enhance_for_detection(image)
    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8


def test_enhance_for_detection_expands_narrow_intensity_range():
    # 전체 화면이 100~120 사이 좁은 범위에만 몰린 저대비 그라데이션.
    # CLAHE가 정확히 이런 경우(좁은 동적 범위)를 0~255로 펴서 대비를 올린다.
    gradient = np.linspace(100, 120, 60, dtype=np.uint8)
    base = np.tile(gradient, (60, 1))
    base_bgr = np.stack([base, base, base], axis=-1)

    enhanced = enhance_for_detection(base_bgr)

    gray_before = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    assert gray_after.std() > gray_before.std()


# ── detect() 와의 연결 ────────────────────────────────────────────


class _FakeTensorRow:
    """torch 텐서 흉내 — `.cpu().numpy()` 체인만 지원한다."""

    def __init__(self, values: list[float]) -> None:
        self._array = np.asarray(values, dtype=np.float64)

    def cpu(self) -> _FakeTensorRow:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeObb:
    def __init__(self, cx: float, cy: float, w: float, h: float, angle_rad: float) -> None:
        self.conf = np.array([0.9])
        self.xywhr = [_FakeTensorRow([cx, cy, w, h, angle_rad])]

    def __len__(self) -> int:
        return 1


class _NoObb:
    obb = None


class _FakeResult:
    def __init__(self, obb: _FakeObb) -> None:
        self.obb = obb


class _FakeModel:
    def __init__(self, results: list) -> None:
        self._results = results
        self.received_input: np.ndarray | None = None

    def predict(self, image: np.ndarray, **_kwargs: object) -> list:
        self.received_input = image
        return self._results


def test_detect_applies_clahe_to_model_input_by_default(monkeypatch):
    fake_model = _FakeModel([_NoObb()])  # 미검출로 일찍 반환시켜 크롭 경로는 안 탄다
    monkeypatch.setattr(detection, "load_model", lambda: fake_model)

    def _fake_enhance(image: np.ndarray, *_args: object, **_kwargs: object) -> np.ndarray:
        return np.full_like(image, 255)

    monkeypatch.setattr(detection, "enhance_for_detection", _fake_enhance)

    original = np.zeros((10, 10, 3), dtype=np.uint8)
    detection.detect(original)

    assert fake_model.received_input is not None
    assert np.all(fake_model.received_input == 255)  # 가짜 enhance 결과가 모델에 들어갔다


def test_detect_use_clahe_false_skips_enhancement(monkeypatch):
    fake_model = _FakeModel([_NoObb()])
    monkeypatch.setattr(detection, "load_model", lambda: fake_model)

    def _must_not_be_called(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("use_clahe=False면 enhance_for_detection이 호출되면 안 된다")

    monkeypatch.setattr(detection, "enhance_for_detection", _must_not_be_called)

    original = np.zeros((10, 10, 3), dtype=np.uint8)
    detection.detect(original, use_clahe=False)

    assert fake_model.received_input is not None
    assert np.array_equal(fake_model.received_input, original)


def test_detect_crop_comes_from_original_not_enhanced_image(monkeypatch):
    size = 40
    original = np.zeros((size, size, 3), dtype=np.uint8)

    fake_obb = _FakeObb(cx=size / 2, cy=size / 2, w=20, h=10, angle_rad=0.0)
    fake_model = _FakeModel([_FakeResult(fake_obb)])
    monkeypatch.setattr(detection, "load_model", lambda: fake_model)
    monkeypatch.setattr(
        detection, "enhance_for_detection", lambda image, *a, **k: np.full_like(image, 255)
    )

    result = detection.detect(original)

    assert result is not None
    assert np.all(fake_model.received_input == 255)  # 모델 입력은 enhance된 것
    assert not np.all(result.crop_bgr_uint8 == 255)  # 크롭은 원본(전부 0)에서 왔다
