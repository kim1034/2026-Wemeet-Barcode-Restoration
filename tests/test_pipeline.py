"""wemeet/sw/pipeline/ 의 run() 검증.

wemeet/ai/ 아래는 아직 NotImplementedError 스켈레톤이다 (AI파트가 채울
자리). 여기서는 그 패키지를 건드리지 않고, pipeline 이 참조하는 이름만
monkeypatch 로 가짜 값으로 바꿔치기해서 4단계 연결 자체를 검증한다.
"""

import numpy as np
import pytest

from wemeet.schemas import DetectedBarcode, GeometryField
from wemeet.sw.pipeline import run


def _fake_detect(image_bgr: np.ndarray) -> DetectedBarcode | None:
    return DetectedBarcode(crop_bgr_uint8=image_bgr, angle_deg_ccw=0.0, confidence=0.9)


def _fake_estimate_geometry(target: DetectedBarcode) -> GeometryField:
    dst = np.array([[x, y] for y in (0.0, 0.5, 1.0) for x in (0.0, 0.33, 0.66, 1.0)])
    src = dst + 0.02  # 약한 왜곡 흉내
    return GeometryField(
        control_points_dst_norm=dst,
        control_points_src_norm=src,
        method="tps",
        confidence=0.9,
    )


def test_pipeline_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch, clean_barcode_bgr: np.ndarray
) -> None:
    monkeypatch.setattr("wemeet.sw.pipeline.detect", _fake_detect)
    monkeypatch.setattr("wemeet.sw.pipeline.estimate_geometry", _fake_estimate_geometry)

    result = run(clean_barcode_bgr)

    assert result.ok
    assert result.decode.text == "WEMEET0001"
    assert result.decode.failure_reason is None
    assert result.detected is not None
    assert result.rectified is not None


def test_pipeline_reports_not_detected_when_detect_finds_nothing(
    monkeypatch: pytest.MonkeyPatch, clean_barcode_bgr: np.ndarray
) -> None:
    monkeypatch.setattr("wemeet.sw.pipeline.detect", lambda image_bgr: None)

    result = run(clean_barcode_bgr)

    assert not result.ok
    assert result.detected is None
    assert result.rectified is None
    assert result.decode.failure_reason == "not_detected"


def test_pipeline_degrades_when_geometry_confidence_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 디코더가 절대 읽을 수 없는 순수 노이즈 크롭 — 1차 시도가 반드시 실패해야
    # estimate_geometry 분기까지 도달한다.
    rng = np.random.default_rng(0)
    noise_gray = rng.integers(0, 256, size=(120, 300), dtype=np.uint8)
    noise_bgr = np.repeat(noise_gray[:, :, None], 3, axis=2)

    def _low_confidence_geometry(target: DetectedBarcode) -> GeometryField:
        field = _fake_estimate_geometry(target)
        field.confidence = 0.1
        return field

    monkeypatch.setattr("wemeet.sw.pipeline.detect", _fake_detect)
    monkeypatch.setattr("wemeet.sw.pipeline.estimate_geometry", _low_confidence_geometry)

    result = run(noise_bgr)

    assert not result.ok
    assert result.decode.degraded
    assert result.decode.failure_reason == "decode_failed"
    assert result.rectified is not None
    assert result.rectified.field is None


def test_pipeline_survives_exception_from_estimate_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(1)
    noise_gray = rng.integers(0, 256, size=(120, 300), dtype=np.uint8)
    noise_bgr = np.repeat(noise_gray[:, :, None], 3, axis=2)

    def _raises(target: DetectedBarcode) -> GeometryField:
        raise RuntimeError("아직 학습되지 않은 모델")

    monkeypatch.setattr("wemeet.sw.pipeline.detect", _fake_detect)
    monkeypatch.setattr("wemeet.sw.pipeline.estimate_geometry", _raises)

    result = run(noise_bgr)  # 예외가 여기서 튀어나오면 안 된다

    assert not result.ok
    assert result.decode.degraded
