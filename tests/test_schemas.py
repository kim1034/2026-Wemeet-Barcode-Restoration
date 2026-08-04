import numpy as np
import pytest

from wemeet.schemas import (
    DecodeResult,
    DetectedBarcode,
    PipelineResult,
    RestoredBarcode,
)


def _crop(h: int = 20, w: int = 60) -> DetectedBarcode:
    """정상적인 DetectedBarcode 하나. 다른 테스트의 재료로 쓴다."""
    return DetectedBarcode(
        crop_bgr_uint8=np.zeros((h, w, 3), dtype=np.uint8),
        angle_deg_ccw=0.0,
        confidence=0.9,
    )


def test_잘못된_dtype은_거부된다():
    with pytest.raises(AssertionError):
        DetectedBarcode(
            crop_bgr_uint8=np.zeros((10, 10, 3), dtype=np.float32),
            angle_deg_ccw=0.0,
            confidence=0.5,
        )


def test_confidence가_1을_넘으면_거부된다():
    with pytest.raises(AssertionError):
        DetectedBarcode(
            crop_bgr_uint8=np.zeros((10, 10, 3), dtype=np.uint8),
            angle_deg_ccw=0.0,
            confidence=1.5,
        )


def test_복원_이미지가_흑백_단일채널이_아니면_거부된다():
    with pytest.raises(AssertionError):
        RestoredBarcode(
            image_gray_uint8=np.zeros((10, 10, 3), dtype=np.uint8),
            source=_crop(),
        )


def test_판독_실패시_symbology도_None이어야_한다():
    with pytest.raises(AssertionError):
        DecodeResult(
            text=None,
            symbology="CODE128",
            retry_count=0,
            failure_reason="decode_failed",
        )


def test_실패에는_반드시_이유가_있어야_한다():
    with pytest.raises(AssertionError):
        DecodeResult(text=None, symbology=None, retry_count=0, failure_reason=None)


def test_성공했는데_실패_이유가_있으면_거부된다():
    with pytest.raises(AssertionError):
        DecodeResult(
            text="123",
            symbology="CODE128",
            retry_count=0,
            failure_reason="decode_failed",
        )


def test_재시도는_3회를_넘을_수_없다():
    with pytest.raises(AssertionError):
        DecodeResult(text="123", symbology="CODE128", retry_count=4)


def test_total_ms는_단계별_시간의_합이다():
    r = DecodeResult(
        text="123",
        symbology="CODE128",
        retry_count=0,
        stage_ms={"detect": 40.0, "restore": 200.0, "decode": 20.0},
    )
    assert r.total_ms == 260.0


def test_판독_성공이면_ok는_True다():
    original = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = _crop()
    result = PipelineResult(
        original_bgr_uint8=original,
        detected=crop,
        restored=RestoredBarcode(
            image_gray_uint8=np.zeros((20, 60), dtype=np.uint8),
            source=crop,
        ),
        decode=DecodeResult(text="123", symbology="CODE128", retry_count=0),
    )
    assert result.ok is True


def test_탐지_실패여도_PipelineResult는_만들어진다():
    """파이프라인은 예외를 던지지 않는다. 실패는 값으로 표현한다 (설계 §5)."""
    result = PipelineResult(
        original_bgr_uint8=np.zeros((100, 200, 3), dtype=np.uint8),
        detected=None,
        restored=None,
        decode=DecodeResult(
            text=None,
            symbology=None,
            retry_count=0,
            failure_reason="not_detected",
        ),
    )
    assert result.ok is False
    assert result.decode.failure_reason == "not_detected"
