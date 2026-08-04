import numpy as np
import pytest

from wemeet.schemas import (
    DecodeResult,
    DetectedBarcode,
    GeometryField,
    PipelineResult,
    RectifiedBarcode,
)


def _crop(h: int = 20, w: int = 60) -> DetectedBarcode:
    """정상적인 DetectedBarcode 하나. 다른 테스트의 재료로 쓴다."""
    return DetectedBarcode(
        crop_bgr_uint8=np.zeros((h, w, 3), dtype=np.uint8),
        angle_deg_ccw=0.0,
        confidence=0.9,
    )


def _field(n: int = 12) -> GeometryField:
    """정상적인 GeometryField 하나. 격자 좌표는 0~1 정규화다."""
    xs = np.linspace(0.0, 1.0, 4)
    ys = np.linspace(0.0, 1.0, n // 4)
    gx, gy = np.meshgrid(xs, ys)
    dst = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return GeometryField(
        control_points_dst_norm=dst,
        control_points_src_norm=dst + 0.01,
        method="tps",
        confidence=0.8,
    )


# ── DetectedBarcode ────────────────────────────────────────────────


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


# ── GeometryField ──────────────────────────────────────────────────


def test_제어점이_4개보다_적으면_거부된다():
    """TPS 를 풀 수 없다. 실측으로 8개를 권장한다 (설계 §8.2)."""
    with pytest.raises(AssertionError):
        GeometryField(
            control_points_dst_norm=np.zeros((3, 2)),
            control_points_src_norm=np.zeros((3, 2)),
            method="tps",
            confidence=0.5,
        )


def test_src와_dst_제어점_개수가_다르면_거부된다():
    with pytest.raises(AssertionError):
        GeometryField(
            control_points_dst_norm=np.zeros((8, 2)),
            control_points_src_norm=np.zeros((6, 2)),
            method="tps",
            confidence=0.5,
        )


def test_좌표가_xy_두_개가_아니면_거부된다():
    with pytest.raises(AssertionError):
        GeometryField(
            control_points_dst_norm=np.zeros((8, 3)),
            control_points_src_norm=np.zeros((8, 3)),
            method="tps",
            confidence=0.5,
        )


def test_모르는_보정_방식은_거부된다():
    """'restore' 같은 값이 들어오면 막는다. 우리는 이미지를 생성하지 않는다."""
    with pytest.raises(AssertionError):
        GeometryField(
            control_points_dst_norm=np.zeros((8, 2)),
            control_points_src_norm=np.zeros((8, 2)),
            method="restore",
            confidence=0.5,
        )


def test_기하_추정_신뢰도가_범위를_넘으면_거부된다():
    with pytest.raises(AssertionError):
        GeometryField(
            control_points_dst_norm=np.zeros((8, 2)),
            control_points_src_norm=np.zeros((8, 2)),
            method="tps",
            confidence=1.5,
        )


def test_기하_추정_결과에는_이미지가_없다():
    """AI 가 픽셀을 만들어 반환할 수 없다는 것이 계약이다 (설계 §2.1)."""
    field = _field()
    이미지처럼_보이는_필드 = [
        name
        for name in vars(field)
        if "image" in name or "bgr" in name or "gray" in name or "crop" in name
    ]
    assert 이미지처럼_보이는_필드 == []


# ── RectifiedBarcode ───────────────────────────────────────────────


def test_보정_결과가_흑백_단일채널이_아니면_거부된다():
    with pytest.raises(AssertionError):
        RectifiedBarcode(
            image_gray_uint8=np.zeros((10, 10, 3), dtype=np.uint8),
            source=_crop(),
            field=_field(),
        )


def test_보정하지_않고_통과시킬_수_있다():
    """field=None 은 '보정 없이 1차 디코딩을 시도했다'를 뜻한다 (설계 §6)."""
    plain = RectifiedBarcode(
        image_gray_uint8=np.zeros((20, 60), dtype=np.uint8),
        source=_crop(),
        field=None,
    )
    assert plain.field is None


# ── DecodeResult ───────────────────────────────────────────────────


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
        stage_ms={
            "detect": 40.0,
            "decode_first": 15.0,
            "estimate": 120.0,
            "warp": 18.0,
            "decode": 25.0,
        },
    )
    assert r.total_ms == 218.0


# ── PipelineResult ─────────────────────────────────────────────────


def test_판독_성공이면_ok는_True다():
    crop = _crop()
    result = PipelineResult(
        original_bgr_uint8=np.zeros((100, 200, 3), dtype=np.uint8),
        detected=crop,
        rectified=RectifiedBarcode(
            image_gray_uint8=np.zeros((20, 60), dtype=np.uint8),
            source=crop,
            field=_field(),
        ),
        decode=DecodeResult(text="123", symbology="CODE128", retry_count=0),
    )
    assert result.ok is True


def test_탐지_실패여도_PipelineResult는_만들어진다():
    """파이프라인은 예외를 던지지 않는다. 실패는 값으로 표현한다 (설계 §5)."""
    result = PipelineResult(
        original_bgr_uint8=np.zeros((100, 200, 3), dtype=np.uint8),
        detected=None,
        rectified=None,
        decode=DecodeResult(
            text=None,
            symbology=None,
            retry_count=0,
            failure_reason="not_detected",
        ),
    )
    assert result.ok is False
    assert result.decode.failure_reason == "not_detected"


def test_보정하지_못해도_결과가_나온다():
    """기하 추정을 신뢰할 수 없으면 보정 없이 진행하고 degraded 로 표시한다."""
    crop = _crop()
    result = PipelineResult(
        original_bgr_uint8=np.zeros((100, 200, 3), dtype=np.uint8),
        detected=crop,
        rectified=RectifiedBarcode(
            image_gray_uint8=np.zeros((20, 60), dtype=np.uint8),
            source=crop,
            field=None,
        ),
        decode=DecodeResult(
            text=None,
            symbology=None,
            retry_count=3,
            failure_reason="decode_failed",
            degraded=True,
        ),
    )
    assert result.ok is False
    assert result.decode.degraded is True


# ── 픽스처 ─────────────────────────────────────────────────────────


def test_렌더링한_바코드는_DetectedBarcode에_그대로_들어간다(clean_barcode_bgr):
    """픽스처가 계약이 요구하는 형식(BGR, uint8, 3채널)으로 이미지를 준다."""
    detected = DetectedBarcode(
        crop_bgr_uint8=clean_barcode_bgr,
        angle_deg_ccw=0.0,
        confidence=1.0,
    )
    assert detected.crop_bgr_uint8.ndim == 3
    assert detected.crop_bgr_uint8.shape[2] == 3
    assert detected.crop_bgr_uint8.dtype == np.uint8
