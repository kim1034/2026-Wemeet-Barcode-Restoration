"""wemeet/sw/decoding/ 의 decode() 검증.

pyzbar 우선 -> zxing-cpp 폴백 흐름 자체는 구현 세부다. 여기서는 계약
(DecodeResult) 이 지켜지는지만 본다: 성공하면 failure_reason 이 없고,
실패하면 반드시 채워진다 (schemas.py 의 __post_init__ 이 강제하는 규칙).
"""

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from wemeet.schemas import DetectedBarcode, RectifiedBarcode
from wemeet.sw.decoding import decode


def _rectified_from_gray(gray: np.ndarray) -> RectifiedBarcode:
    detected = DetectedBarcode(
        crop_bgr_uint8=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        angle_deg_ccw=0.0,
        confidence=0.9,
    )
    return RectifiedBarcode(image_gray_uint8=gray, source=detected, field=None)


def test_decode_succeeds_on_clean_code128(clean_barcode_bgr: np.ndarray) -> None:
    gray = cv2.cvtColor(clean_barcode_bgr, cv2.COLOR_BGR2GRAY)

    result = decode(_rectified_from_gray(gray))

    assert result.text == "WEMEET0001"
    assert result.failure_reason is None


@pytest.mark.parametrize(
    "make_gray",
    [
        lambda: np.full((120, 300), 255, dtype=np.uint8),  # 빈 흰 이미지
        lambda: np.zeros((120, 300), dtype=np.uint8),  # 빈 검은 이미지
        lambda: np.random.default_rng(0).integers(0, 256, size=(120, 300), dtype=np.uint8),
    ],
    ids=["blank_white", "blank_black", "pure_noise"],
)
def test_decode_fails_on_unreadable_images(make_gray: Callable[[], np.ndarray]) -> None:
    result = decode(_rectified_from_gray(make_gray()))

    assert result.text is None
    assert result.failure_reason == "decode_failed"


def test_failure_reason_presence_matches_text(clean_barcode_bgr: np.ndarray) -> None:
    """schemas.py 의 규칙: text 가 있으면 failure_reason 은 없고, 없으면 반드시 있다."""
    gray = cv2.cvtColor(clean_barcode_bgr, cv2.COLOR_BGR2GRAY)
    success = decode(_rectified_from_gray(gray))
    failure = decode(_rectified_from_gray(np.zeros((120, 300), dtype=np.uint8)))

    assert success.text is not None
    assert success.failure_reason is None

    assert failure.text is None
    assert failure.failure_reason is not None
