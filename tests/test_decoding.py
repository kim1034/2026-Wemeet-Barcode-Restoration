"""wemeet/sw/decoding/ 의 decode() 검증.

pyzbar 우선 -> zxing-cpp 폴백 흐름 자체는 구현 세부다. 여기서는 계약
(DecodeResult) 이 지켜지는지만 본다: 성공하면 failure_reason 이 없고,
실패하면 반드시 채워진다 (schemas.py 의 __post_init__ 이 강제하는 규칙).
"""

import ctypes
import importlib
import io
import sys
from collections.abc import Callable

import cv2
import numpy as np
import pytest
from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image

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


def test_pyzbar_dll_load_failure_falls_back_to_zxing(
    monkeypatch: pytest.MonkeyPatch, clean_barcode_bgr: np.ndarray
) -> None:
    """VC++ 2013 런타임이 없는 Windows 처럼 zbar DLL 로드가 실패해도 zxing-cpp 로 읽는다."""
    import wemeet.sw.decoding as decoding

    def _missing_dll(name: str):
        raise FileNotFoundError(f"Could not find module '{name}' (or one of its dependencies)")

    for name in [m for m in sys.modules if m == "pyzbar" or m.startswith("pyzbar.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(ctypes.cdll, "LoadLibrary", _missing_dll)
    try:
        importlib.reload(decoding)
        assert decoding._pyzbar_decode is None

        gray = cv2.cvtColor(clean_barcode_bgr, cv2.COLOR_BGR2GRAY)
        assert decoding.decode(_rectified_from_gray(gray)).text == "WEMEET0001"
    finally:
        monkeypatch.undo()
        importlib.reload(decoding)


# ---------------------------------------------------------------- CLAHE 재시도


def _veiled_barcode(text: str = "WEMEET0001") -> np.ndarray:
    """왼쪽 40%를 반투명한 흰 막이 덮고 흐림·노이즈가 낀 Code128.

    비닐 반사막이 막대 일부만 옅게 만든 실제 크롭을 흉내 낸다. 막대 폭 약 4px 에서
    zxing 은 이 이미지를 원본으로는 못 읽고 CLAHE 후에는 읽는다 (노이즈 시드 20개 전부).
    """
    buffer = io.BytesIO()
    options = {"module_width": 0.4, "module_height": 14.0, "quiet_zone": 4.0, "dpi": 300}
    Code128(text, writer=ImageWriter()).write(buffer, options={**options, "write_text": False})
    buffer.seek(0)
    gray = np.asarray(Image.open(buffer).convert("L")).astype(np.float32)

    height, width = gray.shape
    _, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    edge = np.clip(xx / (width * 0.05), 0, 1) * np.clip((width * 0.4 - xx) / (width * 0.05), 0, 1)
    alpha = 0.9 * edge
    veiled = gray * (1 - alpha) + 228 * alpha
    blurred = cv2.GaussianBlur(veiled, (0, 0), 1.5)
    noisy = blurred + np.random.default_rng(0).normal(0, 4.0, blurred.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def test_decode_rescues_a_barcode_the_raw_image_cannot_read() -> None:
    veiled = _veiled_barcode()

    # 전제: 원본으로는 안 읽힌다. 디코더 버전이 바뀌어 읽히게 되면 이 테스트가 알려준다.
    assert decode(_rectified_from_gray(veiled), enhance=False).text is None

    result = decode(_rectified_from_gray(veiled))

    assert result.text == "WEMEET0001"
    assert result.failure_reason is None


def test_decode_does_not_touch_contrast_when_the_raw_image_reads(
    monkeypatch: pytest.MonkeyPatch, clean_barcode_bgr: np.ndarray
) -> None:
    """원본이 읽히면 CLAHE 를 부르지도 않는다 — 읽히던 이미지의 결과가 바뀔 수 없다."""
    import wemeet.sw.decoding as decoding

    def _must_not_run(gray: np.ndarray) -> np.ndarray:
        raise AssertionError("원본이 읽혔는데 CLAHE 가 불렸다")

    monkeypatch.setattr(decoding, "_enhance_contrast", _must_not_run)
    gray = cv2.cvtColor(clean_barcode_bgr, cv2.COLOR_BGR2GRAY)

    assert decode(_rectified_from_gray(gray)).text == "WEMEET0001"


def test_decode_enhance_false_never_runs_clahe(monkeypatch: pytest.MonkeyPatch) -> None:
    import wemeet.sw.decoding as decoding

    def _must_not_run(gray: np.ndarray) -> np.ndarray:
        raise AssertionError("enhance=False 인데 CLAHE 가 불렸다")

    monkeypatch.setattr(decoding, "_enhance_contrast", _must_not_run)
    result = decode(_rectified_from_gray(np.zeros((120, 300), dtype=np.uint8)), enhance=False)

    assert result.text is None
    assert result.failure_reason == "decode_failed"


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 50)], ids=["1x1", "2x2", "3x50"])
def test_decode_fails_cleanly_on_tiny_images(shape: tuple[int, int]) -> None:
    """파이프라인은 예외를 던지지 않는다 — CLAHE 가 작은 이미지에서 터지면 안 된다."""
    result = decode(_rectified_from_gray(np.full(shape, 128, dtype=np.uint8)))

    assert result.text is None
    assert result.failure_reason == "decode_failed"
