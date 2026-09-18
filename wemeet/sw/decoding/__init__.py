"""Stage 4 (디코딩) — pyzbar 우선 시도, 실패 시 zxing-cpp 폴백.

`decode()` 는 `RectifiedBarcode` 하나를 받아 순수하게 판독만 한다. TPS로
편 이미지든(`field` 있음) 보정 전 원본 크롭이든(`field=None`) 같은 타입으로
들어오므로 이 함수 안에서 그 차이를 구분하지 않는다 — 어느 쪽이든 온 것을
그대로 판독만 시도한다.

지원 심볼로지: Code128, EAN-13.

재시도 루프는 여기 없다. `pipeline.py` 가 돈다 — 재시도는 보정 파라미터를
바꿔 `apply_field` 부터 다시 부르는 것이라 `decode()` 혼자서는 결정할 수
없다. `retry_count`·`stage_ms`도 마찬가지로 `pipeline.py`가 채우는 값이라
여기서는 기본값 그대로 둔다.

여러 파일로 나눠 구현해도 됩니다 — 바깥(`wemeet.sw.pipeline`)에는 이
`__init__.py` 가 내보내는 `decode()` 하나만 보이면 됩니다.
"""

from PIL import Image  # 파이썬에서 이미지를 다루는 라이브러리

from wemeet.schemas import DecodeResult, RectifiedBarcode

# pyzbar 는 OS 레벨 공유 라이브러리(zbar)가 있어야 동작한다. Windows 휠은
# 그 라이브러리를 번들해서 문제없지만, Linux(CI)는 시스템에 libzbar0 가
# 따로 설치돼 있지 않으면 import 자체가 ImportError 로 죽는다. zxingcpp 와
# 같은 방식으로 방어해서, 없는 환경에서는 그 백엔드만 조용히 빠지게 한다.
try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
except ImportError:
    _pyzbar_decode = None

try:
    import zxingcpp as _zxingcpp
except ImportError:
    _zxingcpp = None


def _decode_with_pyzbar(gray_image: Image.Image) -> str | None:
    if _pyzbar_decode is None:
        return None
    try:
        found = _pyzbar_decode(gray_image)
        if not found:
            return None
        return found[0].data.decode()
    except Exception:
        return None


def _decode_with_zxingcpp(gray_image: Image.Image) -> str | None:
    if _zxingcpp is None:
        return None
    try:
        for result in _zxingcpp.read_barcodes(gray_image):
            if result.valid and result.text:
                return result.text
        return None
    except Exception:
        return None


def decode(image: RectifiedBarcode) -> DecodeResult:
    gray_image = Image.fromarray(image.image_gray_uint8)

    text = _decode_with_pyzbar(gray_image)
    if text is None:
        text = _decode_with_zxingcpp(gray_image)

    if text is None:
        return DecodeResult(text=None, retry_count=0, failure_reason="decode_failed")

    return DecodeResult(text=text, retry_count=0)
