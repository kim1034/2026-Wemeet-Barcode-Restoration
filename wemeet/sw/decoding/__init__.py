"""Stage 4 (디코딩) — pyzbar 우선 시도, 실패 시 zxing-cpp 폴백.

`decode()` 는 `RectifiedBarcode` 하나를 받아 순수하게 판독만 한다. TPS로
편 이미지든(`field` 있음) 보정 전 원본 크롭이든(`field=None`) 같은 타입으로
들어오므로 이 함수 안에서 그 차이를 구분하지 않는다 — 어느 쪽이든 온 것을
그대로 판독만 시도한다.

**원본으로 못 읽으면 CLAHE 로 대비를 올린 이미지로 한 번 더 읽는다.** 비닐막 반사가
막대 일부만 옅게 만들면 기하가 멀쩡해도 못 읽는데, 이때 대비 보정이 먹힌다. 원본을
먼저 읽으므로 원래 읽히던 이미지는 결과가 같고, 1차 판독과 보정 후 판독이 모두
`decode()` 를 쓰므로 양쪽에 적용된다. `enhance=False` 로 끌 수 있다 (재시도 루프와는
별개다 — 같은 이미지를 한 번 더 읽을 뿐이다).

지원 심볼로지: Code128, EAN-13.

재시도 루프는 여기 없다. `pipeline.py` 가 돈다 — 재시도는 보정 파라미터를
바꿔 `apply_field` 부터 다시 부르는 것이라 `decode()` 혼자서는 결정할 수
없다. `retry_count`·`stage_ms`도 마찬가지로 `pipeline.py`가 채우는 값이라
여기서는 기본값 그대로 둔다.

여러 파일로 나눠 구현해도 됩니다 — 바깥(`wemeet.sw.pipeline`)에는 이
`__init__.py` 가 내보내는 `decode()` 하나만 보이면 됩니다.
"""

import cv2
import numpy as np
from PIL import Image  # 파이썬에서 이미지를 다루는 라이브러리

from wemeet.schemas import DecodeResult, RectifiedBarcode

# 실제 웹캠 크롭 6장에서 고른 값 (원본 2장 → CLAHE 5장 판독). 검출용 CLAHE
# (`wemeet.ai.detection.enhance`)와 별개다 — 이쪽은 흑백을 받고 설정도 다르다.
_CLAHE_CLIP_LIMIT = 4.0
_CLAHE_TILE_GRID = (4, 4)

# pyzbar 는 OS 레벨 공유 라이브러리(zbar)가 있어야 동작하고, import 하는 순간
# 그 DLL 을 불러온다. 없는 환경에서는 그 백엔드만 조용히 빠지게 한다.
#   · Linux(CI): 시스템에 libzbar0 가 없으면 ImportError
#   · Windows: 휠에 DLL 은 번들돼 있지만 VC++ 2013 런타임(msvcr120.dll)이 없으면
#     FileNotFoundError, 32/64비트가 섞이면 OSError. 둘 다 OSError 계열이다.
try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
except (ImportError, OSError):
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


def _read(gray: np.ndarray) -> str | None:
    gray_image = Image.fromarray(gray)
    text = _decode_with_pyzbar(gray_image)
    if text is None:
        text = _decode_with_zxingcpp(gray_image)
    return text


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    # 호출마다 만든다 — 서버가 스레드풀에서 돌리는데 CLAHE 객체는 공유하면 안전하지 않다
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID)
    return clahe.apply(gray)


def decode(image: RectifiedBarcode, enhance: bool = True) -> DecodeResult:
    gray = image.image_gray_uint8

    text = _read(gray)
    if text is None and enhance:
        text = _read(_enhance_contrast(gray))

    if text is None:
        return DecodeResult(text=None, retry_count=0, failure_reason="decode_failed")

    return DecodeResult(text=text, retry_count=0)
