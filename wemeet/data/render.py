"""깨끗한 바코드 렌더링. python-barcode 의존을 이 파일에 가둔다."""

import io

import barcode
import cv2
import numpy as np
from barcode.writer import ImageWriter

_DPI = 300
_BASE_MODULE_PX = 8  # 기본 렌더 폭 — 정수 픽셀 경계로 정확함


def render_clean(text: str, module_px: float, height_px: int = 220) -> np.ndarray:
    """모듈 폭이 정확히 module_px 가 되는 Code128 을 그린다.

    python-barcode 는 정수 픽셀 경계로 반올림하므로, 정수 기본 모듈 폭으로 렌더링한 후
    연속 공간에서 한 번에 리샘플한다.
    """
    # 기본 모듈을 정수 픽셀로 렌더링
    base_module_mm = _BASE_MODULE_PX * 25.4 / _DPI
    # 콰이엇 존: 양쪽에 각각 10 모듈 (설계 규격과 일치)
    quiet_zone_mm = 10 * base_module_mm

    writer = ImageWriter()
    obj = barcode.get("code128", text, writer=writer)
    buf = io.BytesIO()
    obj.write(buf, options={
        "module_width": base_module_mm,
        "module_height": 12.0,
        "quiet_zone": quiet_zone_mm,
        "write_text": False,
        "dpi": _DPI,
    })
    buf.seek(0)
    arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    w = arr.shape[1]

    # 전체 모듈 개수 (기본 렌더에서 정확히 정수)
    M = w / _BASE_MODULE_PX

    # 목표 크기: 연속 공간에서 반올림
    target_width = int(round(M * module_px))

    # 한 번에 리샘플 (가로, 세로 모두)
    return cv2.resize(arr, (target_width, height_px), interpolation=cv2.INTER_AREA)
