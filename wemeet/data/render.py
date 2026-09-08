"""깨끗한 바코드 렌더링. python-barcode 의존을 이 파일에 가둔다."""

import io

import barcode
import cv2
import numpy as np
from barcode.writer import ImageWriter

_DPI = 300


def render_clean(text: str, module_px: float, height_px: int = 220) -> np.ndarray:
    """모듈 폭이 정확히 module_px 가 되는 Code128 을 그린다.

    python-barcode 는 mm 로 받으므로 dpi 를 고정하고 환산한다.
    """
    module_mm = module_px * 25.4 / _DPI
    writer = ImageWriter()
    obj = barcode.get("code128", text, writer=writer)
    buf = io.BytesIO()
    obj.write(buf, options={
        "module_width": module_mm,
        "module_height": 12.0,
        "quiet_zone": 2.0,
        "write_text": False,
        "dpi": _DPI,
    })
    buf.seek(0)
    arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = arr.shape
    return cv2.resize(arr, (w, height_px), interpolation=cv2.INTER_AREA)
