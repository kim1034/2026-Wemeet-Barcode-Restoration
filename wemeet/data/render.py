"""깨끗한 바코드 렌더링. python-barcode 의존을 이 파일에 가둔다."""

import io

import barcode
import cv2
import numpy as np
from barcode.writer import ImageWriter

_DPI = 300
_BASE_MODULE_PX = 8  # 기본 렌더 폭 — 정수 픽셀 경계로 정확함
_QUIET_MODULES = 20  # 양쪽 10 모듈씩. 아래 quiet_zone 과 같은 값이어야 한다


def module_count(text: str) -> int:
    """콰이엇 존을 포함한 총 모듈 개수. 래스터화하지 않는다.

    python-barcode 의 인코더가 모듈 문자열을 그대로 준다. 종횡비를 유도하려면
    렌더 '전에' 폭을 알아야 하는데, 이 방법이면 이미지를 만들지 않고 알 수 있다.
    """
    return len(barcode.get("code128", text).build()[0]) + _QUIET_MODULES


def render_clean(text: str, module_px: float, height_px: int) -> np.ndarray:
    """모듈 폭이 정확히 module_px 가 되는 Code128 을 그린다.

    height_px 에 기본값을 두지 않는다 — 예전의 H_OBS=220 은 물리적으로 틀린
    상수였다 (설계 §1.1). 호출부가 종횡비에서 유도한 값을 반드시 넘겨야 한다.
    """
    base_module_mm = _BASE_MODULE_PX * 25.4 / _DPI
    quiet_zone_mm = (_QUIET_MODULES // 2) * base_module_mm

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

    # ImageWriter 가 위아래에 흰 여백을 6.67% 씩 붙인다 (설계 §1.3). 그대로 두면
    # 종횡비가 15% 어긋나고, 흰 화소가 정반사에서 포화해 sat_ratio 를 부풀리고,
    # geometric_margin 이 이미 모사하는 검출기 여백과 이중 계상된다.
    # 비율을 상수로 박지 않고 이미지에서 찾는다 — 라이브러리가 바뀌어도 맞는다.
    rows = np.where(arr.min(axis=1) < 128)[0]
    arr = arr[rows.min():rows.max() + 1]

    # 모듈 개수는 '렌더된 폭' 에서 낸다. module_count() 를 다시 부르면 위의
    # test_module_count_matches_rendered_width 가 동어반복이 된다 -- 그 테스트의
    # 값어치는 인코더 경로와 라이터 경로가 독립으로 같은 답을 낸다는 데 있다.
    M = arr.shape[1] / _BASE_MODULE_PX
    target_width = int(round(M * module_px))
    return cv2.resize(arr, (target_width, height_px), interpolation=cv2.INTER_AREA)
