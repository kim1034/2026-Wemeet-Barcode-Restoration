"""Stage 1 (검출) — 바코드 영역을 찾아 기울기 보정 후 크롭한다.

@AI파트가 구현할 패키지입니다. docs/architecture.md, wemeet/schemas.py 참고.

여러 파일로 나눠 구현해도 됩니다 — 바깥(`wemeet.sw.pipeline`)에는 이
`__init__.py` 가 내보내는 `detect()` 하나만 보이면 됩니다.

이 패키지는 wemeet.sw / wemeet.data 를 import 하지 않습니다 (import-linter
규칙, docs/architecture.md 「의존 방향」). 공용이 필요하면 wemeet.schemas
만 씁니다.
"""

import numpy as np

from wemeet.schemas import DetectedBarcode


def detect(image_bgr: np.ndarray) -> DetectedBarcode | None:
    """사진에서 바코드 영역을 찾아 기울기 보정 후 크롭한다.

    - 이 프로젝트는 사진 한 장에 바코드가 하나만 있다고 가정한다
    - 바코드가 없으면 None 을 반환한다 (예외를 던지지 않는다)
    - confidence 는 0.0~1.0
    """
    raise NotImplementedError("AI파트가 구현할 부분입니다.")
