"""Stage 1~4 를 이어 붙인다 — 4단계를 부르는 지휘자.

`docs/architecture.md` 「코드가 흐르는 순서」 그대로다. 어떤 상황에서도
예외를 던지지 않고 `PipelineResult` 를 반환한다 — 현장 컨베이어에서 예외가
올라오면 서비스가 멈추기 때문이다. 실패는 예외가 아니라 값으로 표현한다.

이 프로젝트는 사진 한 장에 바코드가 하나만 있다고 가정한다 — 그래서
`detect()` 도 리스트가 아니라 `DetectedBarcode | None` 하나만 반환한다.

여러 파일로 나눠 구현해도 됩니다 — 바깥(`server.py` 등)에는 이
`__init__.py` 가 내보내는 `run()` 하나만 보이면 됩니다.
"""

import cv2
import numpy as np

from wemeet.ai.detection import detect
from wemeet.ai.geometry import estimate_geometry
from wemeet.schemas import DecodeResult, PipelineResult, RectifiedBarcode
from wemeet.sw.decoding import decode
from wemeet.sw.rectify import apply_field

# 재시도에서 바꾸는 것은 보간법뿐이다. 기하 추정(AI, 150ms)은 다시 돌리지
# 않고 같은 GeometryField 를 재사용한다 — 기하 보정(OpenCV, 20ms)만 다시
# 한다 (docs/architecture.md 「재시도에서 무엇을 바꾸나」).
_RETRY_INTERPOLATIONS = (cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR)

# 초기값. 근거 없는 값이라 AI파트가 실측 후 조정한다 (docs/architecture.md
# 「아직 정하지 않은 것」).
_MIN_GEOMETRY_CONFIDENCE = 0.3


def run(image_bgr: np.ndarray) -> PipelineResult:
    try:
        detected = detect(image_bgr)
    except Exception:
        detected = None

    if detected is None:
        return PipelineResult(
            original_bgr_uint8=image_bgr,
            detected=None,
            rectified=None,
            decode=DecodeResult(
                text=None,
                retry_count=0,
                failure_reason="not_detected",
            ),
        )

    # 1차 시도 — 보정 없이 먼저 읽어본다. 수직으로만 휜 바코드는 여기서
    # 끝난다 (실측: 진폭 32px 까지 읽힌다).
    plain = RectifiedBarcode(
        image_gray_uint8=cv2.cvtColor(detected.crop_bgr_uint8, cv2.COLOR_BGR2GRAY),
        source=detected,
        field=None,
    )
    result = decode(plain)
    if result.text is not None:
        return PipelineResult(image_bgr, detected, plain, result)

    try:
        field = estimate_geometry(detected)
    except Exception:
        # AI 코드가 예외를 던져도 파이프라인은 멈추지 않는다 — 보정 없이
        # 진행한 1차 결과를 degraded 로 표시해 돌려준다.
        result.degraded = True
        return PipelineResult(image_bgr, detected, plain, result)

    if field.confidence < _MIN_GEOMETRY_CONFIDENCE:
        # 추정을 신뢰할 수 없다. 신뢰할 수 없는 필드로 펴면 오히려 더
        # 안 읽힌다 — 보정을 건너뛰고 1차 결과를 degraded 로 표시한다.
        result.degraded = True
        return PipelineResult(image_bgr, detected, plain, result)

    # 보정 + 디코딩. 실패하면 보간법을 바꿔 최대 3회 재시도한다.
    rectified = plain
    for i, interpolation in enumerate(_RETRY_INTERPOLATIONS):
        rectified = apply_field(detected, field, interpolation=interpolation)
        result = decode(rectified)
        result.retry_count = i
        if result.text is not None:
            return PipelineResult(image_bgr, detected, rectified, result)

    return PipelineResult(image_bgr, detected, rectified, result)
