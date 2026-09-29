"""Stage 1~4 를 이어 붙인다 — 4단계를 부르는 지휘자.

`docs/architecture.md` 「코드가 흐르는 순서」 그대로다. 어떤 상황에서도
예외를 던지지 않고 `PipelineResult` 를 반환한다 — 현장 컨베이어에서 예외가
올라오면 서비스가 멈추기 때문이다. 실패는 예외가 아니라 값으로 표현한다.

이 프로젝트는 사진 한 장에 바코드가 하나만 있다고 가정한다 — 그래서
`detect()` 도 리스트가 아니라 `DetectedBarcode | None` 하나만 반환한다.

단계마다 걸린 시간을 `DecodeResult.stage_ms` 에 기록한다. 500ms 예산을
초과해도 **중단하지 않는다** — 중단하면 읽을 수 있었던 화물을 버리는 것이고,
무엇보다 초과 원인을 알 수 없게 된다 (docs/architecture.md 「시간 예산」).

여러 파일로 나눠 구현해도 됩니다 — 바깥(`server.py` 등)에는 이
`__init__.py` 가 내보내는 `run()` 하나만 보이면 됩니다.
"""

import time

import cv2
import numpy as np

from wemeet.ai.detection import detect
from wemeet.ai.geometry import estimate_geometry
from wemeet.schemas import (
    DecodeResult,
    DetectedBarcode,
    GeometryField,
    PipelineResult,
    RectifiedBarcode,
)
from wemeet.sw.decoding import decode
from wemeet.sw.rectify import apply_field

# 재시도에서 바꾸는 것은 **출력 캔버스 배율**이다. 기하 추정(AI, 150ms)은 다시
# 돌리지 않고 같은 GeometryField 를 재사용한다 — 기하 보정(OpenCV)만 다시 한다.
#
# 보간법 3종을 돌리던 것을 배율로 바꿨다. TPS 커널은 스케일 불변이 아니라서
# 출력 크기가 보간 자체를 바꾸고, 추론 시점에는 펴진 폭(w_flat)을 알 수 없다.
# 평가 세트 실측(docs/stage3-rectify-guide.md 3절):
#
#     크롭과 같은 크기(1.0) 한 번      75%
#     배율 3.0 한 번                   89%
#     배율 1.0 → 1.5 → 3.0 (이 값)     95%
#
# 보간법은 어느 배율에서든 INTER_CUBIC 로 고정한다.
_RETRY_OUT_SCALES = (1.0, 1.5, 3.0)

# 초기값. 근거 없는 값이라 AI파트가 실측 후 조정한다 (docs/architecture.md
# 「아직 정하지 않은 것」). 현재 2단계 모델은 확신도를 학습하지 않아 항상
# 1.0 을 반환하므로 이 분기는 실제로는 타지 않는다 — 확신도를 내는 모델로
# 교체될 때를 위해 남겨둔다 (docs/stage2-handoff.md 「아직 안 정해진 것 ①」).
_MIN_GEOMETRY_CONFIDENCE = 0.3


def _timed(stage_ms: dict[str, float], key: str, func, *args, **kwargs):
    """func 를 부르고 걸린 시간을 stage_ms[key] 에 **누적**한다.

    누적인 이유는 재시도 때문이다 — `warp` 는 최대 3번 불리는데, 예산
    초과 여부를 보려면 그 합이 필요하다.
    """
    start = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        stage_ms[key] = stage_ms.get(key, 0.0) + (time.perf_counter() - start) * 1000.0


def _with_timing(result: DecodeResult, stage_ms: dict[str, float]) -> DecodeResult:
    result.stage_ms = stage_ms
    return result


def _rectify_and_decode(
    detected: DetectedBarcode, field: GeometryField, stage_ms: dict[str, float]
) -> tuple[RectifiedBarcode, DecodeResult]:
    """배율을 바꿔 가며 최대 3회 편다. 마지막 시도의 (보정 결과, 판독 결과)."""
    rectified, result = None, None
    for attempt, out_scale in enumerate(_RETRY_OUT_SCALES):
        rectified = _timed(stage_ms, "warp", apply_field, detected, field, out_scale=out_scale)
        result = _timed(stage_ms, "decode" if attempt == 0 else "retry", decode, rectified)
        result.retry_count = attempt
        if result.text is not None:
            break
    return rectified, result


def run(image_bgr: np.ndarray) -> PipelineResult:
    stage_ms: dict[str, float] = {}

    try:
        detected = _timed(stage_ms, "detect", detect, image_bgr)
    except Exception:
        detected = None

    if detected is None:
        return PipelineResult(
            original_bgr_uint8=image_bgr,
            detected=None,
            rectified=None,
            decode=_with_timing(
                DecodeResult(text=None, retry_count=0, failure_reason="not_detected"),
                stage_ms,
            ),
        )

    # 1차 시도 — 보정 없이 먼저 읽어본다. 수직으로만 휜 바코드는 여기서
    # 끝난다 (실측: 진폭 32px 까지 읽힌다).
    plain = RectifiedBarcode(
        image_gray_uint8=cv2.cvtColor(detected.crop_bgr_uint8, cv2.COLOR_BGR2GRAY),
        source=detected,
        field=None,
    )
    result = _timed(stage_ms, "decode_first", decode, plain)
    if result.text is not None:
        return PipelineResult(image_bgr, detected, plain, _with_timing(result, stage_ms))

    try:
        field = _timed(stage_ms, "estimate", estimate_geometry, detected)
    except Exception:
        # AI 코드가 예외를 던져도 파이프라인은 멈추지 않는다 — 보정 없이
        # 진행한 1차 결과를 degraded 로 표시해 돌려준다.
        result.degraded = True
        return PipelineResult(image_bgr, detected, plain, _with_timing(result, stage_ms))

    if field.confidence < _MIN_GEOMETRY_CONFIDENCE:
        # 추정을 신뢰할 수 없다. 신뢰할 수 없는 필드로 펴면 오히려 더
        # 안 읽힌다 — 보정을 건너뛰고 1차 결과를 degraded 로 표시한다.
        result.degraded = True
        return PipelineResult(image_bgr, detected, plain, _with_timing(result, stage_ms))

    rectified, result = _rectify_and_decode(detected, field, stage_ms)
    return PipelineResult(image_bgr, detected, rectified, _with_timing(result, stage_ms))
