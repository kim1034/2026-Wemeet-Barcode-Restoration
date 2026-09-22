"""Stage 1 (검출) — 바코드 영역을 찾아 기울기 보정 후 크롭한다.

@AI파트가 구현할 패키지입니다. docs/architecture.md, wemeet/schemas.py 참고.

여러 파일로 나눠 구현해도 됩니다 — 바깥(`wemeet.sw.pipeline`)에는 이
`__init__.py` 가 내보내는 `detect()` 하나만 보이면 됩니다.

이 패키지는 wemeet.sw / wemeet.data 를 import 하지 않습니다 (import-linter
규칙, docs/architecture.md 「의존 방향」). 공용이 필요하면 wemeet.schemas
만 씁니다.

---

구현: YOLO11n-OBB 단일 클래스 + 박스 모양으로 규격 판정.
측정치와 근거는 docs/experiments/2026-09-22-stage1-detection 에 있습니다.

    합성 val 2,000장 · mAP50 0.973 · mAP50-95 0.943 · 추론 4.1ms (RTX 4050)

**ultralytics 가 없어도 이 모듈은 import 됩니다.** 실제로 `detect()` 를 부를
때 처음 필요해집니다 (yolo_obb.py 의 라이선스 주석 참고).

검출 입력에는 CLAHE 대비 향상을 적용합니다(`contrast.py`) — 크롭 자체는
원본에서 뜨므로 Stage 2 도메인과는 무관합니다.
"""

import numpy as np

from wemeet.ai.detection.contrast import enhance_for_detection
from wemeet.ai.detection.geometry import CROP_PAD, crop_upright, decide_ratio, fit_ratio, upright
from wemeet.ai.detection.yolo_obb import DEFAULT_IMGSZ, load_model
from wemeet.schemas import DetectedBarcode

__all__ = ["detect"]

#: 이 값 미만은 바코드로 치지 않는다.
CONF_THRESHOLD = 0.25
#: 검출 입력에 CLAHE 대비 향상을 적용할지. 끄면 원본 그대로 모델에 들어간다.
USE_CLAHE = True


def detect(image_bgr: np.ndarray) -> DetectedBarcode | None:
    """사진에서 바코드 영역을 찾아 기울기 보정 후 크롭한다.

    - 이 프로젝트는 사진 한 장에 바코드가 하나만 있다고 가정한다
    - 바코드가 없으면 None 을 반환한다 (예외를 던지지 않는다)
    - confidence 는 0.0~1.0

    크롭은 **막대 영역을 규격 비율(5:3 또는 6:4)로 맞추고 각 변에 10% 여백을
    붙인 것**이다. 여백은 2단계 기하 추정이 막대 끝을 판단할 여지를 남긴다.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.size == 0:
        return None

    detection_input = enhance_for_detection(image_bgr) if USE_CLAHE else image_bgr
    result = load_model().predict(
        detection_input, imgsz=DEFAULT_IMGSZ, conf=CONF_THRESHOLD, verbose=False
    )[0]
    obb = result.obb
    if obb is None or len(obb) == 0:
        return None

    best = int(obb.conf.argmax())
    cx, cy, w, h, angle_rad = (float(v) for v in obb.xywhr[best].cpu().numpy())
    cx, cy, w, h, angle_rad = upright(cx, cy, w, h, angle_rad)

    ratio, _ = decide_ratio(w, h)
    w, h = fit_ratio(w, h, ratio)

    crop, angle_deg_ccw = crop_upright(image_bgr, cx, cy, w, h, angle_rad, pad=CROP_PAD)
    if crop.size == 0:
        return None

    return DetectedBarcode(
        crop_bgr_uint8=np.ascontiguousarray(crop, dtype=np.uint8),
        angle_deg_ccw=angle_deg_ccw,
        confidence=float(obb.conf[best]),
    )
