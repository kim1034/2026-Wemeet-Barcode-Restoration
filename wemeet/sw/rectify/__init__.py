"""Stage 3 (기하 보정) — GeometryField 의 제어점으로 실제 픽셀을 편다.

AI가 예측한 것은 좌표뿐이다 (`wemeet/schemas.py` 의 `GeometryField`). 여기서
그 좌표로 매핑을 풀고 `cv2.remap` 으로 기존 픽셀을 옮긴다 — 새 픽셀을 만들지 않는다.

`cv2.createThinPlateSplineShapeTransformer()` 는 쓰지 않는다.
`opencv-contrib-python` 이 필요한데 이 프로젝트는 base `opencv-python` 만
쓴다 (`pyproject.toml`). numpy로 (N+3)×(N+3) 선형계를 직접 푼다 (`tps.py`).

입력 → 출력
    target.crop_bgr_uint8   (H, W, 3) BGR uint8   펼 대상. 정규화 좌표의 기준 크기
    field                   정규화 제어점 dst/src (N, 2). 확정 격자는 16×3 = 48점
    →  RectifiedBarcode.image_gray_uint8  (round(H·out_scale), round(W·out_scale)) uint8

보정할 수 없으면(제어점이 특이·비정상) 예외 대신 원본
크롭을 흑백으로 그대로 통과시키고 `field=None` 으로 표시한다 — 파이프라인은 이런
입력 하나 때문에 죽으면 안 된다 (docs/architecture.md).

여러 파일로 나눠 구현해도 됩니다 — 바깥(`wemeet.sw.pipeline`)에는 이
`__init__.py` 가 내보내는 `apply_field()` 하나만 보이면 됩니다.
"""

import math

import cv2
import numpy as np

from wemeet.schemas import DetectedBarcode, GeometryField, RectifiedBarcode
from wemeet.sw.rectify.tps import tps_maps

# TPS 매핑을 평가하는 격자 간격(입력 크롭 기준 px). 4 = 1/4 격자.
# 전체 픽셀 평가는 warp 예산(24 ms)의 수 배다 (docs/stage2-handoff.md 「격자 축소」).
# 출력 배율을 키워도 평가 점 개수가 늘지 않도록 출력 기준 간격은 이 값 × out_scale 이다.
_FLOW_STEP_PX = 4.0


def _passthrough(target: DetectedBarcode, gray: np.ndarray) -> RectifiedBarcode:
    return RectifiedBarcode(image_gray_uint8=gray, source=target, field=None)


def _as_points(points) -> np.ndarray | None:
    """제어점을 float64 (N, 2) 로 바꾼다. 비정상이면 None.

    schemas.py 는 `.shape` 만 검사하므로 torch.Tensor 같은 것도 통과해 들어온다.
    여기서 numpy 로 강제 변환해 뒤쪽 연산이 깊은 곳에서 터지지 않게 한다.
    """
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        return None
    return arr


def apply_field(
    target: DetectedBarcode,
    field: GeometryField,
    interpolation: int = cv2.INTER_CUBIC,
    out_scale: float = 1.0,
) -> RectifiedBarcode:
    """크롭을 제어점대로 펴서 흑백 이미지로 돌려준다.

    Args:
        target: 1단계 결과. `crop_bgr_uint8` 의 원본 크기가 정규화 좌표의 기준이다.
        field: 2단계 결과. dst = 펴진 격자, src = 그 내용이 크롭에서 지금 있는 위치.
        interpolation: `cv2.remap` 보간법 (`cv2.INTER_*`).
        out_scale: 출력 캔버스 크기 = 크롭 크기 × out_scale. TPS 커널은 스케일 불변이
            아니라서 이 값이 보간 결과 자체를 바꾼다 (docs/stage3-rectify-guide.md 3절).
            파이프라인이 재시도 축으로 쓴다.

    Returns:
        RectifiedBarcode. 성공하면 `field` 에 입력 field 가, 보정을 포기하면 None 이 들어간다.
    """
    if not (math.isfinite(out_scale) and out_scale > 0):
        raise ValueError(f"out_scale 은 양의 유한수여야 한다: {out_scale}")

    gray = cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY)
    in_h, in_w = gray.shape
    out_h = max(2, round(in_h * out_scale))
    out_w = max(2, round(in_w * out_scale))

    dst_norm = _as_points(field.control_points_dst_norm)
    src_norm = _as_points(field.control_points_src_norm)
    if dst_norm is None or src_norm is None or dst_norm.shape != src_norm.shape:
        return _passthrough(target, gray)

    # 1.0 은 마지막 픽셀의 "중심" 이다. w 가 아니라 w-1 을 곱한다.
    # dst 는 출력 캔버스 기준, src 는 입력 크롭 기준 — 두 캔버스 크기가 다를 수 있다.
    dst_px = dst_norm * np.array([out_w - 1, out_h - 1], dtype=np.float64)
    src_px = src_norm * np.array([in_w - 1, in_h - 1], dtype=np.float64)

    try:
        map_x, map_y = tps_maps(dst_px, src_px, (out_h, out_w), step=_FLOW_STEP_PX * out_scale)
    except np.linalg.LinAlgError:
        # 제어점이 특이(거의 일직선, 중복 등)라 TPS 선형계를 풀 수 없다.
        return _passthrough(target, gray)
    if not (np.isfinite(map_x).all() and np.isfinite(map_y).all()):
        return _passthrough(target, gray)

    # src 는 크롭 밖(-0.5~1.5)을 가리킬 수 있다. 기본값(검정 채움)이면 그 영역이
    # 막대로 읽히므로 가장자리를 복제한다.
    warped = cv2.remap(gray, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)
    return RectifiedBarcode(image_gray_uint8=warped, source=target, field=field)
