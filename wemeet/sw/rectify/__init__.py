"""Stage 3 (기하 보정) — GeometryField 의 제어점으로 실제 픽셀을 편다.

AI가 예측한 것은 좌표뿐이다 (`wemeet/schemas.py` 의 `GeometryField`). 여기서
그 좌표로 TPS(Thin Plate Spline) 매핑을 풀고 `cv2.remap` 으로 기존 픽셀을
옮긴다 — 새 픽셀을 만들지 않는다.

`cv2.createThinPlateSplineShapeTransformer()` 는 쓰지 않는다.
`opencv-contrib-python` 이 필요한데 이 프로젝트는 base `opencv-python` 만
쓴다 (`pyproject.toml`). numpy로 (N+3)×(N+3) 선형계를 직접 푼다.

여러 파일로 나눠 구현해도 됩니다 — 바깥(`wemeet.sw.pipeline`)에는 이
`__init__.py` 가 내보내는 `apply_field()` 하나만 보이면 됩니다.
"""

import cv2
import numpy as np

from wemeet.schemas import DetectedBarcode, GeometryField, RectifiedBarcode


def _tps_flow(
    dst_px: np.ndarray, src_px: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """제어점 대응으로 dense flow map(`map_x`, `map_y`)을 만든다.

    `dst_px` 를 정의역으로 계수를 푼다. `cv2.remap` 은 "출력 픽셀이 입력
    어디서 오는가"를 원하므로, 출력(dst) 좌표에서 평가했을 때 입력(src)
    좌표를 내놓는 함수여야 한다. 반대로 풀면(=src 를 정의역으로 풀면)
    왜곡이 두 번 적용되는데, 약한 왜곡에서는 우연히 읽혀서 놓치기 쉽다.
    """
    n = len(dst_px)
    d = np.linalg.norm(dst_px[:, None, :] - dst_px[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(d > 0, d**2 * np.log(d**2), 0.0)
    p = np.hstack([np.ones((n, 1)), dst_px])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T

    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    dg = np.linalg.norm(grid[:, None, :] - dst_px[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ug = np.where(dg > 0, dg**2 * np.log(dg**2), 0.0)
    pg = np.hstack([np.ones((len(grid), 1)), grid])

    out = []
    for axis in (0, 1):
        b = np.concatenate([src_px[:, axis], np.zeros(3)])
        coef = np.linalg.solve(a, b)
        out.append((ug @ coef[:n] + pg @ coef[n:]).reshape(h, w).astype(np.float32))
    return out[0], out[1]


def apply_field(
    target: DetectedBarcode,
    field: GeometryField,
    interpolation: int = cv2.INTER_CUBIC,
) -> RectifiedBarcode:
    gray = cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = np.array([w - 1, h - 1], dtype=np.float64)
    dst_px = field.control_points_dst_norm * scale
    src_px = field.control_points_src_norm * scale

    try:
        map_x, map_y = _tps_flow(dst_px, src_px, (h, w))
    except np.linalg.LinAlgError:
        # 제어점이 특이(거의 일직선 등)라 TPS 선형계를 풀 수 없다. 보정을
        # 포기하고 원본 크롭을 그대로 통과시킨다 — 파이프라인은 이런 입력
        # 하나 때문에 죽으면 안 된다 (docs/architecture.md).
        return RectifiedBarcode(image_gray_uint8=gray, source=target, field=None)

    warped = cv2.remap(gray, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)
    return RectifiedBarcode(image_gray_uint8=warped, source=target, field=field)
