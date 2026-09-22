"""TPS(Thin Plate Spline) 매핑 계산. 픽셀 좌표만 다룬다 — 정규화 환산은 `__init__.py` 가 한다.

`cv2.remap` 은 "출력 픽셀 (x, y) 의 값을 입력 어디서 가져오나" 를 원한다. 그래서
여기서 푸는 함수는 **출력(dst) 좌표 → 입력(src) 좌표** 방향이다. 반대로 풀면
왜곡이 두 번 걸리는데, 약한 왜곡에서는 우연히 읽혀서 놓치기 쉽다
(docs/stage3-rectify-guide.md 2절).

전체 픽셀마다 TPS 를 평가하면 `warp` 예산(24 ms)의 수 배가 든다. 매핑은 매끄러우므로
거친 격자에서만 평가하고 이중선형으로 늘린다 (docs/stage2-handoff.md 「격자 축소」).
"""

import math

import cv2
import numpy as np


def _kernel(d2: np.ndarray) -> np.ndarray:
    """TPS 방사 기저 U(r) = r² log r². 입력은 거리의 제곱이다."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d2 > 0, d2 * np.log(d2), 0.0)


def solve_tps(dst_px: np.ndarray, src_px: np.ndarray) -> np.ndarray:
    """dst 를 정의역으로 하는 TPS 계수를 푼다. 반환 shape (N+3, 2) — 열이 x, y 축.

    제어점이 특이(일직선, 중복 등)하면 `np.linalg.LinAlgError` 를 던진다.
    `np.linalg.solve` 는 부동소수점 반올림 때문에 특이 행렬에서도 에러 없이 엉터리 해를
    내는 경우가 있어서(실측: 일직선 8점) 풀기 전에 직접 검사한다.
    """
    n = len(dst_px)
    diff = dst_px[:, None, :] - dst_px[None, :, :]
    d2 = (diff**2).sum(axis=2)
    p = np.hstack([np.ones((n, 1)), dst_px])
    if np.linalg.matrix_rank(p) < 3:
        raise np.linalg.LinAlgError("제어점이 한 직선 위에 있어 아핀 항을 풀 수 없다")
    if (d2[np.triu_indices(n, k=1)] < 1e-12).any():
        raise np.linalg.LinAlgError("dst 제어점에 중복이 있다")
    k = _kernel(d2)
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
    b = np.zeros((n + 3, 2))
    b[:n] = src_px
    return np.linalg.solve(a, b)


def _evaluate(coef: np.ndarray, dst_px: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """격자 xs × ys 위에서 TPS 를 평가한다. 반환은 (map_x, map_y), 각 shape (len(ys), len(xs))."""
    n = len(dst_px)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    diff = pts[:, None, :] - dst_px[None, :, :]
    u = _kernel((diff**2).sum(axis=2))
    out = u @ coef[:n] + coef[n] + pts @ coef[n + 1 :]
    shape = (len(ys), len(xs))
    return (
        out[:, 0].reshape(shape).astype(np.float32),
        out[:, 1].reshape(shape).astype(np.float32),
    )


def tps_maps(
    dst_px: np.ndarray,
    src_px: np.ndarray,
    out_shape: tuple[int, int],
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """`cv2.remap` 에 넣을 (map_x, map_y) 를 만든다. 각 shape = out_shape, float32.

    Args:
        dst_px: (N, 2) 출력 캔버스 기준 픽셀 좌표 (x, y). 펴진 격자.
        src_px: (N, 2) 입력 이미지 기준 픽셀 좌표 (x, y). 그 내용이 지금 있는 곳.
        out_shape: (높이, 너비). 출력 캔버스 크기.
        step: 평가 격자 간격(출력 px). 1 이면 모든 픽셀에서 평가(참조 구현과 동일),
            4 면 대략 1/4 격자에서 평가 후 늘린다.

    거친 격자는 **양 끝 픽셀을 포함하도록**(모서리 정렬) 잡는다. `cv2.resize` 는 픽셀
    중심 정렬이라 가장자리에서 값이 상수로 고정돼 최대 `step/2` px 어긋난다 — 그래서
    `cv2.remap` 으로 직접 늘린다.
    """
    out_h, out_w = out_shape
    coef = solve_tps(dst_px, src_px)
    if step <= 1.0:
        return _evaluate(coef, dst_px, np.arange(out_w, dtype=np.float64), np.arange(out_h))

    cw = max(2, math.ceil((out_w - 1) / step) + 1)
    ch = max(2, math.ceil((out_h - 1) / step) + 1)
    xs = np.linspace(0.0, out_w - 1, cw)
    ys = np.linspace(0.0, out_h - 1, ch)
    coarse_x, coarse_y = _evaluate(coef, dst_px, xs, ys)

    # 출력 픽셀 → 거친 격자 인덱스. 모서리 정렬이라 0 과 끝이 정확히 맞는다.
    ix = np.arange(out_w, dtype=np.float32) * np.float32((cw - 1) / max(out_w - 1, 1))
    iy = np.arange(out_h, dtype=np.float32) * np.float32((ch - 1) / max(out_h - 1, 1))
    grid_x, grid_y = np.meshgrid(ix, iy)
    map_x = cv2.remap(coarse_x, grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    map_y = cv2.remap(coarse_y, grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return map_x, map_y
