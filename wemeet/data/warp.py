"""기울기 → 매핑 → 관측 이미지. 방향 함정을 이 파일에 가둔다.

규약 (설계 §2):
    x = 관측 좌표, s = 펴진 좌표
    J = ds/dx = sqrt(1 + z_x^2) >= 1     신장률
    m = dx/ds = 1/J <= 1                 압축률
    s(x) = cumsum(J) = cumsum(1/m)       <- cumsum(m) 이 아니다
"""

import cv2
import numpy as np


def flat_coord(zx: np.ndarray):
    """관측 -> 펴진 매핑. 행마다 가로로만 적분한다 (v = y 근사).

    돌려주는 것은 (m, s_hat, s_total):
        m       압축률 (h, w).  0 < m <= 1
        s_hat   행마다 0~1 로 정규화한 펴진 좌표 (h, w)
        s_total 펴진 총 길이(px). 행 평균
    """
    m = 1.0 / np.sqrt(1.0 + zx ** 2)
    integrand = 1.0 / m
    s = np.cumsum(integrand, axis=1) - integrand[:, :1]
    s_hat = s / s[:, -1:]
    return m, s_hat, float(s[:, -1].mean())


def fit_obs_width(make_grad, w_flat: int, h: int, iters: int = 4):
    """펴진 길이가 w_flat 이 되도록 관측 폭을 맞춘다.

    감기면 좁아 보이므로 관측 폭은 펴진 폭보다 작다. 고정점 반복으로 찾는다.
    """
    w = w_flat
    for _ in range(iters):
        zx, zy = make_grad(w, h)
        _, _, s_total = flat_coord(zx)
        w_new = max(32, int(round(w * w_flat / s_total)))
        if w_new == w:
            break
        w = w_new
    zx, zy = make_grad(w, h)
    return w, zx, zy


def apply_warp(clean: np.ndarray, s_hat: np.ndarray) -> np.ndarray:
    """깨끗한 이미지를 관측 이미지로 찌그러뜨린다."""
    h_obs, w_obs = s_hat.shape
    h0, w0 = clean.shape
    map_x = (s_hat * (w0 - 1)).astype(np.float32)
    gy = np.linspace(0, h0 - 1, h_obs, dtype=np.float32)
    map_y = np.tile(gy[:, None], (1, w_obs))
    return cv2.remap(clean, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)
