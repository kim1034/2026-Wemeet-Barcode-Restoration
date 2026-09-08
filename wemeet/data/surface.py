"""레시피 파라미터 → 높이장 기울기. 설계 §3 수식의 집.

전부 (z_x, z_y) 를 돌려준다. slope 인자는 언제나 "이 성분의 최대 |z_x|" 다 —
진폭이 아니다. 그래야 기울기 예산을 성분끼리 선형으로 나눠 쓸 수 있다.
"""

import math

import numpy as np


def grad_cylinder(w: int, h: int, theta_deg: float):
    """전역 원통 감김. 관측 폭이 w 일 때 가장자리 기울기가 tan(theta)."""
    if theta_deg <= 0:
        return np.zeros((h, w)), np.zeros((h, w))
    th = math.radians(theta_deg)
    c = (w - 1) / 2.0
    r = (w - 1) / (2.0 * math.sin(th))
    x = np.arange(w, dtype=np.float64) - c
    zx = -x / np.sqrt(np.maximum(r * r - x * x, 1e-9))
    return np.tile(zx, (h, 1)), np.zeros((h, w))


def _axis(w: int, h: int, psi_deg: float):
    """능선 축 t = (x-cx)cos(psi) + (y-cy)sin(psi) 와 cos/sin."""
    psi = math.radians(psi_deg)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    t = (gx - (w - 1) / 2) * math.cos(psi) + (gy - (h - 1) / 2) * math.sin(psi)
    return t, math.cos(psi), math.sin(psi)


def grad_sine(w, h, slope, lam, psi_deg=0.0, phase=0.0):
    """물결 주름. slope = 이 성분의 최대 |z_x|."""
    t, cp, sp = _axis(w, h, psi_deg)
    d = slope * np.cos(2 * np.pi * t / lam + phase)
    return d, d * (sp / cp if cp else 0.0)


def grad_crease(w, h, slope, w_c, psi_deg=0.0, offset=0.0):
    """접힌 능선. 기울기가 -slope -> +slope 로 tanh 전이."""
    t, cp, sp = _axis(w, h, psi_deg)
    d = slope * np.tanh((t - offset) / max(w_c, 1e-6))
    return d, d * (sp / cp if cp else 0.0)


def octave_weights(octaves: int, persistence: float) -> np.ndarray:
    """옥타브별 기울기 가중치. 합이 1 이 되게 정규화한다.

    스파이크는 균등(persistence=1.0)이었다. 실측에서 균등은 목표 구간 5%,
    0.7 은 10% 다 (설계 §3).
    """
    w = np.array([persistence ** j for j in range(octaves)], dtype=np.float64)
    return w / w.sum()


def grad_crumple(w, h, slope, lam0, rng, octaves=3, persistence=0.7,
                 psi_spread=25.0):
    """여러 스케일 능선의 합 = 구김. 옥타브마다 파장이 절반, 기울기가 persistence 배."""
    weights = octave_weights(octaves, persistence)
    zx = np.zeros((h, w))
    zy = np.zeros((h, w))
    for j, weight in enumerate(weights):
        a, b = grad_sine(w, h, slope * weight, lam0 / (2 ** j),
                         psi_deg=float(rng.uniform(-psi_spread, psi_spread)),
                         phase=float(rng.uniform(0, 2 * np.pi)))
        zx += a
        zy += b
    return zx, zy


def slope_budget(d_m0: float, c_min: float = 1.0) -> float:
    """판독 하한 d_m0 * m_min >= c_min 에서 나오는 최대 허용 기울기 S_max.

    v1 은 c_min = 1.0 이다. 2.0 을 쓰면 d_m0 < 2.0 에서 정의되지 않아
    목표 구간이 열리는 저해상도 절반이 표현 불가가 된다 (설계 §3).
    """
    ratio = d_m0 / c_min
    if ratio < 1.0:
        raise ValueError(
            f"d_m0={d_m0} 가 c_min={c_min} 보다 작아 S_max 가 정의되지 않는다"
        )
    return math.sqrt(ratio ** 2 - 1.0)


def limit_slope(zx, zy, s_max: float):
    """초과분을 clip 하지 않고 기울기를 통째로 줄인다.

    np.clip(m, ...) 은 안 된다 — m 만 자르면 대응하는 z 가 없어져서
    음영·반사가 보여주는 곡률과 기하가 어긋난다 (실측 법선각 최대 11.32°).
    """
    peak = float(np.abs(zx).max())
    scale = min(1.0, s_max / peak) if peak > 0 else 1.0
    return zx * scale, zy * scale, scale
