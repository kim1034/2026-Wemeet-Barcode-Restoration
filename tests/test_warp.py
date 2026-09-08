import math

import numpy as np

from wemeet.data.render import render_clean
from wemeet.data.surface import grad_crease, grad_cylinder, grad_sine
from wemeet.data.warp import apply_warp, fit_obs_width, flat_coord


def _cylinder_geometry(w, theta_deg):
    th = math.radians(theta_deg)
    return (w - 1) / 2.0, (w - 1) / (2.0 * math.sin(th))


def test_cumsum_of_inverse_m_matches_arcsin_within_half_pixel():
    """설계 §10-1. J = 1/m 를 적분하면 r*arcsin((x-c)/r) 이다."""
    w, theta = 401, 45.0
    zx, _ = grad_cylinder(w, 8, theta)
    _, s_hat, s_total = flat_coord(zx)

    c, r = _cylinder_geometry(w, theta)
    x = np.arange(w, dtype=np.float64) - c
    analytic = r * (np.arcsin(x / r) - np.arcsin(-c / r))
    numeric = s_hat[0] * s_total

    assert np.abs(numeric - analytic).max() < 0.5


def test_cumsum_of_m_fails_the_same_comparison():
    """설계 §10-2. 오류판(cumsum(m))은 위 대조에서 크게 벗어난다."""
    w, theta = 401, 45.0
    zx, _ = grad_cylinder(w, 8, theta)
    m = 1.0 / np.sqrt(1.0 + zx ** 2)

    wrong = np.cumsum(m, axis=1) - m[:, :1]
    c, r = _cylinder_geometry(w, theta)
    x = np.arange(w, dtype=np.float64) - c
    analytic = r * (np.arcsin(x / r) - np.arcsin(-c / r))

    assert np.abs(wrong[0] - analytic).max() > 5.0


def test_m_never_exceeds_one():
    """설계 §10-7."""
    for theta in (10.0, 30.0, 60.0, 75.0):
        zx, _ = grad_cylinder(301, 8, theta)
        m, _, _ = flat_coord(zx)
        assert m.max() <= 1.0 + 1e-12
        assert m.min() > 0.0


def test_m_and_slope_stay_consistent():
    """설계 §10-6. sqrt(1/m^2 - 1) 이 |z_x| 와 같아야 한다."""
    zx, _ = grad_crease(301, 8, slope=1.5, w_c=4.0)
    m, _, _ = flat_coord(zx)
    recovered = np.sqrt(1.0 / m ** 2 - 1.0)
    assert np.abs(recovered - np.abs(zx)).max() < 1e-9


def test_s_hat_is_normalised_per_row():
    zx, _ = grad_crease(201, 8, slope=1.0, w_c=3.0, psi_deg=20.0)
    _, s_hat, _ = flat_coord(zx)
    assert np.allclose(s_hat[:, 0], 0.0)
    assert np.allclose(s_hat[:, -1], 1.0)


def test_s_total_is_the_row_mean_not_any_single_row():
    """s_total 은 행 평균이어야 한다. grad_cylinder 는 모든 행이 동일해 이 계약을
    검증하지 못한다 (설계 §10) -- 행마다 총합이 실제로 다른 grad_sine 픽스처를 쓴다.
    """
    zx, _ = grad_sine(201, 60, slope=1.0, lam=60.0, psi_deg=30.0)
    _, _, s_total = flat_coord(zx)

    m = 1.0 / np.sqrt(1.0 + zx ** 2)
    row_totals = np.cumsum(1.0 / m, axis=1)[:, -1] - (1.0 / m)[:, 0]

    # 이 픽스처가 실제로 행마다 다른 총합을 갖는지 먼저 확인 (아니면 아래 대조가 공허하다)
    assert row_totals.max() - row_totals.min() > 1.0

    assert abs(s_total - row_totals.mean()) < 1e-9
    assert abs(s_total - row_totals[0]) > 1e-9
    assert abs(s_total - row_totals[-1]) > 1e-9
    assert abs(s_total - row_totals.max()) > 1e-9
    assert abs(s_total - row_totals.min()) > 1e-9


def test_fit_obs_width_recovers_flat_length():
    """감기면 좁아 보인다. 펴진 길이가 w_flat 이 되도록 관측 폭을 맞춘다."""
    w_flat, h = 400, 8
    w_obs, zx, _ = fit_obs_width(
        lambda w, hh: grad_cylinder(w, hh, 50.0), w_flat, h)
    _, _, s_total = flat_coord(zx)
    assert w_obs < w_flat
    assert abs(s_total - w_flat) / w_flat < 0.05


def test_apply_warp_shape_and_dtype():
    clean = render_clean("WEMEET0001", module_px=4.0, height_px=220)
    zx, _ = grad_crease(301, 220, slope=1.0, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat)
    assert obs.shape == s_hat.shape
    assert obs.dtype == np.uint8
