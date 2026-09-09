import math

import numpy as np
import pytest

from wemeet.data.surface import (grad_crease, grad_crumple, grad_cylinder,
                                 grad_sine, limit_slope, octave_weights,
                                 slope_budget)


def test_cylinder_edge_slope_is_tan_theta():
    zx, zy = grad_cylinder(401, 8, 45.0)
    assert abs(abs(zx).max() - math.tan(math.radians(45.0))) < 0.02
    assert np.allclose(zy, 0.0)


def test_cylinder_zero_angle_is_flat():
    zx, zy = grad_cylinder(101, 8, 0.0)
    assert np.allclose(zx, 0.0) and np.allclose(zy, 0.0)


def test_sine_peak_slope_equals_requested():
    zx, _ = grad_sine(400, 8, slope=0.8, lam=200.0)
    assert abs(abs(zx).max() - 0.8) < 0.01


def test_crease_saturates_to_plus_minus_slope():
    zx, _ = grad_crease(400, 8, slope=1.2, w_c=3.0)
    assert abs(zx.max() - 1.2) < 0.01
    assert abs(zx.min() + 1.2) < 0.01


def test_octave_weights_decay_by_persistence():
    w = octave_weights(3, 0.7)
    assert abs(w.sum() - 1.0) < 1e-12
    assert abs(w[1] / w[0] - 0.7) < 1e-12
    assert abs(w[2] / w[1] - 0.7) < 1e-12


def test_crumple_respects_total_slope():
    rng = np.random.default_rng(0)
    zx, _ = grad_crumple(400, 8, slope=1.0, lam0=300.0, rng=rng)
    assert abs(zx).max() <= 1.0 + 1e-9


def test_slope_budget_uses_c_min_one_by_default():
    assert abs(slope_budget(2.0) - math.sqrt(3.0)) < 1e-12
    assert abs(slope_budget(2.0, c_min=2.0) - 0.0) < 1e-12


def test_slope_budget_is_defined_below_two_px():
    """c_min=1.0 이라야 저해상도(d_m0 < 2.0)가 표현된다."""
    assert slope_budget(1.6) > 0.0
    with pytest.raises(ValueError):
        slope_budget(1.6, c_min=2.0)


def test_limit_slope_scales_both_axes_together():
    zx = np.array([[0.0, 4.0]])
    zy = np.array([[0.0, 2.0]])
    out_x, out_y, scale = limit_slope(zx, zy, s_max=2.0)
    assert abs(scale - 0.5) < 1e-12
    assert abs(out_x.max() - 2.0) < 1e-12
    assert abs(out_y.max() - 1.0) < 1e-12


def test_limit_slope_does_not_amplify():
    zx = np.array([[0.0, 1.0]])
    zy = np.zeros_like(zx)
    _, _, scale = limit_slope(zx, zy, s_max=5.0)
    assert scale == 1.0


def test_crumple_persistence_default_differs_from_uniform():
    """persistence=0.7 기본값이 uniform(1.0)과 다른 저주파 우위를 준다."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    # 기본값 (persistence=0.7) vs 명시적 uniform (persistence=1.0)
    zx_decay, _ = grad_crumple(200, 8, slope=1.0, lam0=200.0, rng=rng_a)
    zx_uniform, _ = grad_crumple(200, 8, slope=1.0, lam0=200.0, rng=rng_b, persistence=1.0)

    # 같은 seed 이므로 방향과 위상이 같음. 가중치만 다름.
    # decay(0.7) 는 고주파가 덜 포함되므로, 고주파만 필터링했을 때 에너지가 더 낮다.
    # 고주파 검출: 2차 미분(2-tap Laplacian)의 에너지
    hf_decay = float(np.sum(np.abs(np.diff(zx_decay, n=2, axis=1))))
    hf_uniform = float(np.sum(np.abs(np.diff(zx_uniform, n=2, axis=1))))
    assert hf_decay < hf_uniform, \
        f"persistence=0.7 should have lower high-freq energy ({hf_decay}) than uniform ({hf_uniform})"

    # 추가 검증: 필드들이 실제로 다르다 (같은 seed 이지만 가중치 때문에 다름)
    assert not np.allclose(zx_decay, zx_uniform), \
        "Fields with different persistence should differ"


def test_sine_max_zx_equals_slope_at_varied_psi():
    """grad_sine 계약: max|z_x| = slope. psi 에 무관하게."""
    for psi_deg in [0.0, 25.0, 45.0, -25.0, -45.0]:
        zx, _ = grad_sine(400, 8, slope=0.7, lam=150.0, psi_deg=psi_deg)
        max_zx = abs(zx).max()
        assert abs(max_zx - 0.7) < 0.01, \
            f"psi={psi_deg}°: max|z_x|={max_zx} should equal slope=0.7"


def test_crease_max_zx_equals_slope_at_varied_psi():
    """grad_crease 계약: max|z_x| = slope. psi 에 무관하게."""
    for psi_deg in [0.0, 25.0, 45.0, -25.0, -45.0]:
        zx, _ = grad_crease(400, 8, slope=0.9, w_c=2.5, psi_deg=psi_deg)
        max_zx = abs(zx).max()
        assert abs(max_zx - 0.9) < 0.01, \
            f"psi={psi_deg}°: max|z_x|={max_zx} should equal slope=0.9"


def test_sine_raises_on_near_singular_psi():
    """psi 가 ±90° 에 가까우면 ValueError."""
    with pytest.raises(ValueError, match="cos.*발산"):
        grad_sine(200, 8, slope=1.0, lam=100.0, psi_deg=89.5)
    with pytest.raises(ValueError, match="cos.*발산"):
        grad_sine(200, 8, slope=1.0, lam=100.0, psi_deg=-89.5)


def test_crease_raises_on_near_singular_psi():
    """psi 가 ±90° 에 가까우면 ValueError."""
    with pytest.raises(ValueError, match="cos.*발산"):
        grad_crease(200, 8, slope=1.0, w_c=3.0, psi_deg=89.5)
    with pytest.raises(ValueError, match="cos.*발산"):
        grad_crease(200, 8, slope=1.0, w_c=3.0, psi_deg=-89.5)
