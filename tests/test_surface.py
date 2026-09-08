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
