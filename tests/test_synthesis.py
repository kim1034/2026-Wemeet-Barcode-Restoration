import dataclasses
import hashlib

import numpy as np

from wemeet.data.synthesis import BUCKETS, build, draw_recipe, recipe_from_dict, recipe_to_dict


def test_bucket_ranges_cover_the_spec_span():
    assert BUCKETS["L"][0] == 1.6
    assert BUCKETS["H"][1] == 6.0
    assert BUCKETS["L"][1] == BUCKETS["M"][0]
    assert BUCKETS["M"][1] == BUCKETS["H"][0]


def test_draw_recipe_respects_bucket():
    rng = np.random.default_rng(1)
    for name, (lo, hi) in BUCKETS.items():
        for i in range(50):
            r = draw_recipe(rng, name, i)
            assert lo <= r.d_m0 <= hi
            assert 1.3 <= r.d_t <= 2.6
            assert r.d_t < 0.97 * r.d_m0


def test_draw_recipe_is_deterministic_for_same_seed():
    a = draw_recipe(np.random.default_rng(7), "L", 3)
    b = draw_recipe(np.random.default_rng(7), "L", 3)
    assert recipe_to_dict(a) == recipe_to_dict(b)


def test_recipe_round_trips_through_dict():
    r = draw_recipe(np.random.default_rng(2), "M", 0)
    assert recipe_from_dict(recipe_to_dict(r)) == r


def test_build_is_bit_reproducible():
    """설계 §10-9. 같은 레시피 두 번 -> 이미지 바이트가 동일하다."""
    r = draw_recipe(np.random.default_rng(11), "L", 0)
    a, b = build(r), build(r)
    assert hashlib.sha256(a.obs.tobytes()).hexdigest() == \
           hashlib.sha256(b.obs.tobytes()).hexdigest()
    assert np.array_equal(a.src_norm, b.src_norm)


def test_build_honours_the_contract_ranges():
    rng = np.random.default_rng(5)
    for i in range(8):
        s = build(draw_recipe(rng, "L", i))
        assert s.dst_norm.shape == (18, 2)
        assert s.src_norm.shape == (18, 2)
        assert s.dst_norm.min() >= 0.0 and s.dst_norm.max() <= 1.0
        assert s.src_norm.min() >= -0.5 and s.src_norm.max() <= 1.5


def test_build_reports_m_min_and_saturation():
    s = build(draw_recipe(np.random.default_rng(3), "L", 0))
    assert 0.0 < s.m_min <= 1.0
    assert 0.0 <= s.sat_ratio <= 1.0
    assert 0.0 < s.scale <= 1.0


def test_slope_target_is_hit_not_just_bounded():
    """S_t 는 상한이 아니라 목표다. 최소 모듈 폭이 d_t 근처여야 한다."""
    rng = np.random.default_rng(9)
    for i in range(6):
        r = draw_recipe(rng, "L", i)
        s = build(r)
        assert abs(s.m_min * r.d_m0 - r.d_t) < 0.25 * r.d_t


def test_geometric_augmentation_precedes_control_points():
    """설계 §5: 제어점(⑤)은 기하 증강(④) 이후의 최종 G 에서 뽑는다.

    조립 순서가 뒤바뀌어 증강 전 G 로 제어점을 뽑으면, rot_deg 를 바꿔도
    src_norm 이 전혀 달라지지 않는다 — 이 테스트는 그 순서 버그를 잡는다.
    """
    rng = np.random.default_rng(42)
    r = draw_recipe(rng, "L", 0)
    r0 = dataclasses.replace(r, rot_deg=0.0)
    r1 = dataclasses.replace(r, rot_deg=8.0)
    s0, s1 = build(r0), build(r1)
    assert not np.allclose(s0.src_norm, s1.src_norm, atol=1e-6)


def test_psi_actually_shades_via_z_y():
    """z_y = z_x*tan(psi) 는 psi != 0 일 때만 0 이 아니다 (surface.py).

    Task 5 의 음영 테스트는 psi=0 픽스처만 써서 z_y 가 한 번도 검증되지
    않았다. 실제 분포는 psi ~ U(-45, 45) 를 뽑으므로, 여기서 psi 가
    음영 결과에 실제로 반영되는지 값으로 확인한다.
    """
    rng = np.random.default_rng(1)
    r = None
    for i in range(50):
        candidate = draw_recipe(rng, "L", i)
        if candidate.preset == "sine":
            r = candidate
            break
    assert r is not None, "sine 프리셋을 뽑지 못했다 — 시드나 분포를 확인하라"
    r0 = dataclasses.replace(r, psi=0.0)
    r1 = dataclasses.replace(r, psi=30.0)
    s0, s1 = build(r0), build(r1)
    assert not np.array_equal(s0.obs, s1.obs)
