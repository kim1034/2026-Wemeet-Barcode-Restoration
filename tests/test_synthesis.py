import dataclasses
import hashlib

import numpy as np

from wemeet.data.optics import shade
from wemeet.data.synthesis import (
    BUCKETS,
    K_A,
    K_D,
    PRESETS,
    build,
    draw_recipe,
    recipe_from_dict,
    recipe_to_dict,
)


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


def test_build_depends_on_the_seed_not_just_the_recipe_shape():
    """설계의 핵심 결정: 이미지는 저장하지 않고 레시피(seed 포함)만 남긴다.

    seed 만 다른 두 레시피가 같은 이미지를 내면, 저장된 레시피가 더 이상
    이미지를 설명하지 못한다. seed 를 무시하고 내부에서 고정 시드를 쓰는
    구현이라면 이 테스트가 잡는다 (a, b 가 byte-identical 이 됨).
    """
    rng = np.random.default_rng(2)
    r = draw_recipe(rng, "M", 0)
    a = build(dataclasses.replace(r, seed=111))
    b = build(dataclasses.replace(r, seed=222))
    assert not np.array_equal(a.obs, b.obs)


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
    """S_t 는 상한이 아니라 목표다. 최소 모듈 폭이 d_t 에 사실상 정확히
    맞아야 한다 (renormalize 를 하면 부동소수점 오차 수준, ~1e-15).

    리뷰에서 드러난 문제: 원래 폭(0.25*d_t)과 무작위 프리셋 추출만으로는
    seed 운이 나쁘면 통과한다 — crease/sine 은 renormalize 없이도 S_t 에
    거의 근접하고, octave(옥타브 합)만 크게 못 미친다. octave 는 preset
    확률이 0.30 이라 6회 추출에 한 번도 안 걸릴 확률이 상당하다(측정:
    seed(9) 자체는 1/6 이 걸려 잡지만, 300개 다른 시드 중 285개는
    ceiling-only 버그를 놓친다). 그래서 프리셋 네 가지를 전부 강제로
    순회해 discriminate 하는 조합이 반드시 들어가게 하고, 허용폭도
    실측 오차 대비 넉넉하되(1% of d_t) 놓치지 않을 만큼 좁힌다.
    """
    rng = np.random.default_rng(9)
    for preset in PRESETS:
        for i in range(3):
            r = dataclasses.replace(draw_recipe(rng, "L", i), preset=preset)
            s = build(r)
            assert abs(s.m_min * r.d_m0 - r.d_t) < 0.01 * r.d_t


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


def test_psi_changes_the_rendered_output():
    """psi 를 바꾸면 파이프라인 최종 출력이 달라진다.

    주의: psi 는 grad_sine 안에서 z_x 축 자체도 회전시키므로, 이 테스트
    하나만으로는 z_y 가 격리되어 검증되지 않는다 (psi 가 바뀌면 z_x 도
    바뀌어 warp 기하 자체가 달라진다 — 리뷰에서 지적된 지점). z_y 를
    격리한 검증은 test_shade_uses_z_y_not_just_z_x 를 보라.
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


def test_shade_uses_z_y_not_just_z_x():
    """Task 5 의 음영 테스트는 psi=0 픽스처만 써서 z_y 가 한 번도 검증되지
    않았다 (z_y = z_x*tan(psi) 는 psi=0 이면 항상 0). 실제 분포는
    psi ~ U(-45, 45) 를 뽑으므로 z_y 가 production 에서 실제로 쓰인다.

    전체 파이프라인을 거치면 psi 가 z_x 도 함께 바꿔 격리되지 않으므로
    (test_psi_changes_the_rendered_output 참고), 여기서는 shade() 를 직접
    불러 z_x 를 고정하고 z_y 만 바꿔 결과가 달라지는지 확인한다. z_y 를
    무시하는 shade 라면 이 테스트가 잡는다.
    """
    h, w = 40, 60
    base = np.full((h, w), 200, dtype=np.uint8)
    zx = np.full((h, w), 0.3)
    zy_zero = np.zeros((h, w))
    zy_nonzero = np.full((h, w), 0.3)
    light = (0.3, 0.4, 1.0)

    img_zero, _ = shade(base, zx, zy_zero, light, K_A, K_D)
    img_nonzero, _ = shade(base, zx, zy_nonzero, light, K_A, K_D)
    assert not np.array_equal(img_zero, img_nonzero)
