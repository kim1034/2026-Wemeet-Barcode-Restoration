import dataclasses
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from wemeet.data.optics import shade
from wemeet.data.synthesis import (
    ALL_BUCKETS,
    ASPECT_HI,
    ASPECT_LO,
    BUCKETS,
    EVAL_BUCKETS,
    K_A,
    K_D,
    N_X,
    N_Y,
    PRESETS,
    Recipe,
    _make_grad,
    build,
    draw_recipe,
    recipe_from_dict,
    recipe_to_dict,
)
from wemeet.data.warp import control_points


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
        assert s.dst_norm.shape == (N_X * N_Y, 2)
        assert s.src_norm.shape == (N_X * N_Y, 2)
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


def test_build_hands_back_the_dense_field_that_produced_the_control_points():
    """제어점 개수 실험(2026-09-09)이 요구한다. 제어점은 G 에서 뽑히므로
    "제어점 개수와 무관한 복원 상한" 을 재려면 증강까지 끝난 G 가 필요하다.
    build() 안에서만 존재하면 실험 코드가 build() 를 통째로 복제해야 하고,
    그러면 본체가 바뀔 때 조용히 어긋난다.

    필드가 있다는 것만 보면 아무 배열이나 넣어도 통과하므로, 돌려받은 G 에서
    control_points() 를 다시 뽑아 Sample 이 이미 담고 있는 제어점과 일치하는지로
    "그 제어점을 만든 바로 그 G" 임을 확인한다.
    """
    s = build(draw_recipe(np.random.default_rng(7), "M", 0))
    dst, src = control_points(s.g, N_X, N_Y, s.obs.shape, s.u_lo, s.u_hi)
    assert np.abs(dst - s.dst_norm).max() == 0.0
    assert np.abs(src - s.src_norm).max() == 0.0


def test_dense_field_is_the_augmented_one_not_the_pre_augmentation_one():
    """회전·여백 증강이 G 를 옮긴다. 증강 전 G 를 돌려주면 좌표가 통째로
    어긋나는데, 위 테스트는 그것도 통과시킬 수 있다 -- control_points 를
    같은(틀린) G 로 다시 뽑으면 역시 일치하기 때문이다.

    그래서 여기서는 s.u_lo/u_hi 가 실제로 증강 결과를 담고 있는지, 그리고
    G 가 증강된 이미지 안을 가리키는지를 본다. rot_deg 와 margin 을 크게
    준 레시피에서 증강 전 G 는 이 범위를 벗어난다.
    """
    base = draw_recipe(np.random.default_rng(9), "M", 0)
    r = replace(base, rot_deg=4.0, margin=(0.12, 0.12, 0.10, 0.10))
    s = build(r)
    h, w = s.obs.shape
    inside = s.g[:, :, 0]
    assert (s.u_hi - s.u_lo) > 0.0
    # 여백을 12% 씩 덧대면 G 의 x 는 크롭 폭의 가운데 쪽으로 몰린다.
    assert inside.min() > 0.0
    assert inside.max() < w - 1
    assert s.g[:, :, 1].max() <= h - 1


def test_h_obs_is_gone():
    """220 은 물리적으로 틀린 상수였다. 다시 스며들지 못하게 잠근다."""
    import wemeet.data.synthesis as syn
    assert not hasattr(syn, "H_OBS")


def test_aspect_is_drawn_in_range():
    rng = np.random.default_rng(0)
    for bucket in BUCKETS:
        for i in range(30):
            r = draw_recipe(rng, bucket, i)
            assert ASPECT_LO <= r.aspect <= ASPECT_HI


def test_flat_label_aspect_matches_recipe():
    """펴진 라벨의 w/h 가 레시피의 aspect 다 (설계 검증 #2).

    관측 크롭이 아니라 펴진 것을 잰다 — 감기면 좁아 보이는 것이 물리적으로 맞다.
    """
    rng = np.random.default_rng(1)
    for bucket in BUCKETS:
        for i in range(10):
            r = draw_recipe(rng, bucket, i)
            s = build(r)
            assert s.w_flat / s.h_flat == pytest.approx(r.aspect, abs=0.02)
            assert ASPECT_LO - 0.02 <= s.w_flat / s.h_flat <= ASPECT_HI + 0.02


def test_crease_transition_is_relative_to_width():
    """주름 전이가 이미지 폭의 '같은 비율' 을 차지한다 (설계 검증 #4).

    w_c 가 절대 픽셀이던 시절에는 폭이 3배가 되면 비율이 1/3 로 줄었다 —
    고해상도 버킷만 물리적으로 3.75배 날카로운 주름을 받았다.
    cyl_share=0 으로 두어 원통 성분을 끄고 주름만 본다.
    """
    r = Recipe(
        seed=1, bucket="L", text="WEMEET0000", d_m0=2.0, d_t=1.5, aspect=2.15,
        preset="crease", cyl_share=0.0, psi=0.0, w_c_f=0.029, lam_f=1.0,
        phase=0.0, offset=0.0, light=(0.0, 0.0, 1.0), ks=0.0, p=100.0,
        sigma=0.0, noise=0.0, jpeg=90, rot_deg=0.0, margin=(0.0, 0.0, 0.0, 0.0),
    )
    make = _make_grad(r, s_t=1.0)
    fractions = []
    for w in (600, 1800):
        zx, _ = make(w, 150)
        row = zx[75]
        peak = float(np.abs(row).max())
        inside = np.where(np.abs(row) < 0.8 * peak)[0]
        fractions.append((inside.max() - inside.min() + 1) / w)
    assert fractions[0] == pytest.approx(fractions[1], rel=0.05), fractions
    # 분석해: 2*atanh(0.8)*w_c_f = 0.0637
    assert fractions[0] == pytest.approx(2 * np.arctanh(0.8) * r.w_c_f, rel=0.05)


def test_text_wraps_at_10000():
    """10,000번째부터 텍스트가 길어지면 w_flat 이 d_m0 과 무관하게 움직인다 (설계 §2)."""
    a = draw_recipe(np.random.default_rng(0), "L", 0)
    b = draw_recipe(np.random.default_rng(0), "L", 10000)
    c = draw_recipe(np.random.default_rng(0), "L", 107999)
    assert a.text == b.text == "WEMEET0000"
    assert c.text == "WEMEET7999"


def test_eval_buckets_are_inside_train_buckets():
    """학습 분포가 평가 3벌을 덮어야 한다 — 안 덮으면 성능 차이가 분포 불일치가 된다."""
    for ev, tr in (("low", "L"), ("mid", "M"), ("high", "H")):
        elo, ehi = EVAL_BUCKETS[ev]
        tlo, thi = BUCKETS[tr]
        assert tlo <= elo and ehi <= thi, f"{ev} 가 {tr} 밖으로 나간다"
    assert set(ALL_BUCKETS) == set(BUCKETS) | set(EVAL_BUCKETS)


def test_recipe_roundtrip_rejects_old_recipes():
    """aspect·w_c_f 없는 옛 JSON 은 조용히 통과하면 안 된다 (설계 §8).

    옛 레시피를 새 파이프라인으로 재생성하면 다른 이미지가 나오므로,
    조용한 성공이 조용한 오염이 된다.
    """
    from wemeet.data.synthesis import recipe_from_dict, recipe_to_dict
    r = draw_recipe(np.random.default_rng(2), "M", 3)
    d = recipe_to_dict(r)
    assert recipe_from_dict(d) == r
    old = {k: v for k, v in d.items() if k not in ("aspect", "w_c_f")}
    old["w_c"] = 4.0
    with pytest.raises(TypeError):
        recipe_from_dict(old)


def test_sample_carries_h_flat():
    s = build(draw_recipe(np.random.default_rng(3), "L", 0))
    assert isinstance(s.h_flat, int) and s.h_flat > 0
    assert dataclasses.fields(type(s))[7].name == "h_flat"
