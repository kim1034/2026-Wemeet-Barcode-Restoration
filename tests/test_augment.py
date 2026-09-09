import inspect

import numpy as np

from wemeet.data.augment import geometric_margin, geometric_rotate, photometric
from wemeet.data.surface import grad_crease
from wemeet.data.warp import build_G, control_points, flat_coord


def _fixture():
    zx, _ = grad_crease(301, 220, slope=1.2, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    img = np.full(s_hat.shape, 200, dtype=np.uint8)
    return img, build_G(s_hat)


def test_rotate_round_trip_restores_G():
    img, g = _fixture()
    img2, g2 = geometric_rotate(img, g, 7.0)
    _, g3 = geometric_rotate(img2, g2, -7.0)
    assert np.abs(g3 - g).max() < 1e-6


def test_rotate_keeps_image_shape():
    img, g = _fixture()
    img2, g2 = geometric_rotate(img, g, 12.0)
    assert img2.shape == img.shape
    assert g2.shape == g.shape


def test_rotate_actually_moves_G():
    """round-trip 만으로는 G 를 안 옮기는(no-op) 구현도 통과한다 -- +7도 후 -7도를
    그대로 되돌리면 애초에 안 옮긴 것과 옮긴 것을 구분 못 한다(둘 다 diff=0).
    그래서 단일 회전이 실제로 좌표를 바꾸는지 별도로 확인한다.
    """
    img, g = _fixture()
    _, g2 = geometric_rotate(img, g, 7.0)
    assert not np.allclose(g2, g)


def test_padding_keeps_full_u_range():
    img, g = _fixture()
    img2, _, u_lo, u_hi = geometric_margin(img, g, 0.08, 0.05, 0.05, 0.05)
    assert (u_lo, u_hi) == (0.0, 1.0)
    assert img2.shape[1] > img.shape[1]


def test_cutting_shrinks_u_range():
    img, g = _fixture()
    img2, _, u_lo, u_hi = geometric_margin(img, g, -0.02, -0.02, 0.0, 0.0)
    assert img2.shape[1] < img.shape[1]
    assert u_lo > 0.0 and u_hi < 1.0


def test_margin_is_clamped_to_spec_range():
    img, g = _fixture()
    _, _, lo_a, hi_a = geometric_margin(img, g, -0.30, 0.0, 0.0, 0.0)
    _, _, lo_b, hi_b = geometric_margin(img, g, -0.02, 0.0, 0.0, 0.0)
    assert abs(lo_a - lo_b) < 1e-9 and abs(hi_a - hi_b) < 1e-9


def test_asymmetric_margin_shifts_G_by_the_correct_side_not_swapped():
    """네 변을 한 함수에서 처리하다 left/right 나 top/bottom 인자가 뒤바뀌면
    폭 증가량(합)은 그대로라 폭만 보는 검사로는 안 잡힌다. 이동량이 정확히
    left/top 인자에서 나온 값과 같은지 좌표로 직접 확인한다.
    """
    img, g = _fixture()
    h, w = img.shape
    img2, g2, _, _ = geometric_margin(img, g, 0.08, 0.02, 0.05, 0.01)
    expected_left_shift = round(0.08 * w)
    expected_top_shift = round(0.05 * h)
    assert abs((g2[..., 0].min() - g[..., 0].min()) - expected_left_shift) < 1e-6
    assert abs((g2[..., 1].min() - g[..., 1].min()) - expected_top_shift) < 1e-6


def test_mixed_pad_and_cut_shifts_G_and_narrows_u_range_correctly():
    """플랜 자체 리뷰가 지목한 최대 리스크: 한 변은 패딩(+), 다른 변은
    컷(-)인 "섞인 부호" 조합은 순수 패딩(test_padding_keeps_full_u_range)
    이나 대칭 컷(test_cutting_shrinks_u_range)으로는 건드려지지 않는다 --
    좌우/상하 각각의 pad·crop 변수가 뒤바뀌는 인덱스 슬립은 패딩만 있거나
    대칭으로 자르기만 할 때는 숨어 있다가, 한쪽은 패딩하고 다른 쪽은 잘라먹을
    때만 드러난다.

    기대값은 실행 결과를 베끼지 않고 대수적으로 유도한다: 패딩과 크롭은
    서로 배타적이라(dl>=0 이면 패딩만 있고, dl<0 이면 크롭만 있다) 좌표
    이동량은 부호와 무관하게 항상 정확히 dl(왼쪽), dt(위쪽) 그 자체다 --
    오른쪽/아래쪽 인자는 캔버스 크기만 바꾸고 좌표는 옮기지 않는다.
    u_lo/u_hi 는 원본(미변환) g 에 그 dl 과 새 폭만 적용해 독립적으로
    다시 계산해 맞춘다.
    """
    img, g = _fixture()
    h, w = img.shape
    left, right, top, bottom = -0.02, 0.10, 0.05, -0.01
    dl, dr = round(left * w), round(right * w)
    dt, db = round(top * h), round(bottom * h)

    img2, g2, u_lo, u_hi = geometric_margin(img, g, left, right, top, bottom)

    assert img2.shape == (h + dt + db, w + dl + dr)
    assert np.allclose(g2[..., 0] - g[..., 0], dl)
    assert np.allclose(g2[..., 1] - g[..., 1], dt)

    new_w = w + dl + dr
    moved_x = g[..., 0] + dl
    inside = ((moved_x >= 0) & (moved_x <= new_w - 1)).all(axis=0)
    us = np.linspace(0.0, 1.0, g.shape[1])
    assert abs(u_lo - us[inside][0]) < 1e-9
    assert abs(u_hi - us[inside][-1]) < 1e-9


def test_control_points_stay_inside_contract_range_after_augmentation():
    img, g = _fixture()
    img, g = geometric_rotate(img, g, 5.0)
    img, g, u_lo, u_hi = geometric_margin(img, g, 0.06, 0.04, 0.05, 0.05)
    dst, src = control_points(g, 6, 3, img.shape, u_lo, u_hi)
    assert dst.min() >= 0.0 and dst.max() <= 1.0
    assert src.min() >= -0.5 and src.max() <= 1.5


def test_photometric_never_receives_coordinates():
    """설계 §5. 시그니처로 사고를 막는다."""
    params = inspect.signature(photometric).parameters
    assert "g" not in params and "G" not in params


def test_photometric_leaves_control_points_bit_identical():
    """설계 §10-5."""
    img, g = _fixture()
    before = control_points(g, 6, 3, img.shape)
    photometric(img, np.random.default_rng(0), sigma=0.6, noise=3.0, jpeg=80)
    after = control_points(g, 6, 3, img.shape)
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])


def test_photometric_is_identity_when_all_off():
    img, _ = _fixture()
    out = photometric(img, np.random.default_rng(0))
    assert np.array_equal(out, img)


def test_photometric_changes_pixels_when_on():
    img, _ = _fixture()
    out = photometric(img, np.random.default_rng(0), sigma=0.7, noise=4.0, jpeg=70)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_rotation_with_G_beats_stale_G_on_rectified_pixels():
    """설계 §10-4. 증강 검증에 디코딩을 쓰면 안 된다 — 20도까지 틀려도 읽힌다.

    기준: G 를 같이 변환한 쪽의 평균 차이가 미변환 쪽의 절반 이하.
    """
    from scripts.label_recipes import rectify
    from wemeet.data.render import render_clean
    from wemeet.data.warp import apply_warp

    clean = render_clean("WEMEET0001", module_px=5.5, height_px=220)
    zx, _ = grad_crease(clean.shape[1], 220, slope=1.4, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat)
    g = build_G(s_hat)
    out_shape = (220, clean.shape[1])

    dst0, src0 = control_points(g, 6, 3, obs.shape)
    base = rectify(obs, dst0, src0, out_shape).astype(np.float64)

    rotated, g_rot = geometric_rotate(obs, g, 2.0)
    dst_ok, src_ok = control_points(g_rot, 6, 3, rotated.shape)
    with_g = rectify(rotated, dst_ok, src_ok, out_shape).astype(np.float64)
    stale = rectify(rotated, dst0, src0, out_shape).astype(np.float64)

    err_ok = np.abs(with_g - base).mean()
    err_stale = np.abs(stale - base).mean()
    assert err_ok < err_stale / 2.0


def test_photometric_blurs_before_adding_noise():
    """순서가 뒤바뀌면(노이즈 -> 블러) 블러가 노이즈를 뭉개 표준편차가 크게
    줄어든다 (실측: 올바른 순서 std~9.98, 뒤바뀐 순서 std~1.59). 그래서 표준
    편차로 순서를 가른다. 균일 이미지로는 블러 자체를 못 보이므로 여기서만
    지역 픽스처를 쓴다.
    """
    img = np.full((64, 64), 128, dtype=np.uint8)
    out = photometric(img, np.random.default_rng(0), sigma=2.0, noise=10.0, jpeg=None)
    assert out.astype(np.float64).std() > 5.0
