from collections import Counter

import numpy as np

from scripts.label_recipes import BANDS, fill_bucket, label_sample, rectify, tps_flow
from wemeet.data.synthesis import Sample, build, draw_recipe


def test_tps_flow_is_identity_for_identical_control_points():
    """설계 §10-11."""
    grid = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 60.0], [100.0, 60.0]])
    mx, my = tps_flow(grid, grid, (61, 101))
    gy, gx = np.mgrid[0:61, 0:101]
    assert np.abs(mx - gx).max() < 1e-3
    assert np.abs(my - gy).max() < 1e-3


def test_rectify_returns_requested_shape():
    s = build(draw_recipe(np.random.default_rng(4), "L", 0))
    out = rectify(s.obs, s.dst_norm, s.src_norm, (220, s.w_flat))
    assert out.shape == (220, s.w_flat)
    assert out.dtype == np.uint8


def test_bands_are_exactly_the_four_in_the_spec():
    assert BANDS == ("target", "hard", "first_ok", "burned")


def test_burned_requires_saturation_above_tau(monkeypatch):
    """포화가 tau 를 넘는 불가만 burned 다. 나머지 불가는 hard 로 남긴다."""
    import scripts.label_recipes as mod
    monkeypatch.setattr(mod, "decode", lambda img: None)
    obs = np.zeros((20, 40), dtype=np.uint8)
    dst = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    hot = Sample(obs, dst, dst, 0.5, sat_ratio=0.20, scale=1.0, w_flat=40)
    cool = Sample(obs, dst, dst, 0.5, sat_ratio=0.01, scale=1.0, w_flat=40)
    assert label_sample(hot, (20, 40), tau=0.04) == "burned"
    assert label_sample(cool, (20, 40), tau=0.04) == "hard"


def test_first_ok_short_circuits_the_second_decode(monkeypatch):
    """단락이 없으면 H 버킷이 20h -> 33h 가 된다."""
    import scripts.label_recipes as mod
    calls = []

    def counting_decode(img):
        calls.append(1)
        return "WEMEET0000"

    monkeypatch.setattr(mod, "decode", counting_decode)
    obs = np.zeros((20, 40), dtype=np.uint8)
    dst = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    s = Sample(obs, dst, dst, 0.5, 0.0, 1.0, 40)
    assert label_sample(s, (20, 40), tau=0.04) == "first_ok"
    assert len(calls) == 1


def test_fill_bucket_stops_at_the_render_cap():
    kept, stats = fill_bucket("L", {"target": 2, "hard": 2, "first_ok": 1},
                              render_cap=40, seed=1, tau=0.04)
    assert stats["rendered"] <= 40
    assert len(kept) <= 5
    for band, quota in {"target": 2, "hard": 2, "first_ok": 1}.items():
        assert stats["kept"][band] <= quota


def test_fill_bucket_never_backfills_a_shortfall():
    """H 에서 불가가 모자라도 1차 성공으로 메우지 않는다 (설계 §7)."""
    kept, stats = fill_bucket("L", {"target": 1, "hard": 1, "first_ok": 1},
                              render_cap=30, seed=2, tau=0.04)
    for band, quota in {"target": 1, "hard": 1, "first_ok": 1}.items():
        assert stats["kept"][band] <= quota


def test_fill_bucket_records_every_recipe_including_burned_and_surplus():
    """레시피는 기록의 원본이다: burned 와 쿼터를 넘긴 first_ok 도 버리지 않고
    stats["labelled"] 에 전부 남는다. "버림" 은 학습 로더의 일이다 (설계 총칙).

    이 seed 는 실측으로 burned 3개, first_ok 27개(쿼터 1개를 훌쩍 넘김)를
    만든다 -- kept 는 여전히 쿼터만큼(1개)만 담아 기존 계약을 안 건드리면서,
    labelled 는 렌더된 30개 전부를 담아야 한다.
    """
    kept, stats = fill_bucket("L", {"target": 1, "hard": 1, "first_ok": 1},
                              render_cap=30, seed=1, tau=0.04)
    labelled_seen = Counter(band for _, band, _, _ in stats["labelled"])
    assert len(stats["labelled"]) == stats["rendered"]
    assert dict(labelled_seen) == stats["seen"]
    assert labelled_seen["burned"] > 0
    assert labelled_seen["first_ok"] > stats["kept"].get("first_ok", 0)
