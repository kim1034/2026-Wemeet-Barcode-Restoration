import pytest

from scripts.generate_v1 import SHARDS, merge_stats, preset_mix, tau_from_sats


def _part(shard, rendered, seen, kept, shortfall, hit):
    return {
        "bucket": "L",
        "shard": shard,
        "shards": 3,
        "rendered": rendered,
        "hit_cap": hit,
        "seen": seen,
        "kept": kept,
        "shortfall": shortfall,
        "target_sats": [],
        "target_presets": {},
    }


def test_merge_sums_counters():
    merged = merge_stats(
        [
            _part(0, 10, {"target": 2, "hard": 1}, {"target": 2}, {}, False),
            _part(1, 20, {"target": 3, "burned": 4}, {"target": 3}, {"hard": 1}, True),
        ]
    )
    assert merged["rendered"] == 30
    assert merged["seen"] == {"target": 5, "hard": 1, "burned": 4}
    assert merged["kept"] == {"target": 5}
    assert merged["shortfall"] == {"hard": 1}


def test_merge_keeps_hit_cap_per_shard():
    """부족분이 분포 탓인지 쪼갠 탓인지 갈리려면 샤드별로 남아야 한다 (설계 §5)."""
    merged = merge_stats(
        [
            _part(0, 10, {}, {}, {}, False),
            _part(1, 20, {}, {}, {}, True),
        ]
    )
    assert merged["hit_cap"] == [False, True]


def test_merge_pools_target_sats_instead_of_averaging_percentiles():
    """샤드별 백분위수를 평균내면 틀린다. 원자료를 모아 한 번에 낸다 (설계 §5)."""
    a = _part(0, 1, {}, {}, {}, False) | {"target_sats": [0.10, 0.20]}
    b = _part(1, 1, {}, {}, {}, False) | {"target_sats": [0.90]}
    merged = merge_stats([a, b])
    assert sorted(merged["target_sats"]) == [0.10, 0.20, 0.90]
    assert "tau_suggestion" not in merged


def test_merge_sums_target_presets():
    a = _part(0, 1, {}, {}, {}, False) | {"target_presets": {"crease": 2}}
    b = _part(1, 1, {}, {}, {}, False) | {"target_presets": {"crease": 1, "sine": 1}}
    assert merge_stats([a, b])["target_presets"] == {"crease": 3, "sine": 1}


def test_tau_is_95th_percentile():
    assert tau_from_sats([0.10, 0.90]) == pytest.approx(0.86, abs=0.01)
    assert tau_from_sats([]) is None


def test_preset_mix_normalizes():
    assert preset_mix({"crease": 3, "sine": 1}) == {"crease": 0.75, "sine": 0.25}
    assert preset_mix({}) == {}


def test_shard_allocation_is_fifteen():
    assert sum(SHARDS.values()) == 15
