import pathlib
import subprocess
from collections import Counter

import numpy as np

from scripts.label_recipes import (
    BANDS,
    code_commit,
    decode,
    fill_bucket,
    label_sample,
    rectify,
    tps_flow,
)
from wemeet.data.synthesis import H_OBS, Sample, build, draw_recipe


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


def test_wrong_tps_kernel_fails_to_decode_a_known_target_sample():
    """설계: rectify 가 band 판정의 핵심이다 -- 커널 공식이 틀리면 데이터셋
    전체가 조용히 잘못 라벨링된다(잘못 편 이미지가 그냥 "hard" 로 떨어질 뿐
    아무 데도 에러가 안 난다). RBF 보간은 제어점에서는 어떤 커널로도 정확한
    것이 정의라서(test_tps_flow_is_identity_for_identical_control_points 가
    그걸 증명은 못 하고 오히려 그 사실 때문에 커널 자체를 못 잡는다 --
    실측: mutant 커널로도 4-corner 항등 오차 ~4e-15, 내부점 9개를 써도 동일),
    커널 오류는 제어점 "사이" 에서만 드러난다.

    그래서 독립된 TPS 기준 없이, 1차 디코딩은 실패하지만 보정 후에는
    성공하는 실제 target 표본이 보정 후에도 계속 디코딩되는지로 커널을
    간접 검증한다. seed=5 로 뽑은 네 번째 레시피가 그런 표본임을 실측으로
    확인했다 (decode(obs) 는 None, decode(rectify(...)) 는 성공).
    """
    rng = np.random.default_rng(5)
    for i in range(4):
        recipe = draw_recipe(rng, "L", i)
    sample = build(recipe)
    assert decode(sample.obs) is None
    fixed = rectify(sample.obs, sample.dst_norm, sample.src_norm, (H_OBS, sample.w_flat))
    assert decode(fixed) is not None


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


def test_fill_bucket_shortfall_is_never_papered_over_by_relabeling():
    """설계 §7: 부족분은 메우지 않는다. 개수만 보면(바로 위 브리프 원본 테스트)
    재분류(backfill) 도 통과한다 -- 부족한 밴드의 자리를 다른 밴드의 표본으로
    채우고 그 표본의 기록만 부족한 밴드 이름으로 바꿔치기해도 쿼터 개수는
    똑같이 맞아떨어지기 때문이다(실측: 76개 시드/캡 조합 전부 통과).

    그래서 여기서는 개수가 아니라 구성을 본다: kept 에 담긴 각 표본이 실제로
    자기 밴드로 라벨링됐는지 다시 계산해 맞춰보고, 진짜 부족분은 숫자를
    맞추지 않고 shortfall 에 정직하게 남는지를 함께 확인한다.

    seed=1 은 실측으로 30장 안에 쿼터를 다 못 채우는 진짜 부족분을 만든다.
    **어느 밴드가 모자라는지는 하드코딩하지 않는다** -- 그것은 제어점 개수에
    딸린 값이고(N_X 를 6 에서 16 으로 올리자 seed=1 에서 target 이 실제로
    나오기 시작했다), 이 테스트가 지키려는 것은 "부족분을 정직하게 남기는가"
    이지 특정 시드의 밴드 구성이 아니다.
    """
    quota = {"target": 1, "hard": 1, "first_ok": 1}
    kept, stats = fill_bucket("L", quota, render_cap=30, seed=1, tau=0.04)

    for recipe, band, sat, m_min in kept:
        sample = build(recipe)
        assert label_sample(sample, (H_OBS, sample.w_flat), 0.04) == band

    assert len(kept) < sum(quota.values())
    assert stats["shortfall"], "진짜 부족분이 있어야 이 테스트가 의미를 갖는다"
    # 부족분이 숫자를 맞추려고 재분류되지 않았는가: 밴드마다 채운 것 + 부족분 = 쿼터
    for band, missing in stats["shortfall"].items():
        assert stats["kept"].get(band, 0) + missing == quota[band]
    assert len(kept) + sum(stats["shortfall"].values()) == sum(quota.values())


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


def test_recipe_header_records_the_code_commit_that_rendered_it():
    """설계 §1: "레시피 헤더에 코드 커밋 해시를 기록한다 -- 코드가 바뀌면 같은
    레시피도 다른 이미지가 된다." 레시피는 이미지가 아니라 재생성 지시서라서,
    같은 JSONL 이 코드 버전에 따라 다른 데이터셋이 된다. 그 사실이 헤더에
    없으면 나중에 성능 차이가 데이터 탓인지 코드 탓인지 가릴 수가 없다.

    키 존재만 보면 상수 문자열을 박아 놔도 통과하므로, 실제 HEAD 와 맞는지를
    독립적으로(subprocess 로 다시 물어봐서) 대조한다.
    """
    head = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True,
                          text=True, check=True).stdout.strip()
    _, stats = fill_bucket("L", {"first_ok": 1}, render_cap=1, seed=3, tau=0.08)
    assert stats["code"]["commit"] == head
    assert isinstance(stats["code"]["dirty"], bool)


def test_code_commit_reports_dirty_instead_of_hiding_it():
    """dirty 인 트리에서는 커밋 해시만으로 이미지를 재현할 수 없다. 그래도
    라벨링을 막지 않고 사실만 남기는 것이 설계 의도다 -- 그러니 dirty 가
    항상 False 로 굳어 있으면(= 사실상 기록이 없으면) 안 된다.

    이 테스트 파일 자체가 워킹 트리에 있는 동안에는 판정할 수 없으므로,
    임시 파일을 만들어 dirty 가 실제로 True 로 뒤집히는지를 본다.
    """
    scratch = pathlib.Path("__dirty_probe__.tmp")
    scratch.write_text("probe", encoding="utf-8")
    try:
        assert code_commit()["dirty"] is True
    finally:
        scratch.unlink()
