"""구운 평가 세트를 검사한다. 설계 검증 #2 #3 #5 #7.

data/eval 이 없으면 건너뛴다 -- CI 에서는 생성물이 없다.
"""
import json
import os
import random

import cv2
import numpy as np
import pytest

from scripts.label_recipes import decode, rectify

SETS = ("low", "mid", "high")
pytestmark = pytest.mark.skipif(not os.path.isdir("data/eval/low"),
                                reason="평가 세트가 아직 안 구워졌다")


def _rows(name):
    with open(f"data/eval/{name}/manifest.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("name", SETS)
def test_every_row_has_ground_truth(name):
    for row in _rows(name):
        assert row["text"] and row["band"] == "target"


def test_three_sets_have_the_same_count():
    counts = {n: len(_rows(n)) for n in SETS}
    assert len(set(counts.values())) == 1, counts


@pytest.mark.parametrize("name", SETS)
def test_flat_aspect_is_in_range(name):
    for row in _rows(name):
        assert 2.0 - 0.02 <= row["aspect"] <= 2.3 + 0.02


@pytest.mark.parametrize("name", SETS)
def test_round_trip_decodes(name):
    """구운 PNG 를 옆 NPZ 제어점으로 펴면 전부 읽힌다.

    이게 진짜 검증이다 -- 굽기 경로와 NPZ 좌표계가 맞는지를 왕복으로 확인한다.
    설계 §8.5 가 '가장 비싼 버그' 로 지목한 방향 뒤집힘이 여기서 잡힌다.

    500장 중 50장만 본다 -- 전수는 게이트에 너무 느리다(생성 시점과 리뷰에서 별도로
    돌렸다). 앞 50장이 아니라 고정 시드로 뽑은 50장인 이유는, manifest 가 굽기 순서라
    앞부분이 특정 샤드에 치우칠 수 있기 때문이다. 시드를 박아 게이트는 재현된다.
    """
    rows = _rows(name)
    picked = random.Random(20260916).sample(range(len(rows)), min(50, len(rows)))
    rows = [rows[i] for i in sorted(picked)]
    ok = 0
    for row in rows:
        obs = cv2.imread(f"data/eval/{name}/{row['file']}", cv2.IMREAD_GRAYSCALE)
        z = np.load(f"data/eval/{name}/{row['npz']}")
        fixed = rectify(obs, z["dst_norm"], z["src_norm"],
                        (row["h_flat"], row["w_flat"]))
        if decode(fixed) == row["text"]:
            ok += 1
    assert ok == len(rows), f"{name}: {ok}/{len(rows)}"
