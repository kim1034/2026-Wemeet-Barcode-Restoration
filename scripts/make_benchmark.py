"""E2E 벤치마크 v1 — 전체 사진 100장을 만든다. 한 번 만들고 고정한다.

구성 (low/mid/high 에 고르게):

    1차 성공(first_ok) 15 · 목표(target) 80 · 불가(hard) 5

**구간은 크롭이 아니라 장면(사진 전체)에서 판정한다.** 크롭 기준 라벨
(`label_recipes.label_sample`)을 그대로 쓰면 크롭 기준 목표 구간 80장 중 28장이
장면에서는 보정 없이 읽혔다 — zxing 의 이진화가 바코드 주변 배경에 따라 달라진다
(주변 10px → 12장, 120px → 36장, 흰 여백 덧대기로는 2~5장). 그래서:

    first_ok  사진 전체를 디코딩하면 정답
    target    사진 전체로는 안 되고, 크롭을 정답 제어점으로 펴면 정답
    hard      둘 다 안 됨 (포화 소실 `burned` 은 뽑지 않는다)

판정 디코더는 v1 과 같은 zxing-cpp 단독(`label_recipes.decode`)이다. SW `decode()`
는 pyzbar 를 먼저 쓰므로 조건 A 가 first_ok 보다 조금 더 읽을 수 있다.

학습(seed 42)·평가(seed 7)와 겹치지 않게 seed 2026 으로 뽑는다. 장면은 컨베이어 벨트 위
택배 상자·비닐 봉투의 송장 라벨이다 (`scripts/benchmark_scene.py`). 크롭은 리사이즈·
회전 없이 원본 픽셀 그대로 붙는다.

**현장 비율이 아니라 팀이 정한 구성이다.** 이 세트의 수치는 이 구성의 수치다.

    uv run python -m scripts.make_benchmark --out downloads/benchmark-v1
"""

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.benchmark_scene import compose
from scripts.label_recipes import code_commit, decode, rectify
from wemeet.data.synthesis import build, draw_recipe, recipe_to_dict

PER_BAND = {  # bucket -> target / hard / first_ok
    "low": {"target": 27, "hard": 2, "first_ok": 5},
    "mid": {"target": 27, "hard": 2, "first_ok": 5},
    "high": {"target": 26, "hard": 1, "first_ok": 5},
}
TAU = 0.0741  # v1 과 같은 값 (data/hf/README.md)
# 글꼴(Fonts)과 라벨 문구(WORDS)만 빌려 쓴다
SCENE_GEN = Path("docs/experiments/2026-09-22-stage1-detection/code/make_barcode_dataset.py")


def _scene_gen():
    spec = importlib.util.spec_from_file_location("make_barcode_dataset", SCENE_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _band(scene: np.ndarray, sample, text: str) -> str:
    if decode(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)) == text:
        return "first_ok"
    fixed = rectify(sample.obs, sample.dst_norm, sample.src_norm, (sample.h_flat, sample.w_flat))
    if decode(fixed) == text:
        return "target"
    return "burned" if sample.sat_ratio > TAU else "hard"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=20000, help="버킷당 렌더 상한")
    args = ap.parse_args()

    gen = _scene_gen()
    fonts = gen.Fonts()
    for sub in ("images", "crops"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    rows, stats = [], {}
    for b_i, (bucket, quota) in enumerate(PER_BAND.items()):
        recipe_rng = np.random.default_rng([args.seed, b_i])
        kept, seen, rendered = Counter(), Counter(), 0
        while any(kept[b] < n for b, n in quota.items()):
            if rendered >= args.n:
                raise RuntimeError(f"{bucket}: 렌더 상한 도달, 부족 {dict(kept)} — --n 을 올리세요")
            # 번호가 버킷끼리 겹치지 않게 인덱스를 엇갈려 준다 (텍스트는 index % 10000)
            recipe = draw_recipe(recipe_rng, bucket, rendered * len(PER_BAND) + b_i)
            scene_rng = np.random.default_rng([args.seed, b_i, rendered])
            rendered += 1
            sample = build(recipe)
            scene, bg_kind, box = compose(sample.obs, scene_rng, fonts, gen.WORDS)
            band = _band(scene, sample, recipe.text)
            seen[band] += 1
            if kept[band] >= quota.get(band, 0):
                continue
            kept[band] += 1

            name = f"{len(rows):03d}"
            cv2.imwrite(str(args.out / "images" / f"{name}.png"), scene)
            cv2.imwrite(str(args.out / "crops" / f"{name}.png"), sample.obs)
            np.savez(
                args.out / "crops" / f"{name}.npz",
                dst_norm=sample.dst_norm,
                src_norm=sample.src_norm,
            )
            row = recipe_to_dict(recipe)
            row.update(
                id=name,
                file=f"images/{name}.png",
                crop=f"crops/{name}.png",
                npz=f"crops/{name}.npz",
                band=band,
                sat_ratio=sample.sat_ratio,
                m_min=sample.m_min,
                w_flat=sample.w_flat,
                h_flat=sample.h_flat,
                background=bg_kind,
                crop_box_xywh=box,
            )
            rows.append(row)
        stats[bucket] = {"rendered": rendered, "seen": dict(seen)}
        print(bucket, stats[bucket], flush=True)

    header = {
        "seed": args.seed,
        "render_cap": args.n,
        "tau": TAU,
        "per_band": PER_BAND,
        "band_rule": "scene-level, zxing-cpp only",
        "scene": "conveyor belt / box|vinyl / shipping label",
        "code": code_commit(),
        "buckets": stats,
    }
    with open(args.out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_stats": header}, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(header, ensure_ascii=False, indent=2))
    print(f"{len(rows)} samples -> {args.out}")


if __name__ == "__main__":
    main()
