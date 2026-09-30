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

판정 디코더는 **SW `decode()` 의 zxing-cpp 경로 그대로**(`_decode_with_zxingcpp`)다.
v1 의 `label_recipes.decode` 는 `read_barcode`(한 개)를 쓰는데, 이 장면에서는
`read_barcodes` 와 결과가 달랐다 — 목표 80장 중 13장을 `read_barcodes` 만 읽었고,
1장은 `read_barcode` 가 틀린 번호를 냈다. 판정이 조건 A 와 같은 경로여야 구성이 성립한다.
pyzbar 는 판정에 쓰지 않는다 (PC 마다 깔렸는지가 달라서). pyzbar 가 있는 PC 에서는
조건 A 가 first_ok 보다 조금 더 읽을 수 있다.

학습(seed 42)·평가(seed 7)와 겹치지 않게 seed 2026 으로 뽑는다. 장면은 컨베이어 벨트 위
택배 상자·비닐 봉투의 송장 라벨이다 (`scripts/benchmark_scene.py`). 크롭은 리사이즈·
회전 없이 원본 픽셀 그대로 붙는다.

**현장 비율이 아니라 팀이 정한 구성이다.** 이 세트의 수치는 이 구성의 수치다.

    uv run python -m scripts.make_benchmark --out downloads/benchmark-v1

**v2** (`--version v2`, seed 2027) 는 같은 규칙에 두 가지를 더한다.

- **막대 비율을 반반 섞는다.** 1단계는 막대 영역을 5:3·6:4 로, 2단계 합성은 2.0~2.3 으로
  가정했고 어느 쪽도 실물로 확인되지 않았다. 50장씩 넣고 `aspect_family` 로 표시해,
  실물이 확정되면 맞는 절반을 쓴다. 레시피의 `aspect` 만 덮어쓰고 `wemeet/data` 는 그대로다
- **포장을 ±15° 돌린다.** v1 은 기울기가 없어 1단계의 기울기 보정이 평가되지 않았다.
  크롭은 돌리면서 한 번 보간된다. 크롭 네 꼭짓점을 `crop_quad` 로 남긴다 (검출 채점용)

    uv run python -m scripts.make_benchmark --version v2 --out downloads/benchmark-v2
"""

import argparse
import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.benchmark_scene import compose, compose_tilted
from scripts.label_recipes import code_commit, rectify
from wemeet.data.synthesis import build, draw_recipe, recipe_to_dict
from wemeet.sw.decoding import _decode_with_zxingcpp

PER_BAND = {  # bucket -> target / hard / first_ok
    "low": {"target": 27, "hard": 2, "first_ok": 5},
    "mid": {"target": 27, "hard": 2, "first_ok": 5},
    "high": {"target": 26, "hard": 1, "first_ok": 5},
}
WIDE, SPEC = "2.0-2.3", "5:3|6:4"  # 2단계 합성 가정 / 1단계 검출 가정
# v2: 버킷마다 두 비율을 반씩. 합은 PER_BAND 와 같다 (15/80/5)
PER_BAND_V2 = {
    "low": {
        WIDE: {"target": 14, "hard": 1, "first_ok": 2},
        SPEC: {"target": 13, "hard": 1, "first_ok": 3},
    },
    "mid": {
        WIDE: {"target": 14, "hard": 1, "first_ok": 2},
        SPEC: {"target": 13, "hard": 1, "first_ok": 3},
    },
    "high": {
        WIDE: {"target": 13, "hard": 1, "first_ok": 2},
        SPEC: {"target": 13, "hard": 0, "first_ok": 3},
    },
}
VERSIONS = {
    "v1": {"seed": 2026, "tilt": 0.0, "quota": {b: {WIDE: q} for b, q in PER_BAND.items()}},
    "v2": {"seed": 2027, "tilt": 15.0, "quota": PER_BAND_V2},
}
TAU = 0.0741  # v1 과 같은 값 (data/hf/README.md)
# 글꼴(Fonts)과 라벨 문구(WORDS)만 빌려 쓴다
SCENE_GEN = Path("docs/experiments/2026-09-22-stage1-detection/code/make_barcode_dataset.py")


def _scene_gen():
    spec = importlib.util.spec_from_file_location("make_barcode_dataset", SCENE_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode(gray: np.ndarray) -> str | None:
    return _decode_with_zxingcpp(Image.fromarray(gray))


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
    ap.add_argument("--version", choices=sorted(VERSIONS), default="v1")
    ap.add_argument("--seed", type=int, default=None, help="기본값은 버전별 (v1 2026, v2 2027)")
    ap.add_argument("--n", type=int, default=20000, help="버킷당 렌더 상한")
    args = ap.parse_args()

    cfg = VERSIONS[args.version]
    seed = cfg["seed"] if args.seed is None else args.seed
    tilt = cfg["tilt"]
    families = list(next(iter(cfg["quota"].values())))
    gen = _scene_gen()
    fonts = gen.Fonts()
    for sub in ("images", "crops"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    rows, stats = [], {}
    for b_i, (bucket, quota) in enumerate(cfg["quota"].items()):
        recipe_rng = np.random.default_rng([seed, b_i])
        kept, seen, rendered = Counter(), Counter(), 0
        while any(kept[f, b] < n for f, q in quota.items() for b, n in q.items()):
            if rendered >= args.n:
                raise RuntimeError(f"{bucket}: 렌더 상한 도달, 부족 {dict(kept)} — --n 을 올리세요")
            # 번호가 버킷끼리 겹치지 않게 인덱스를 엇갈려 준다 (텍스트는 index % 10000)
            recipe = draw_recipe(recipe_rng, bucket, rendered * len(PER_BAND) + b_i)
            scene_rng = np.random.default_rng([seed, b_i, rendered])
            family = families[rendered % len(families)]
            if tilt:
                aux = np.random.default_rng([seed, b_i, rendered, 1])
                if family == SPEC:
                    recipe = replace(recipe, aspect=5 / 3 if aux.random() < 0.5 else 3 / 2)
                angle = float(aux.uniform(-tilt, tilt))
            rendered += 1
            if all(kept[family, b] >= n for b, n in quota[family].items()):
                continue
            sample = build(recipe)
            if tilt:
                scene, bg_kind, quad = compose_tilted(
                    sample.obs, scene_rng, fonts, gen.WORDS, angle
                )
                q = np.array(quad)
                lo, hi = q.min(0), q.max(0)
                box = [int(lo[0]), int(lo[1]), int(hi[0] - lo[0]), int(hi[1] - lo[1])]
            else:
                scene, bg_kind, box = compose(sample.obs, scene_rng, fonts, gen.WORDS)
            band = _band(scene, sample, recipe.text)
            seen[band] += 1
            if kept[family, band] >= quota[family].get(band, 0):
                continue
            kept[family, band] += 1

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
            if tilt:
                row.update(aspect_family=family, tilt_deg=round(angle, 2), crop_quad=quad)
            rows.append(row)
        stats[bucket] = {"rendered": rendered, "seen": dict(seen)}
        print(bucket, stats[bucket], flush=True)

    header = {
        "dataset": f"benchmark-{args.version}",
        "seed": seed,
        "render_cap": args.n,
        "tau": TAU,
        "per_band": cfg["quota"] if tilt else PER_BAND,
        "tilt_deg": tilt,
        "band_rule": "scene-level, sw zxing-cpp path (read_barcodes)",
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
