"""샤드 병렬 생성 드라이버. 오프라인 도구다.

버킷 하나를 샤드 N 개로 쪼개 동시에 돌리고, 결과를 하나의 JSONL 로 합친다.
샤드 수는 고정 상수다 -- 바꾸면 같은 시드라도 다른 데이터가 나오므로
레시피 헤더에 기록한다 (설계 §5).
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from scripts.label_recipes import bake, code_commit, fill_bucket
from wemeet.data.synthesis import ALL_BUCKETS, recipe_to_dict

# 설계 §4 의 프로토타입 시간(L 1.09h / M 3.35h / H 3.89h)에 비례해 나눴다.
# 샤드당 0.54~0.56h 로 고르다. 16번째 코어는 OS 몫이다.
SHARDS = {"L": 2, "M": 6, "H": 7}
EVAL_SHARDS = 5


def merge_stats(parts: list[dict]) -> dict:
    """샤드별 stats 를 합친다. tau 는 여기서 내지 않는다 -- 원자료를 모아 준다."""
    seen: dict[str, int] = {}
    kept: dict[str, int] = {}
    short: dict[str, int] = {}
    presets: dict[str, int] = {}
    sats: list[float] = []
    for p in parts:
        for k, v in p["seen"].items():
            seen[k] = seen.get(k, 0) + v
        for k, v in p["kept"].items():
            kept[k] = kept.get(k, 0) + v
        for k, v in p["shortfall"].items():
            short[k] = short.get(k, 0) + v
        for k, v in p["target_presets"].items():
            presets[k] = presets.get(k, 0) + v
        sats.extend(p["target_sats"])
    return {
        "bucket": parts[0]["bucket"],
        "shards": len(parts),
        "rendered": sum(p["rendered"] for p in parts),
        # 샤드별로 남긴다 -- 부족분이 분포 탓인지 쪼갠 탓인지 갈려야 한다 (설계 §5).
        "hit_cap": [p["hit_cap"] for p in parts],
        "seen": seen,
        "kept": kept,
        "shortfall": short,
        "target_presets": presets,
        "target_sats": sats,
    }


def tau_from_sats(sats: list[float]) -> float | None:
    """목표 구간 포화율의 95 백분위수. 목표는 복구가 증명된 샘플이다 (설계 §7)."""
    return float(np.percentile(sats, 95)) if sats else None


def preset_mix(counts: dict) -> dict:
    """실현된 preset 비율. 30/30/30/10 에서 벗어난다 -- 보고만 한다 (설계 §7)."""
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def _worker(args):
    """샤드 하나를 돌리고 레시피를 '자기 파일에' 쓴다.

    레시피를 부모로 돌려보내지 않는 이유는 메모리다 -- 상한까지 가면 58.5만 행이고
    행당 약 1KB 라 부모가 600MB 를 들고 있게 된다. 덤으로, 중간에 죽어도 끝난
    샤드의 결과가 파일로 남는다.
    """
    bucket, per_band, cap, seed, tau, shard, shards, out = args
    kept, stats = fill_bucket(bucket, per_band, cap, seed, tau, shard, shards)
    path = f"{out}.shard{shard}"
    with open(path, "w", encoding="utf-8") as fh:
        for recipe, band, sat, m_min in stats.pop("labelled"):
            row = recipe_to_dict(recipe)
            row.update(band=band, sat_ratio=sat, m_min=m_min)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return kept, stats, path


def _stats_path(out: str) -> str:
    """통계는 항상 data/stats/ 에 둔다 -- git 에 들어가는 유일한 생성물이다."""
    name = os.path.basename(out)
    if name.endswith(".jsonl"):
        name = name[:-len(".jsonl")]
    return os.path.join("data", "stats", name + ".stats.json")


def generate(bucket, per_band, cap, seed, tau, shards, out, bake_dir=None):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    jobs = [(bucket, per_band, cap, seed, tau, i, shards, out)
            for i in range(shards)]
    with ProcessPoolExecutor(max_workers=shards) as ex:
        results = list(ex.map(_worker, jobs))

    kept = [row for k, _, _ in results for row in k]
    stats = merge_stats([s for _, s, _ in results])
    stats["tau_suggestion"] = tau_from_sats(stats.pop("target_sats"))
    stats["preset_mix_realized"] = preset_mix(stats["target_presets"])
    stats["code"] = code_commit()

    # 샤드 파일을 그대로 이어붙인다. JSON 을 다시 파싱하지 않는다.
    with open(out, "w", encoding="utf-8") as dst:
        dst.write(json.dumps({"_stats": stats}, ensure_ascii=False) + "\n")
        for _, _, path in results:
            with open(path, encoding="utf-8") as src:
                shutil.copyfileobj(src, dst)
            os.remove(path)

    if bake_dir:
        stats["baked"] = bake(kept, bake_dir)

    stats_path = _stats_path(out)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, choices=sorted(ALL_BUCKETS))
    ap.add_argument("--n", type=int, required=True, help="전체 렌더 상한")
    ap.add_argument("--per-band", default="5000/3500/1500", help="target/hard/first_ok")
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bake", default=None)
    args = ap.parse_args()

    t, h, f = (int(v) for v in args.per_band.split("/"))
    shards = args.shards or SHARDS.get(args.bucket, EVAL_SHARDS)
    generate(args.bucket, {"target": t, "hard": h, "first_ok": f},
             args.n, args.seed, args.tau, shards, args.out, args.bake)


if __name__ == "__main__":
    main()
