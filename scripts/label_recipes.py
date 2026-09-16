"""구간 라벨링과 비중 채우기. 오프라인 도구다.

wemeet/data/ 는 이 파일을 import 하지 않는다 — scripts/ 는 의존 규칙 밖이다 (설계 §1).
여기 있는 rectify 는 라벨링 도구다. SW파트가 wemeet/sw/rectify.py 를 내면
아래 tps_flow/rectify 자리를 그 구현으로 갈아끼운다.
"""

import argparse
import json
import os
import subprocess
from collections import Counter

import cv2
import numpy as np
import zxingcpp

from wemeet.data.synthesis import ALL_BUCKETS, Sample, build, draw_recipe, recipe_to_dict

BANDS = ("target", "hard", "first_ok", "burned")

# dirty 판정에서 볼 경로. 렌더 결과를 바꿀 수 있는 것만 넣는다.
CODE_PATHS = ("wemeet", "scripts", "tests", "pyproject.toml", "uv.lock")


def code_commit() -> dict:
    """레시피 헤더에 박을 코드 상태. 코드가 바뀌면 같은 레시피도 다른 이미지가 된다 (설계 §1).

    dirty 면 커밋 해시만으로는 이미지를 재현할 수 없다. 막지 않고 표시만 한다 —
    측정 중에 코드를 만지는 것은 흔하고, 나중에 "이 숫자를 믿을 수 있나"를
    판단할 사람에게 필요한 것은 금지가 아니라 사실이다.

    묻는 것은 "생성 코드가 커밋과 다른가" 이지 "작업 디렉터리가 깨끗한가" 가 아니다.
    그래서 CODE_PATHS 로 좁힌다 — 트리 전체를 보면 data/stats/ 가 git 에 들어가는
    유일한 생성물이라 앞선 버킷이 남긴 stats 파일이 다음 실행에서 untracked 로 잡히고,
    dirty 가 거의 항상 true 가 된다(v1 발행분 9개 중 8개가 그랬다). 항상 true 인
    필드는 진짜로 더러운 실행과 구분되지 않아서 정보량이 0이다.
    """
    def git(*args):
        return subprocess.run(("git", *args), capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--", *CODE_PATHS))}
    except (OSError, subprocess.CalledProcessError):
        # 리포 밖에서 돌리거나 git 이 없을 때. 라벨링 자체를 막을 이유는 없다.
        return {"commit": None, "dirty": None}


def tps_flow(src_px, dst_px, shape, reg: float = 0.0):
    """Thin Plate Spline. 제어점에서 픽셀별 "어디서 가져올지" 지도를 만든다."""
    n = len(dst_px)
    d = np.linalg.norm(dst_px[:, None, :] - dst_px[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(d > 0, d ** 2 * np.log(d ** 2), 0.0)
    k += reg * np.eye(n)
    p = np.hstack([np.ones((n, 1)), dst_px])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T

    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    dg = np.linalg.norm(grid[:, None, :] - dst_px[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ug = np.where(dg > 0, dg ** 2 * np.log(dg ** 2), 0.0)
    pg = np.hstack([np.ones((len(grid), 1)), grid])

    out = []
    for axis in (0, 1):
        b = np.concatenate([src_px[:, axis], np.zeros(3)])
        coef = np.linalg.solve(a, b)
        out.append((ug @ coef[:n] + pg @ coef[n:]).reshape(h, w).astype(np.float32))
    return out[0], out[1]


def rectify(obs, dst_norm, src_norm, out_shape):
    """정규화 제어점으로 편다. 픽셀 환산은 w-1, h-1 을 곱한다."""
    ho, wo = out_shape
    hi, wi = obs.shape
    dst_px = dst_norm * np.array([wo - 1, ho - 1])
    src_px = src_norm * np.array([wi - 1, hi - 1])
    mx, my = tps_flow(src_px, dst_px, out_shape)
    return cv2.remap(obs, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def decode(img):
    try:
        res = zxingcpp.read_barcode(img)
        return res.text if res and res.valid else None
    except Exception:
        return None


def label_sample(sample: Sample, out_shape, tau: float, text: str) -> str:
    """구간 판정. 1차 성공이면 보정·재디코딩을 건너뛴다 (단락).

    「읽혔는가」가 아니라 「맞게 읽혔는가」를 본다. Code128 은 mod-103 체크
    디지트라 훼손된 판독이 약 1/103 로 체크섬을 통과해 **유효하지만 틀린**
    문자열을 돌려준다(실측: 평가 1,500장 중 8장). 틀린 번호를 1차 성공으로
    보면 파이프라인이 기하 보정을 건너뛰고 그 번호를 그대로 내보낸다 --
    `0003` 이 생성 모델 접근을 폐기한 바로 그 이유다.
    """
    if decode(sample.obs) == text:
        return "first_ok"
    fixed = rectify(sample.obs, sample.dst_norm, sample.src_norm, out_shape)
    if decode(fixed) == text:
        return "target"
    return "burned" if sample.sat_ratio > tau else "hard"


def bake(kept, out_dir) -> int:
    """목표 구간만 이미지로 굽는다. 평가 세트 전용.

    코드가 바뀌어도 같은 사진으로 비교해야 성능 변화가 코드 탓인지 데이터 탓인지
    갈린다 (설계 §1). preset 과 열화 파라미터를 같이 남겨 사후에 쪼개 볼 수 있게 한다.
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(f"{out_dir}/manifest.jsonl", "w", encoding="utf-8") as fh:
        for recipe, band, sat, m_min in kept:
            if band != "target":
                continue
            sample = build(recipe)
            name = f"{n:05d}"
            cv2.imwrite(f"{out_dir}/{name}.png", sample.obs)
            np.savez(f"{out_dir}/{name}.npz",
                     dst_norm=sample.dst_norm, src_norm=sample.src_norm)
            row = recipe_to_dict(recipe)
            row.update(file=f"{name}.png", npz=f"{name}.npz", band=band,
                       sat_ratio=sat, m_min=m_min,
                       w_flat=sample.w_flat, h_flat=sample.h_flat)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_counts(total: int, shards: int) -> list[int]:
    """샤드별 몫. 나머지는 앞쪽 샤드에 하나씩 더 준다."""
    base, rem = divmod(total, shards)
    return [base + (1 if i < rem else 0) for i in range(shards)]


def fill_bucket(bucket: str, per_band: dict, render_cap: int, seed: int,
                tau: float, shard: int = 0, shards: int = 1):
    """버킷 하나를 비중대로 채운다. 모자라면 메우지 않고 부족분으로 남긴다.

    render_cap 과 per_band 는 '전체' 값이고 여기서 샤드 몫으로 쪼갠다. 샤드마다
    독립 난수 스트림(default_rng([seed, shard]))을 쓰고, 레시피 인덱스는 앞선
    샤드들의 몫만큼 밀어 겹치지 않게 한다.

    레시피는 기록의 원본이다 — burned 나 쿼터를 넘긴 first_ok 도 버리지 않고
    stats["labelled"] 에 전부 남긴다. "버림" 은 학습 로더의 일이다.
    """
    caps = split_counts(render_cap, shards)
    my_cap = caps[shard]
    index_offset = sum(caps[:shard])
    my_band = {b: split_counts(n, shards)[shard] for b, n in per_band.items()}

    rng = np.random.default_rng([seed, shard])
    kept, kept_count, seen = [], Counter(), Counter()
    labelled = []
    rendered = 0
    sat_of_target = []

    while rendered < my_cap and any(
            kept_count[b] < my_band.get(b, 0) for b in my_band):
        recipe = draw_recipe(rng, bucket, index_offset + rendered)
        sample = build(recipe)
        rendered += 1
        band = label_sample(sample, (sample.h_flat, sample.w_flat), tau, recipe.text)
        seen[band] += 1
        labelled.append((recipe, band, sample.sat_ratio, sample.m_min))
        if band == "target":
            sat_of_target.append(sample.sat_ratio)
        if band == "burned":
            continue
        if kept_count[band] < my_band.get(band, 0):
            kept_count[band] += 1
            kept.append((recipe, band, sample.sat_ratio, sample.m_min))

    stats = {
        "bucket": bucket,
        "shard": shard,
        "shards": shards,
        "code": code_commit(),
        "rendered": rendered,
        "hit_cap": rendered >= my_cap,
        "seen": dict(seen),
        "kept": dict(kept_count),
        "shortfall": {b: my_band[b] - kept_count[b] for b in my_band
                      if kept_count[b] < my_band[b]},
        "tau_suggestion": (float(np.percentile(sat_of_target, 95))
                           if sat_of_target else None),
        # 목표 구간의 원자료만 남긴다. 드라이버가 샤드를 합쳐 한 번에 백분위수를
        # 내야 맞다 -- 샤드별 백분위수를 평균내면 틀린다 (설계 §5).
        "target_sats": sat_of_target,
        "target_presets": dict(Counter(r.preset for r, b, _, _ in labelled
                                       if b == "target")),
        "labelled": labelled,
    }
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, choices=sorted(ALL_BUCKETS))
    ap.add_argument("--n", type=int, required=True, help="렌더 상한")
    ap.add_argument("--per-band", default="5000/3500/1500",
                    help="target/hard/first_ok")
    ap.add_argument("--tau", type=float, default=0.0741)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bake", default=None,
                    help="평가 세트용. 이 디렉터리에 이미지를 굽는다")
    ap.add_argument("--shard", default="0/1",
                    help="i/N. 샤드 i 만 돈다. --n 과 --per-band 는 전체 값이다")
    args = ap.parse_args()

    t, h, f = (int(v) for v in args.per_band.split("/"))
    shard, shards = (int(v) for v in args.shard.split("/"))
    kept, stats = fill_bucket(args.bucket, {"target": t, "hard": h,
                                            "first_ok": f},
                              args.n, args.seed, args.tau, shard, shards)
    # 레시피는 기록의 원본이다: burned 와 쿼터를 넘긴 first_ok 도 포함해
    # 렌더된 전부를 쓴다. "버림" 은 학습 로더의 일이다 (설계 총칙).
    labelled = stats.pop("labelled")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_stats": stats}, ensure_ascii=False) + "\n")
        for recipe, band, sat, m_min in labelled:
            row = recipe_to_dict(recipe)
            row.update(band=band, sat_ratio=sat, m_min=m_min)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.bake:
        bake(kept, args.bake)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
