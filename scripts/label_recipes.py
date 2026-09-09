"""구간 라벨링과 비중 채우기. 오프라인 도구다.

wemeet/data/ 는 이 파일을 import 하지 않는다 — scripts/ 는 의존 규칙 밖이다 (설계 §1).
여기 있는 rectify 는 라벨링 도구다. SW파트가 wemeet/sw/rectify.py 를 내면
아래 tps_flow/rectify 자리를 그 구현으로 갈아끼운다.
"""

import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np
import zxingcpp

from wemeet.data.synthesis import BUCKETS, H_OBS, Sample, build, draw_recipe, recipe_to_dict

BANDS = ("target", "hard", "first_ok", "burned")


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


def label_sample(sample: Sample, out_shape, tau: float) -> str:
    """구간 판정. 1차 성공이면 보정·재디코딩을 건너뛴다 (단락)."""
    if decode(sample.obs) is not None:
        return "first_ok"
    fixed = rectify(sample.obs, sample.dst_norm, sample.src_norm, out_shape)
    if decode(fixed) is not None:
        return "target"
    return "burned" if sample.sat_ratio > tau else "hard"


def fill_bucket(bucket: str, per_band: dict, render_cap: int, seed: int,
                tau: float):
    """버킷 하나를 비중대로 채운다. 모자라면 메우지 않고 부족분으로 남긴다.

    레시피는 기록의 원본이다 — burned 나 쿼터를 넘긴 first_ok 도 버리지 않고
    stats["labelled"] 에 전부 남긴다. "버림" 은 학습 로더의 일이지 여기서 할
    일이 아니다. kept (쿼터만큼만) 는 --bake 용 큐레이션 목록으로 그대로 둔다.
    """
    rng = np.random.default_rng(seed)
    kept, kept_count, seen = [], Counter(), Counter()
    labelled = []
    rendered = 0
    sat_of_target = []

    while rendered < render_cap and any(
            kept_count[b] < per_band.get(b, 0) for b in per_band):
        recipe = draw_recipe(rng, bucket, rendered)
        sample = build(recipe)
        rendered += 1
        band = label_sample(sample, (H_OBS, sample.w_flat), tau)
        seen[band] += 1
        labelled.append((recipe, band, sample.sat_ratio, sample.m_min))
        if band == "target":
            sat_of_target.append(sample.sat_ratio)
        if band == "burned":
            continue
        if kept_count[band] < per_band.get(band, 0):
            kept_count[band] += 1
            kept.append((recipe, band, sample.sat_ratio, sample.m_min))

    stats = {
        "bucket": bucket,
        "rendered": rendered,
        "hit_cap": rendered >= render_cap,
        "seen": dict(seen),
        "kept": dict(kept_count),
        "shortfall": {b: per_band[b] - kept_count[b] for b in per_band
                      if kept_count[b] < per_band[b]},
        "tau_suggestion": (float(np.percentile(sat_of_target, 95))
                           if sat_of_target else None),
        "labelled": labelled,
    }
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, choices=sorted(BUCKETS))
    ap.add_argument("--n", type=int, required=True, help="렌더 상한")
    ap.add_argument("--per-band", default="5000/3500/1500",
                    help="target/hard/first_ok")
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bake", default=None,
                    help="평가 세트용. 이 디렉터리에 이미지를 굽는다")
    args = ap.parse_args()

    t, h, f = (int(v) for v in args.per_band.split("/"))
    kept, stats = fill_bucket(args.bucket, {"target": t, "hard": h,
                                            "first_ok": f},
                              args.n, args.seed, args.tau)
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
        # 평가 세트는 이미지로 굽는다 — 코드가 바뀌어도 같은 사진으로 비교해야
        # 성능 변화가 코드 탓인지 데이터 탓인지 갈린다 (설계 §1).
        os.makedirs(args.bake, exist_ok=True)
        for i, (recipe, band, _, _) in enumerate(kept):
            if band != "target":
                continue
            sample = build(recipe)
            cv2.imwrite(f"{args.bake}/{i:05d}.png", sample.obs)
            np.savez(f"{args.bake}/{i:05d}.npz",
                     dst_norm=sample.dst_norm, src_norm=sample.src_norm)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
