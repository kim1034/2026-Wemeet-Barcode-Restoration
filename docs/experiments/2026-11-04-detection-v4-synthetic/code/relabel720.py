#!/usr/bin/env python3
"""v2 val(dataset_720, 10000번대)의 정답을 라벨 버그 없이 다시 계산한다.

옛 생성기(make_barcode_dataset.py)를 같은 시드로 돌려 배치까지 재현한다. 배치는 원래(버그) 점으로
해서 난수 흐름을 그대로 두고, 같은 변환 M 으로 convexHull 발자국의 박스를 따로 계산한다.
원래 방식의 박스가 기존 라벨과 1px 안으로 같은지로 재현을 확인한다 (11/07: 1000/1000).
배치가 끝나면 나머지(합성·조명)는 정답과 무관하므로 거기서 멈춘다.

    python relabel720.py --gen-dir ../../2026-09-22-stage1-detection/code \
        --data dataset_720 --out relabeled
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=Path, required=True, help="make_barcode_dataset.py 가 있는 폴더")
    ap.add_argument("--module", default="make_barcode_dataset")
    ap.add_argument("--data", type=Path, required=True, help="dataset_720 (meta_010000_020000.csv 포함)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.gen_dir))
    g = __import__(args.module)
    (args.out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    with open(args.data / "meta_010000_020000.csv", encoding="utf-8") as f:
        meta = {int(r["index"]): r for r in csv.DictReader(f)}

    cap = {}
    orig_damage, orig_place = g.apply_damage, g.place

    def apply_damage(*a, **k):
        r = orig_damage(*a, **k)
        cap["foot"] = r[1]
        return r

    def place(rng, alpha_shape, bars, foot_pts, ratio, bg):
        r = orig_place(rng, alpha_shape, bars, foot_pts, ratio, bg)
        if r:
            M = r[0]
            x0, y0, x1, y1 = bars
            edge = g.transform(M, g.rect(x0, y0, x1, y1))
            axis = float(np.arctan2(edge[1][1] - edge[0][1], edge[1][0] - edge[0][0]))
            cnts, _ = cv2.findContours((cap["foot"] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(np.vstack(cnts).reshape(-1, 2)).reshape(-1, 2).astype(np.float32)
            cap["fixed"], cap["buggy"] = g.ratio_box(g.transform(M, hull), axis, ratio), r[1]
            raise Stop
        return r

    g.apply_damage, g.place = apply_damage, place
    W, H = g.CFG["W"], g.CFG["H"]
    fonts = g.Fonts(None)
    stats, bad = collections.defaultdict(list), 0
    for idx in sorted(i for i, r in meta.items() if r["split"] == "val"):
        try:
            g.generate_one(idx, args.seed, fonts, [])
        except Stop:
            pass
        lbl = (args.data / "labels" / "val" / f"{idx:06d}.txt").read_text().split()
        old = np.float32(np.float32(lbl[1:9]).reshape(4, 2) * [W, H])
        if np.abs(old - cap["buggy"]).max() > 1.0:
            bad += 1
            continue
        fixed = np.float32(cap["fixed"])
        inter, _ = cv2.intersectConvexConvex(old, fixed)
        stats[meta[idx]["damage"]].append(inter / (cv2.contourArea(old) + cv2.contourArea(fixed) - inter))
        norm = (fixed / np.float32([W, H])).reshape(-1)
        (args.out / "labels" / "val" / f"{idx:06d}.txt").write_text(f"{lbl[0]} " + " ".join(f"{v:.6f}" for v in norm) + "\n")
    print("재현 실패:", bad)
    for k, v in sorted(stats.items()):
        v = np.array(v)
        print(f"{k:8s} {len(v):4d}장 · 원래 vs 고친 정답 IoU<0.9 {int((v < 0.9).sum()):4d} · 중앙값 {np.median(v):.3f}")


if __name__ == "__main__":
    main()
