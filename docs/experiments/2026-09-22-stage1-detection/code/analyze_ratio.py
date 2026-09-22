#!/usr/bin/env python3
"""
5:3 / 6:4 비율 판정 실패 분석

알고 싶은 것
  1. 어떤 손상 종류에서 비율을 틀리는가
  2. 틀린 판정이 기준선(1.583) 근처의 '애매한' 경우인가, 멀리 벗어난 '확실히 틀린' 경우인가
  3. 애매한 구간만 따로 표시하면 (→ 두 비율로 모두 크롭해서 디코딩으로 결정)
     나머지 확신 구간의 정확도가 얼마나 올라가는가

메타 정보는 dataset_720 과 dataset 양쪽의 meta_*.csv 를 모두 읽어 합친다.
(첫 1만 장은 손상 기능 이전에 만든 것이라 damage 열이 없음 → 'normal' 로 처리)

사용법
  python analyze_ratio.py --weights runs\\obb\\runs\\barcode_v2\\weights\\best.pt
결과
  화면 요약 + ratio_analysis.csv (2,000장 전부의 판정 기록)
"""
import argparse
import ast
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THRESH = (5 / 3 + 3 / 2) / 2   # 1.5833
TRUE_RATIO = {0: 5 / 3, 1: 3 / 2}
NAMES = {0: "5:3", 1: "6:4"}


def poly_iou(a, b):
    import cv2
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def load_meta(dirs):
    meta = {}
    for d in dirs:
        for f in sorted(Path(d).glob("meta_*.csv")):
            with open(f, encoding="utf-8") as fp:
                for row in csv.DictReader(fp):
                    try:
                        key = f"{int(row['index']):06d}"
                    except (KeyError, ValueError):
                        continue
                    if key not in meta or row.get("damage"):
                        meta[key] = row
    return meta


def size_bin(row):
    try:
        w = ast.literal_eval(row.get("bars_screen_px", ""))[0]
    except Exception:
        return "?"
    return "<200px" if w < 200 else ("200-400px" if w < 400 else "400px+")


def analyze(rows):
    """rows: dict 목록 (gt, found, ratio, damage, size ...) → 요약 출력."""
    det = [r for r in rows if r["found"]]
    print(f"\n전체 {len(rows):,}장 · 탐지 {len(det):,}장 ({len(det) / len(rows):.1%})")

    # 1) 손상 종류별
    groups = defaultdict(list)
    for r in rows:
        groups[r["damage"]].append(r)
        for part in r["parts"].split("+"):
            if part and part not in ("-", r["damage"]):
                groups[f"  └ 포함:{part}"].append(r)
    print("\n[1] 손상 종류별 (비율 판정은 탐지된 것 중)")
    print(f"{'손상':<22}{'장수':>6}{'탐지율':>9}{'비율판정':>9}{'최종성공':>9}")
    order = sorted(groups, key=lambda k: (k.startswith("  "), -len(groups[k])))
    for k in order:
        g = groups[k]
        d = [r for r in g if r["found"]]
        acc = sum(r["correct"] for r in d) / len(d) if d else 0
        print(f"{k:<22}{len(g):>6}{len(d) / len(g):>9.1%}{acc:>9.1%}{sum(r['correct'] for r in g) / len(g):>9.1%}")

    # 2) 예측 비율 분포
    print(f"\n[2] 예측 박스 비율 분포 (탐지된 것, 기준선 {THRESH:.4f})")
    for c in (0, 1):
        v = np.array([r["ratio"] for r in det if r["gt"] == c])
        if len(v):
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            print(f"  정답 {NAMES[c]} (참값 {TRUE_RATIO[c]:.3f}): 중앙값 {p50:.3f} · 하위5% {p5:.3f} · 상위5% {p95:.3f}")
    wrong = [r for r in det if not r["correct"]]
    if wrong:
        dist = np.array([abs(r["ratio"] - THRESH) for r in wrong])
        print(f"  틀린 {len(wrong)}건이 기준선에서 떨어진 거리: 중앙값 {np.median(dist):.3f} · "
              f"0.03 이내 {np.mean(dist <= 0.03):.0%} · 0.05 이내 {np.mean(dist <= 0.05):.0%}")

    # 3) 애매 구간 전략
    print("\n[3] 애매 구간 전략: 기준선 ±폭 안이면 '애매' → 두 비율 모두 크롭해서 디코딩으로 결정")
    print(f"{'±폭':>6}{'애매 비율':>10}{'확신 구간 정확도':>16}{'틀린 것 중 애매로 잡힌 비율':>26}")
    for w in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07):
        amb = [r for r in det if abs(r["ratio"] - THRESH) <= w]
        conf = [r for r in det if abs(r["ratio"] - THRESH) > w]
        acc = sum(r["correct"] for r in conf) / len(conf) if conf else 0
        caught = sum(1 for r in wrong if abs(r["ratio"] - THRESH) <= w) / len(wrong) if wrong else 0
        print(f"{w:>6.2f}{len(amb) / len(det):>10.1%}{acc:>16.1%}{caught:>26.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="dataset_720")
    ap.add_argument("--meta-dirs", nargs="+", default=["dataset_720", "dataset"])
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):  # 윈도우 cp949 콘솔 대응
        try:
            st.reconfigure(errors="replace")
        except Exception:
            pass

    meta = load_meta(args.meta_dirs)
    n_dmg = sum(1 for m in meta.values() if m.get("damage"))
    print(f"메타 {len(meta):,}건 연결 (손상 정보 포함 {n_dmg:,}건)")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    paths = [str(p) for p in sorted((Path(args.data) / "images" / "val").glob("*.jpg"))]
    rows = []
    for i in range(0, len(paths), args.batch):   # 한 번에 batch 장씩 (VRAM 보호)
        for res in model.predict(paths[i:i + args.batch], imgsz=args.imgsz, conf=args.conf, verbose=False):
            stem = Path(res.path).stem
            H, W = res.orig_shape
            v = (Path(args.data) / "labels" / "val" / f"{stem}.txt").read_text().split()
            gt = int(v[0])
            gt_pts = np.array(v[1:], float).reshape(4, 2) * [W, H]
            found, iou, ratio, pred = False, 0.0, 0.0, -1
            if res.obb is not None and len(res.obb):
                k = int(res.obb.conf.argmax())
                _, _, bw, bh, _ = res.obb.xywhr[k].cpu().numpy()
                ratio = float(max(bw, bh) / max(min(bw, bh), 1e-6))
                pred = 0 if ratio >= THRESH else 1
                iou = poly_iou(res.obb.xyxyxyxy[k].cpu().numpy(), gt_pts)
                found = iou >= 0.5
            m = meta.get(stem)
            damage = (m.get("damage") or "normal") if m else "?"
            rows.append({"image": stem, "gt": gt, "found": found, "iou": round(iou, 3),
                         "ratio": round(ratio, 4), "pred": pred, "correct": found and pred == gt,
                         "damage": damage, "parts": (m or {}).get("damage_parts", "") or "",
                         "size": size_bin(m) if m else "?",
                         "torn_ratio": (m or {}).get("torn_ratio", ""), "glare_cover": (m or {}).get("glare_cover", "")})
        if (i // args.batch) % 25 == 0:
            print(f"  {min(i + args.batch, len(paths)):,}/{len(paths):,}", flush=True)

    analyze(rows)
    with open("ratio_analysis.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n💾 ratio_analysis.csv (엑셀로 열면 이미지별 판정 기록)")


if __name__ == "__main__":
    main()
