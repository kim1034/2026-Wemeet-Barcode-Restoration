#!/usr/bin/env python3
"""
단일 클래스 모델 평가: 바코드를 찾았는지 + 5:3 / 6:4 를 박스 모양으로 맞게 판정했는지

라벨의 클래스(0=5:3, 1=6:4)는 정답 확인용으로만 쓰고,
모델은 '바코드 하나'만 찾는다. 판정은 예측 박스의 긴 변 ÷ 짧은 변이
1.583 (5/3 과 3/2 의 중간) 보다 크면 5:3, 작으면 6:4.

출력
  - 탐지율 (IoU 0.5 이상으로 찾은 비율), 평균 IoU
  - 비율 판정 정확도 + 2x2 혼동표
  - 손상 종류별 / 막대 크기별 성적 (meta_*.csv 가 데이터 폴더에 있으면)
  - 실패 목록 failures.csv

사용법 (학습 끝난 컨테이너에서, 반납 전에)
  python eval_ratio.py --weights runs/obb/barcode_v2/weights/best.pt --data dataset_720
"""
import argparse
import ast
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RATIO = {0: 5 / 3, 1: 3 / 2}
THRESH = (5 / 3 + 3 / 2) / 2   # 1.5833
NAMES = {0: "5:3", 1: "6:4"}


def ensure_cudnn():
    """train_cheetah.py 와 같은 cuDNN 충돌 대응 (pip nvidia-cudnn 경로로 재실행)."""
    if os.environ.get("_CUDNN_FIXED") or os.name == "nt":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cudnn")
        lib = os.path.join(list(spec.submodule_search_locations)[0], "lib") if spec else None
    except Exception:
        lib = None
    if lib and os.path.isdir(lib) and lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        env = dict(os.environ, _CUDNN_FIXED="1", LD_LIBRARY_PATH=lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


def decide_ratio(w, h):
    """박스 가로·세로 → (판정 클래스, 긴변/짧은변)."""
    r = max(w, h) / max(min(w, h), 1e-6)
    return (0 if r >= THRESH else 1), r


def poly_iou(a, b):
    import cv2
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    ua = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def load_meta(data_dir):
    meta = {}
    for f in Path(data_dir).glob("meta_*.csv"):
        with open(f, encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                meta[f"{int(row['index']):06d}"] = row
    return meta


def size_bin(row):
    try:
        w = ast.literal_eval(row["bars_screen_px"])[0]
    except Exception:
        return "?"
    for lo, hi, name in [(0, 200, "<200px"), (200, 400, "200-400px"), (400, 10 ** 9, "400px+")]:
        if lo <= w < hi:
            return name
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="dataset_720")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=8, help="한 번에 추론할 장수 (VRAM 6GB면 8)")
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):  # 윈도우 cp949 콘솔 대응
        try:
            st.reconfigure(errors="replace")
        except Exception:
            pass
    ensure_cudnn()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    val_dir = Path(args.data) / "images" / "val"
    imgs = sorted(val_dir.glob("*.jpg"))
    meta = load_meta(args.data)
    print(f"val {len(imgs):,}장 평가 · 판정 기준 비율 {THRESH:.4f} (이상 5:3 / 미만 6:4)")

    rows, conf_mat = [], np.zeros((2, 2), int)
    groups = defaultdict(lambda: [0, 0, 0])  # 개수, 탐지, 탐지+판정정답
    paths = [str(p) for p in imgs]
    # 경로 리스트를 통째로 넘기면 Ultralytics 8.4 는 2000장을 한 배치로 올려 VRAM 이 터진다 → 직접 나눠서 넣기
    def chunks():
        for i in range(0, len(paths), args.batch):
            yield from model.predict(paths[i:i + args.batch], imgsz=args.imgsz, conf=args.conf,
                                     verbose=False)
            if (i // args.batch) % 25 == 0:
                print(f"  {min(i + args.batch, len(paths)):,}/{len(paths):,}", flush=True)

    for res in chunks():
        stem = Path(res.path).stem
        H, W = res.orig_shape
        line = (Path(args.data) / "labels" / "val" / f"{stem}.txt").read_text().split()
        gt_cls = int(line[0])
        gt_pts = np.array(line[1:], float).reshape(4, 2) * [W, H]

        found, iou, pred_cls, ratio = False, 0.0, -1, 0.0
        if res.obb is not None and len(res.obb):
            k = int(res.obb.conf.argmax())
            poly = res.obb.xyxyxyxy[k].cpu().numpy()
            _, _, bw, bh, _ = res.obb.xywhr[k].cpu().numpy()
            iou = poly_iou(poly, gt_pts)
            pred_cls, ratio = decide_ratio(bw, bh)
            found = iou >= 0.5
        if found:
            conf_mat[gt_cls, pred_cls] += 1
        ok = found and pred_cls == gt_cls

        m = meta.get(stem, {})
        for key in ("전체", f"손상:{m.get('damage', '?')}", f"크기:{size_bin(m) if m else '?'}", f"정답:{NAMES[gt_cls]}"):
            g = groups[key]
            g[0] += 1
            g[1] += found
            g[2] += ok
        rows.append({"image": stem, "gt": NAMES[gt_cls], "found": found, "iou": round(iou, 3),
                     "pred": NAMES.get(pred_cls, "-"), "ratio": round(ratio, 4), "correct": ok,
                     "damage": m.get("damage", "?"), "bars_px": m.get("bars_screen_px", "?")})

    ious = [r["iou"] for r in rows if r["found"]]
    print(f"\n탐지율 (IoU≥0.5): {groups['전체'][1] / len(rows):.1%} · 평균 IoU {np.mean(ious):.3f}")
    print(f"비율 판정 정확도 (탐지된 것 중): {np.trace(conf_mat) / max(conf_mat.sum(), 1):.1%}")
    print(f"최종 성공률 (탐지 + 판정 모두 정답): {groups['전체'][2] / len(rows):.1%}")
    print("\n혼동표 (행=정답, 열=판정)")
    print(f"          판정5:3  판정6:4\n  정답5:3  {conf_mat[0, 0]:>6}  {conf_mat[0, 1]:>6}\n  정답6:4  {conf_mat[1, 0]:>6}  {conf_mat[1, 1]:>6}")

    print(f"\n{'그룹':<20}{'장수':>6}{'탐지율':>9}{'최종성공':>9}")
    for key in sorted(groups, key=lambda k: (k.split(':')[0] != '전체', k)):
        n, f, o = groups[key]
        print(f"{key:<20}{n:>6}{f / n:>9.1%}{o / n:>9.1%}")

    fails = [r for r in rows if not r["correct"]]
    with open("failures.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(fails)
    print(f"\n💾 실패 {len(fails)}건 → failures.csv (어떤 손상·크기에서 틀렸는지)")


if __name__ == "__main__":
    main()
