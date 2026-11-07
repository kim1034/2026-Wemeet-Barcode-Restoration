#!/usr/bin/env python3
"""
학습 전에 합성 데이터셋을 검사한다. 하나라도 실패하면 exit 1 (파이프라인이 학습으로 넘어가지 않음).

2026-11-04 라벨 버그(정답 박스 절반이 막대 높이의 반만큼 밀림)를 36시간 학습한 뒤에야 찾았다.
같은 생성기로 만든 val 로는 버그가 안 보이므로, 생성기와 무관한 기준으로 본다.

  1. 형식   : 라벨 한 줄 9개 값, 클래스 0/1, 좌표 0~1, 이미지·라벨 짝, 이미지 크기
  2. 위치   : 생성기가 메타 CSV 에 남긴 label_cover (발자국 중 박스 안 비율) 가 전부 기준 이상
  3. 그림   : 무작위 N장에 정답 박스를 그린 격자 (사람이 눈으로 확인)
  4. v2 독립 검사 (--zero-shot) : 박스가 정확한 v2 모델의 예측과 정답의 회전 IoU.
             라벨이 밀려 있으면 IoU 가 낮게 나온다

  python qa_dataset.py dataset_v5 --size 1920 1080
  python qa_dataset.py dataset_v5 --size 1920 1080 --zero-shot best_v2.pt --imgsz 1920
  python qa_dataset.py dataset_720 --size 1280 720 --no-meta --zero-shot best_v2.pt   # 기준선
"""

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CONF_THRESHOLD = 0.25  # wemeet.ai.detection.CONF_THRESHOLD 와 같은 값


def imread(path):
    # cv2.imread 는 Windows 에서 비ASCII 경로를 못 읽는다
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def read_label(path):
    """첫 줄 → (클래스, 정규화 4꼭짓점 (4,2)). 형식이 틀리면 ValueError."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) != 1:
        raise ValueError(f"줄 수 {len(lines)}")
    vals = lines[0].split()
    if len(vals) != 9:
        raise ValueError(f"값 {len(vals)}개")
    cls, pts = int(vals[0]), np.float32(vals[1:]).reshape(4, 2)
    if cls not in (0, 1):
        raise ValueError(f"클래스 {cls}")
    if not np.isfinite(pts).all() or pts.min() < 0 or pts.max() > 1:
        raise ValueError(f"좌표 범위 {pts.min():.3f}~{pts.max():.3f}")
    return cls, pts


def rotated_iou(a, b):
    a, b = np.float32(a).reshape(4, 2), np.float32(b).reshape(4, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def check_format(root, splits, size):
    bad, pairs = [], []
    for split in splits:
        imgs = {p.stem: p for p in (root / "images" / split).glob("*.jpg")}
        lbls = {p.stem: p for p in (root / "labels" / split).glob("*.txt")}
        for stem in sorted(set(imgs) ^ set(lbls)):
            bad.append(f"{split}/{stem}: 이미지·라벨 짝 없음")
        for stem in sorted(set(imgs) & set(lbls)):
            try:
                read_label(lbls[stem])
                with Image.open(imgs[stem]) as im:  # 헤더만 읽는다
                    if size and tuple(im.size) != tuple(size):
                        raise ValueError(f"이미지 크기 {im.size}")
            except Exception as e:  # noqa: BLE001  # 어떤 이유든 불량으로 기록
                bad.append(f"{split}/{stem}: {e}")
                continue
            pairs.append((split, imgs[stem], lbls[stem]))
    return pairs, bad


def check_meta(root, n_images, min_cover):
    rows = []
    for p in sorted(root.glob("meta_*.csv")):
        with open(p, encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    if not rows:
        return {"ok": False, "error": "meta_*.csv 없음"}
    if "label_cover" not in rows[0]:
        return {"ok": False, "error": "label_cover 열 없음 (수정 전 생성기로 만든 데이터)"}
    cover = np.float64([r["label_cover"] for r in rows])
    fill = np.float64([r["box_fill"] for r in rows])
    low = [r["index"] for r, c in zip(rows, cover) if c < min_cover]
    q = lambda a: {f"p{k}": round(float(np.percentile(a, k)), 4) for k in (0, 1, 50, 99)}
    return {
        "ok": not low and len(rows) == n_images,
        "rows": len(rows), "images": n_images,
        "label_cover": q(cover), "box_fill": q(fill),
        f"cover_below_{min_cover}": len(low), "examples": low[:20],
    }


def draw_grid(pairs, out, n, seed):
    pick = random.Random(seed).sample(pairs, min(n, len(pairs)))
    cells = []
    for split, img_p, lbl_p in pick:
        img = imread(img_p)
        h, w = img.shape[:2]
        _, pts = read_label(lbl_p)
        poly = (pts * [w, h]).astype(np.int32)
        cv2.polylines(img, [poly], True, (0, 255, 0), max(2, w // 640), cv2.LINE_AA)
        cell = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(cell, f"{split}/{img_p.stem}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cells.append(cell)
    cells += [np.full_like(cells[0], 40)] * (-len(cells) % 6)
    grid = np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)])
    cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(out))


def damage_of(root):
    """번호 → 손상 종류 (메타 CSV 의 damage_parts, '-' = 손상 없음). 메타가 없으면 빈 dict."""
    out = {}
    for p in sorted(root.glob("meta_*.csv")):
        with open(p, encoding="utf-8") as f:
            out.update({int(r["index"]): r.get("damage_parts", "-") for r in csv.DictReader(f)})
    return out


def iou_stats(ious, missed):
    return {
        "images": len(ious) + missed, "detected": len(ious), "missed": missed,
        "iou_median": round(statistics.median(ious), 4) if ious else 0.0,
        "iou_ge_0.5": round(sum(i >= 0.5 for i in ious) / len(ious), 4) if ious else 0.0,
        "iou_ge_0.75": round(sum(i >= 0.75 for i in ious) / len(ious), 4) if ious else 0.0,
    }


def zero_shot(pairs, weights, imgsz, device, damage):
    """v2 의 가장 확신 높은 박스와 정답의 회전 IoU. 탐지 실패는 IoU 통계에서 뺀다.

    판정은 손상 없는 장면('-')으로 한다 — 반사·찢김은 v2 자체가 박스를 잘못 잡는 경우가 많고
    (찢긴 부분까지 정답에 포함), 11/04 버그는 오히려 변형 없는 직사각형에서 났다."""
    from ultralytics import YOLO

    model = YOLO(weights)
    res = {"all": ([], 0), "clean": ([], 0)}
    for _, img_p, lbl_p in pairs:
        img = imread(img_p)
        h, w = img.shape[:2]
        _, pts = read_label(lbl_p)
        r = model.predict(img, imgsz=imgsz, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
        groups = ["all"] + (["clean"] if damage.get(int(img_p.stem), "-") == "-" else [])
        for g in groups:
            ious, missed = res[g]
            if r.obb is None or len(r.obb) == 0:
                res[g] = (ious, missed + 1)
                continue
            top = int(r.obb.conf.argmax())
            ious.append(rotated_iou(r.obb.xyxyxyxy[top].cpu().numpy(), pts * [w, h]))
    return {g: iou_stats(*v) for g, v in res.items()}


def main():
    ap = argparse.ArgumentParser(description="합성 데이터셋 학습 전 검사")
    ap.add_argument("data", type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--size", nargs=2, type=int, default=None, help="기대 이미지 크기 W H")
    ap.add_argument("--min-cover", type=float, default=0.98)
    ap.add_argument("--no-meta", action="store_true", help="메타 검사 생략 (다른 생성기로 만든 기준 데이터용)")
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--zero-shot", default=None, help="v2 가중치 경로")
    ap.add_argument("--zs-split", default="val")
    ap.add_argument("--zs-max", type=int, default=0, help="v2 검사 장수 상한 (0 = 전부)")
    # 기준값 근거 (11/04, 손상 없는 장면): v2 생성기 val·수정된 v5 는 아래 기준 위, 버그 있는 v3 는 아래
    ap.add_argument("--zs-min-median", type=float, default=0.85)
    ap.add_argument("--zs-min-ge50", type=float, default=0.95)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None, help="보고서·격자 저장 폴더 (기본: 데이터 폴더/qa)")
    args = ap.parse_args()

    out = args.out or args.data / "qa"
    out.mkdir(parents=True, exist_ok=True)
    report, ok = {"data": str(args.data.resolve())}, True

    pairs, bad = check_format(args.data, args.splits, args.size)
    report["format"] = {"pairs": len(pairs), "bad": len(bad), "examples": bad[:20]}
    ok &= not bad and bool(pairs)
    print(f"[형식] 정상 {len(pairs):,} · 불량 {len(bad):,}", *bad[:5], sep="\n  ")

    if not args.no_meta:
        report["meta"] = check_meta(args.data, len(pairs) + len(bad), args.min_cover)
        ok &= report["meta"]["ok"]
        print(f"[위치] {json.dumps(report['meta'], ensure_ascii=False)}")

    if args.grid and pairs:
        draw_grid(pairs, out / "qa_grid.jpg", args.grid, seed=0)
        print(f"[그림] {out / 'qa_grid.jpg'}")

    if args.zero_shot:
        zs_pairs = [p for p in pairs if p[0] == args.zs_split]
        if args.zs_max:
            zs_pairs = random.Random(0).sample(zs_pairs, min(args.zs_max, len(zs_pairs)))
        z = zero_shot(zs_pairs, args.zero_shot, args.imgsz, args.device, damage_of(args.data))
        c = z["clean"]
        z["pass"] = c["detected"] >= 20 and c["iou_median"] >= args.zs_min_median and c["iou_ge_0.5"] >= args.zs_min_ge50
        report["zero_shot"] = z
        ok &= z["pass"]
        print(f"[v2 독립 검사] {json.dumps(z, ensure_ascii=False)}")

    report["pass"] = bool(ok)
    (out / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'✅ 통과' if ok else '❌ 실패'} → {out / 'qa_report.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
