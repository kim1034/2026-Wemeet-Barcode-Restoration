#!/usr/bin/env python3
"""v2(현재 배포) 와 새 가중치들을 같은 조건에서 비교한다. compare_v2_v4.py 의 확장판.

1. 합성 val(v2 학습 때의 val 2,000장, v2 생성기)에서 `model.val()` — mAP·P·R
2. 같은 val 에서 회전 IoU 성공률 — 가장 확신 높은 박스와 정답의 회전 IoU ≥ 0.5 비율.
   11/04 v4_n 은 박스가 막대 높이 절반만큼 밀려 있었는데 mAP50 만으로는 잘 안 보였다.
   생성기가 다른 val 이라 새 생성기의 라벨 버그가 있으면 여기서 드러난다
3. benchmark-v2 100장에서 탐지 성공률 — 확신도 ≥ CONF_THRESHOLD 인 박스가 하나라도
   나오면 성공 (위치 정확도는 보지 않음, 2026-10-02·11-02 기록과 같은 기준)

모든 모델에 CLAHE 없이 원본 이미지를 넣는다 (모델끼리만 비교하려고).

사용법::

    python compare_models.py --model v2=barcode_v2_best.pt --model v5_n=v5_n/best.pt \
        --val-data dataset_720/data.yaml --bench downloads/hf/benchmark/v2
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CONF_THRESHOLD = 0.25  # wemeet.ai.detection.CONF_THRESHOLD 와 같은 값


def imread(path: Path) -> np.ndarray:
    # cv2.imread 는 Windows 에서 비ASCII 경로를 못 읽는다
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def rotated_iou(a, b) -> float:
    a, b = np.float32(a).reshape(4, 2), np.float32(b).reshape(4, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def val_metrics(model: YOLO, data: str, imgsz: int) -> dict:
    m = model.val(data=data, imgsz=imgsz, single_cls=True, batch=8, plots=False, verbose=False)
    return {
        "precision": m.box.mp,
        "recall": m.box.mr,
        "mAP50": m.box.map50,
        "mAP50-95": m.box.map,
    }


def iou_success(model: YOLO, root: Path, imgsz: int) -> dict:
    ious, missed = [], 0
    for img_p in sorted((root / "images" / "val").glob("*.jpg")):
        img = imread(img_p)
        h, w = img.shape[:2]
        vals = (root / "labels" / "val" / f"{img_p.stem}.txt").read_text().split()
        gt = np.float32(vals[1:9]).reshape(4, 2) * [w, h]
        r = model.predict(img, imgsz=imgsz, conf=CONF_THRESHOLD, verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            missed += 1
            continue
        top = int(r.obb.conf.argmax())
        ious.append(rotated_iou(r.obb.xyxyxyxy[top].cpu().numpy(), gt))
    n = len(ious) + missed
    return {
        "images": n,
        "missed": missed,
        "iou_ge_0.5": round(sum(i >= 0.5 for i in ious) / n, 4),  # 놓친 장은 실패로 센다
        "iou_ge_0.75": round(sum(i >= 0.75 for i in ious) / n, 4),
        "iou_median": round(statistics.median(ious), 4) if ious else 0.0,
    }


def bench(model: YOLO, images: list, buckets: list, imgsz: int) -> dict:
    model.predict(images[0], imgsz=imgsz, conf=CONF_THRESHOLD, verbose=False)  # warmup
    hit, ms = {}, []
    for image, bucket in zip(images, buckets):
        t0 = time.perf_counter()
        r = model.predict(image, imgsz=imgsz, conf=CONF_THRESHOLD, verbose=False)[0]
        ms.append((time.perf_counter() - t0) * 1000)
        found = r.obb is not None and len(r.obb) > 0
        hit.setdefault(bucket, []).append(found)
    out = {b: f"{sum(v)}/{len(v)}" for b, v in sorted(hit.items())}
    out["all"] = f"{sum(sum(v) for v in hit.values())}/{len(images)}"
    out["p50_ms"] = round(statistics.median(ms), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="이름=가중치 (여러 번)")
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1280, 1920])
    ap.add_argument("--out", type=Path, default=Path("compare_models.json"))
    args = ap.parse_args()

    val_root = Path(args.val_data).parent
    records = [json.loads(line) for line in open(args.bench / "manifest.jsonl", encoding="utf-8")][1:]
    images = [imread(args.bench / r["file"]) for r in records]
    buckets = [r.get("bucket", "all") for r in records]

    result = {}
    for spec in args.model:
        name, weights = spec.split("=", 1)
        model = YOLO(weights)
        result[name] = {}
        for imgsz in args.imgsz:
            result[name][f"val_{imgsz}"] = val_metrics(model, args.val_data, imgsz)
            result[name][f"iou_{imgsz}"] = iou_success(model, val_root, imgsz)
            result[name][f"bench_{imgsz}"] = bench(model, images, buckets, imgsz)
        print(name, json.dumps(result[name], ensure_ascii=False, indent=1), flush=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
