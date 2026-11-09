#!/usr/bin/env python3
"""v2(현재 배포) 와 v4_n 가중치를 같은 조건에서 비교한다.

1. 합성 val(v2 학습 때의 val 2,000장)에서 `model.val()` — mAP·P·R
2. benchmark-v2 100장에서 탐지 성공률 — 확신도 ≥ CONF_THRESHOLD 인 박스가 하나라도
   나오면 성공 (위치 정확도는 보지 않음, 2026-10-02·11-02 기록과 같은 기준)

두 모델 모두 CLAHE 없이 원본 이미지를 넣는다 (모델끼리만 비교하려고).

사용법::

    python compare_v2_v4.py --v2 barcode_v2_best.pt --v4 v4_n/weights/best.pt \
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


def val_metrics(model: YOLO, data: str, imgsz: int) -> dict:
    m = model.val(data=data, imgsz=imgsz, single_cls=True, batch=8, plots=False, verbose=False)
    return {
        "precision": m.box.mp,
        "recall": m.box.mr,
        "mAP50": m.box.map50,
        "mAP50-95": m.box.map,
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
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1280, 1920])
    ap.add_argument("--out", type=Path, default=Path("compare_v2_v4.json"))
    args = ap.parse_args()

    records = [json.loads(line) for line in open(args.bench / "manifest.jsonl", encoding="utf-8")][1:]
    images = [imread(args.bench / r["file"]) for r in records]
    buckets = [r.get("bucket", "all") for r in records]

    result = {}
    for name, weights in (("v2", args.v2), ("v4_n", args.v4)):
        model = YOLO(weights)
        result[name] = {"val_1280": val_metrics(model, args.val_data, 1280)}
        for imgsz in args.imgsz:
            result[name][f"bench_{imgsz}"] = bench(model, images, buckets, imgsz)
        print(name, json.dumps(result[name], ensure_ascii=False, indent=1))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
