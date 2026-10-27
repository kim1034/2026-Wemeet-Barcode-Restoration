#!/usr/bin/env python3
"""탐지(Stage 1)만 떼어내서 imgsz별 추론 시간을 잰다. 전·후처리(CLAHE)는 포함, 크롭·기하추정은 제외.

사용법::

    python detect_latency.py --data-root downloads/hf/benchmark/v2 --imgsz 640 960 1280 1536 1920
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from wemeet.ai.detection.enhance import enhance_for_detection
from wemeet.ai.detection.yolo_obb import load_model


def percentile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] if f == c else vals[f] + (vals[c] - vals[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 960, 1280, 1536, 1920])
    args = ap.parse_args()

    records = [
        json.loads(line)
        for line in open(args.data_root / "manifest.jsonl", encoding="utf-8")
    ][1:]
    model = load_model()
    enhs = [
        enhance_for_detection(cv2.imread(str(args.data_root / r["file"])), 2.0, (8, 8))
        for r in records
    ]

    for imgsz in args.imgsz:
        model.predict(enhs[0], imgsz=imgsz, conf=0.25, verbose=False)  # warmup
        times = []
        for enh in enhs:
            t0 = time.perf_counter()
            model.predict(enh, imgsz=imgsz, conf=0.25, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        print(
            f"imgsz={imgsz}: avg={avg:.0f}ms p50={percentile(times, 0.5):.0f}ms "
            f"p95={percentile(times, 0.95):.0f}ms min={min(times):.0f}ms max={max(times):.0f}ms"
        )


if __name__ == "__main__":
    main()
