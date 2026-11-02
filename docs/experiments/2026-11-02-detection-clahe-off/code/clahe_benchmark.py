#!/usr/bin/env python3
"""benchmark-v2 에서 검출 CLAHE on/off 의 탐지 성공률과 CLAHE 자체 시간을 잰다.

`detect()` 를 그대로 부른다 — `use_clahe` 만 바꾼다. 탐지 성공 = `detect()` 가
None 이 아님 (위치 정확도는 보지 않음, 2026-10-02 기록과 같은 기준).

사용법::

    python clahe_benchmark.py --data-root downloads/hf/benchmark/v2 --imgsz 1280 1920

imgsz 를 바꿔 가며 재려고 `wemeet.ai.detection.DEFAULT_IMGSZ` 를 덮어쓴다.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import wemeet.ai.detection as detection
from wemeet.ai.detection.enhance import enhance_for_detection


def clahe_ms(images: list, repeat: int = 3) -> list[float]:
    """이미지마다 enhance_for_detection 을 repeat 번 돌려 중앙값(ms)을 모은다."""
    enhance_for_detection(images[0])  # warmup
    out = []
    for image in images:
        runs = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            enhance_for_detection(image)
            runs.append((time.perf_counter() - t0) * 1000)
        out.append(statistics.median(runs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1280, 1920])
    args = ap.parse_args()

    records = [
        json.loads(line) for line in open(args.data_root / "manifest.jsonl", encoding="utf-8")
    ][1:]
    # cv2.imread 는 Windows 에서 비ASCII 경로를 못 읽는다
    images = [
        cv2.imdecode(np.fromfile(str(args.data_root / r["file"]), np.uint8), cv2.IMREAD_COLOR)
        for r in records
    ]
    if any(image is None for image in images):
        raise SystemExit(f"이미지를 읽지 못했다: {args.data_root}")
    buckets = [r.get("bucket", "all") for r in records]
    names = sorted(set(buckets))

    h, w = images[0].shape[:2]
    times = sorted(clahe_ms(images))
    print(f"CLAHE 시간 ({w}x{h}, OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads, "
          f"{len(images)}장): p50={statistics.median(times):.1f}ms "
          f"p95={times[int(0.95 * (len(times) - 1))]:.1f}ms")

    detection.detect(images[0])  # 모델 로드·warmup
    print(f"\n{'imgsz':<7}{'CLAHE':<7}" + "".join(f"{b:<12}" for b in names) + "전체")
    for imgsz in args.imgsz:
        detection.DEFAULT_IMGSZ = imgsz
        for use_clahe in (False, True):
            hits = [detection.detect(img, use_clahe=use_clahe) is not None for img in images]
            row = f"{imgsz:<7}{'on' if use_clahe else 'off':<7}"
            for b in names:
                n = sum(1 for x in buckets if x == b)
                k = sum(hit for hit, x in zip(hits, buckets) if x == b)
                row += f"{k}/{n}".ljust(12)
            print(row + f"{sum(hits)}/{len(hits)}")


if __name__ == "__main__":
    main()
