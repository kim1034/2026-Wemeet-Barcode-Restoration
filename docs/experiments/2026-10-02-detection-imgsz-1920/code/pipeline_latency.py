#!/usr/bin/env python3
"""전체 파이프라인(run()) 지연을 재고, 예산 구간별로 몇 장이 초과하는지 센다.

p95 하나만 보면 극단치 하나 때문인지 구분이 안 되므로, 여러 임계값을 넘는
장수를 같이 센다.

사용법::

    python pipeline_latency.py --data-root downloads/hf/benchmark/v2
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from wemeet.sw.pipeline import run as pipeline_run


def percentile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] if f == c else vals[f] + (vals[c] - vals[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--thresholds", nargs="+", type=int, default=[300, 500, 700, 900])
    args = ap.parse_args()

    records = [
        json.loads(line)
        for line in open(args.data_root / "manifest.jsonl", encoding="utf-8")
    ][1:]

    times = []
    for i, r in enumerate(records):
        img = cv2.imread(str(args.data_root / r["file"]))
        t0 = time.perf_counter()
        pipeline_run(img)
        times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 20 == 0:
            print(f"...{i + 1}/{len(records)}")

    avg = sum(times) / len(times)
    print()
    print(
        f"avg={avg:.0f}ms p50={percentile(times, 0.5):.0f}ms "
        f"p95={percentile(times, 0.95):.0f}ms min={min(times):.0f}ms max={max(times):.0f}ms"
    )
    for thresh in args.thresholds:
        n = sum(1 for t in times if t > thresh)
        print(f"  {thresh}ms 초과: {n}장")


if __name__ == "__main__":
    main()
