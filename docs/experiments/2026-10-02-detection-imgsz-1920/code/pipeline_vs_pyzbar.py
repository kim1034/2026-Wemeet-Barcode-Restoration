#!/usr/bin/env python3
"""조건 A(원본 사진 + pyzbar 단독) vs D(우리 파이프라인 전체)를 비교한다.

`docs/benchmark/README.md` 의 A/D 정의와 같다. 판독 성공 = 정답 번호와 완전 일치.

사용법::

    python pipeline_vs_pyzbar.py --data-root downloads/hf/benchmark/v2
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode

from wemeet.sw.pipeline import run as pipeline_run


def classify(decoded: str | None, gt: str) -> str:
    if decoded is None:
        return "fail"
    return "success" if decoded == gt else "misread"


def percentile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] if f == c else vals[f] + (vals[c] - vals[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    args = ap.parse_args()

    records = [
        json.loads(line)
        for line in open(args.data_root / "manifest.jsonl", encoding="utf-8")
    ][1:]

    results = {
        "pyzbar": {"success": 0, "misread": 0, "fail": 0, "ms": []},
        "pipeline": {"success": 0, "misread": 0, "fail": 0, "ms": []},
    }

    for i, r in enumerate(records):
        gt = r["text"]
        img_bgr = cv2.imread(str(args.data_root / r["file"]))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        found = pyzbar_decode(Image.fromarray(gray))
        dt = (time.perf_counter() - t0) * 1000
        decoded1 = found[0].data.decode() if found else None
        results["pyzbar"][classify(decoded1, gt)] += 1
        results["pyzbar"]["ms"].append(dt)

        t0 = time.perf_counter()
        pr = pipeline_run(img_bgr)
        dt = (time.perf_counter() - t0) * 1000
        results["pipeline"][classify(pr.decode.text, gt)] += 1
        results["pipeline"]["ms"].append(dt)

        if (i + 1) % 20 == 0:
            print(f"...{i + 1}/{len(records)}")

    n = len(records)
    print()
    print(f"결과 — {n}장, 정답 번호와 완전 일치만 성공")
    print()
    print(f"{'':20s}{'① 원본 사진 + pyzbar 단독':28s}{'② 우리 파이프라인 전체':28s}")
    for label, key in (("판독 성공", "success"), ):
        a, b = results["pyzbar"][key], results["pipeline"][key]
        print(f"{label:20s}{f'{a}/{n} ({100*a/n:.0f}%)':28s}{f'{b}/{n} ({100*b/n:.0f}%)':28s}")
    for label, key in (("오독(틀린 번호)", "misread"), ("실패", "fail")):
        a, b = results["pyzbar"][key], results["pipeline"][key]
        print(f"{label:20s}{str(a):28s}{str(b):28s}")
    p50a, p95a = percentile(results["pyzbar"]["ms"], 0.5), percentile(results["pyzbar"]["ms"], 0.95)
    p50b, p95b = percentile(results["pipeline"]["ms"], 0.5), percentile(results["pipeline"]["ms"], 0.95)
    print(f"{'시간 p50/p95':20s}{f'{p50a:.0f}ms/{p95a:.0f}ms':28s}{f'{p50b:.0f}ms/{p95b:.0f}ms':28s}")


if __name__ == "__main__":
    main()
