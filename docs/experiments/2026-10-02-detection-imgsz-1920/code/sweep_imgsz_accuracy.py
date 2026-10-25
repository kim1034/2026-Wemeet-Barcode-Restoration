#!/usr/bin/env python3
"""imgsz별 Stage 1 탐지 성공률을 버킷별로 비교한다.

사용법::

    python sweep_imgsz_accuracy.py --data-root downloads/benchmark-v1 --imgsz 960 1280 1920
    python sweep_imgsz_accuracy.py --data-root downloads/hf/benchmark/v2 --imgsz 960 1920

data-root 는 manifest.jsonl 과 images/ 를 담은 폴더다 (HF `123metro/barcode-datasets`
의 `benchmark/v1` 또는 `benchmark/v2`). manifest.jsonl 첫 줄은 `_stats` 메타데이터라
건너뛴다. 탐지 성공 = conf>=0.25 에서 OBB 박스가 1개 이상 나옴 (위치 정확도는 보지 않음).
"""

import argparse
import json
from pathlib import Path

import cv2

from wemeet.ai.detection.enhance import enhance_for_detection
from wemeet.ai.detection.yolo_obb import load_model

CONF_THRESHOLD = 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[960, 1280, 1920])
    args = ap.parse_args()

    records = [
        json.loads(line)
        for line in open(args.data_root / "manifest.jsonl", encoding="utf-8")
    ][1:]

    model = load_model()
    enhs = []
    for r in records:
        img = cv2.imread(str(args.data_root / r["file"]))
        enhs.append((r.get("bucket", "all"), enhance_for_detection(img, 2.0, (8, 8))))

    buckets = sorted({b for b, _ in enhs})
    counts = {b: {"total": 0} for b in buckets}
    for imgsz in args.imgsz:
        for b in buckets:
            counts[b][imgsz] = 0
        for bucket, enh in enhs:
            res = model.predict(enh, imgsz=imgsz, conf=CONF_THRESHOLD, verbose=False)[0]
            ok = res.obb is not None and len(res.obb) > 0
            if imgsz == args.imgsz[0]:
                counts[bucket]["total"] += 1
            counts[bucket][imgsz] += 1 if ok else 0

    header = "버킷".ljust(6) + "전체".ljust(6) + "".join(f"{sz}".ljust(16) for sz in args.imgsz)
    print(header)
    totals = {"total": 0, **{sz: 0 for sz in args.imgsz}}
    for b in buckets:
        c = counts[b]
        totals["total"] += c["total"]
        row = b.ljust(6) + str(c["total"]).ljust(6)
        for sz in args.imgsz:
            totals[sz] += c[sz]
            pct = 100 * c[sz] / c["total"] if c["total"] else 0
            row += f"{c[sz]}/{c['total']} ({pct:.0f}%)".ljust(16)
        print(row)
    row = "전체".ljust(6) + str(totals["total"]).ljust(6)
    for sz in args.imgsz:
        pct = 100 * totals[sz] / totals["total"] if totals["total"] else 0
        row += f"{totals[sz]}/{totals['total']} ({pct:.0f}%)".ljust(16)
    print(row)


if __name__ == "__main__":
    main()
