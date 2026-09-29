"""E2E 벤치마크 v1 을 돌리고 결과를 기록에 한 줄 추가한다.

같은 100장에 네 조건을 매긴다. 판독 성공 = 정답 번호와 **완전 일치**.

    A  원본 사진을 그대로 decode()          ← 우리 서비스 없이
    B  + 1단계 검출 크롭 (1차 디코딩)
    C  + 2·3단계 기하 보정 1회 (배율 1.0)
    D  + 배율 재시도 = run() 전체           ← 우리 서비스

B·C·D 는 run() 을 한 번만 돌려 결과에서 가른다. run() 은 1차 디코딩이 뭐든
읽으면 거기서 끝나고, 재시도도 뭐든 읽히면 멈추므로 각 조건의 판독 결과가
run() 의 중간 결과와 같다 (`ladder()`).

    uv run python -m scripts.run_benchmark --note "무엇을 바꿨나"

요약은 docs/benchmark/runs.jsonl 에 쌓이고(커밋 대상), 샘플별 결과는
runs/benchmark/ 에 남는다(git 제외).
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import wemeet.sw.decoding as decoding
from scripts.label_recipes import code_commit
from wemeet.schemas import DetectedBarcode, PipelineResult, RectifiedBarcode
from wemeet.sw.decoding import decode
from wemeet.sw.pipeline import run

CONDITIONS = ("A", "B", "C", "D")
LOG = Path("docs/benchmark/runs.jsonl")


def ladder(result: PipelineResult) -> dict[str, str | None]:
    """run() 결과 하나에서 B·C·D 조건의 판독 문자열을 가른다 (None = 못 읽음)."""
    d = result.decode
    early = result.rectified is not None and result.rectified.field is None
    b = d.text if early else None
    c = d.text if early or d.retry_count == 0 else None
    return {"B": b, "C": c, "D": d.text}


def decode_whole(image_bgr: np.ndarray) -> str | None:
    """조건 A — 검출·보정 없이 사진 전체를 같은 decode() 에 넣는다."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    source = DetectedBarcode(crop_bgr_uint8=image_bgr, angle_deg_ccw=0.0, confidence=1.0)
    return decode(RectifiedBarcode(image_gray_uint8=gray, source=source, field=None)).text


def _grade(read: str | None, truth: str) -> str:
    if read is None:
        return "failed"
    return "correct" if read == truth else "misread"


def _summary(graded: list[dict]) -> dict:
    n = len(graded)
    out = {}
    for cond in CONDITIONS:
        cnt = Counter(g[cond] for g in graded)
        by_band = defaultdict(Counter)
        for g in graded:
            by_band[g["band"]][g[cond]] += 1
        out[cond] = {
            "correct": cnt["correct"],
            "misread": cnt["misread"],
            "failed": cnt["failed"],
            "rate": round(cnt["correct"] / n, 4),
            "by_band": {b: f"{c['correct']}/{sum(c.values())}" for b, c in sorted(by_band.items())},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("downloads/benchmark-v1"))
    ap.add_argument("--note", required=True, help="이번 실행에서 무엇이 바뀌었나")
    ap.add_argument("--no-log", action="store_true", help="기록에 남기지 않는다 (시험 실행)")
    args = ap.parse_args()

    lines = (args.data_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:]]  # 첫 줄은 _stats
    run(cv2.imread(str(args.data_root / rows[0]["file"])))  # 모델 로드는 시간에서 뺀다

    graded, total_ms, per_sample = [], [], []
    for row in rows:
        image = cv2.imread(str(args.data_root / row["file"]))
        read = {"A": decode_whole(image)}
        start = time.perf_counter()
        result = run(image)
        total_ms.append((time.perf_counter() - start) * 1000.0)
        read |= ladder(result)
        g = {cond: _grade(read[cond], row["text"]) for cond in CONDITIONS}
        graded.append({"band": row["band"], **g})
        per_sample.append(
            {
                "id": row["id"],
                "band": row["band"],
                "bucket": row["bucket"],
                **g,
                "failure_reason": result.decode.failure_reason,
                "retry_count": result.decode.retry_count,
                "stage_ms": result.decode.stage_ms,
            }
        )

    now = datetime.now().astimezone()
    record = {
        "date": now.isoformat(timespec="seconds"),
        "note": args.note,
        "code": code_commit(),
        "dataset": "benchmark-v1",
        "n": len(rows),
        "decoders": {
            "pyzbar": decoding._pyzbar_decode is not None,
            "zxingcpp": decoding._zxingcpp is not None,
        },
        "conditions": _summary(graded),
        "detected": sum(r["failure_reason"] != "not_detected" for r in per_sample),
        "latency_ms_D": {
            "p50": round(float(np.percentile(total_ms, 50)), 1),
            "p95": round(float(np.percentile(total_ms, 95)), 1),
        },
    }

    out_dir = Path("runs/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    with open(out_dir / f"{stamp}.jsonl", "w", encoding="utf-8") as fh:
        for s in per_sample:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    if not args.no_log:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    for cond in CONDITIONS:
        c = record["conditions"][cond]
        counts = f"정답 {c['correct']:3d}  오독 {c['misread']:2d}  실패 {c['failed']:3d}"
        print(f"{cond}  {counts}  {c['by_band']}")
    print(f"검출 {record['detected']}/{record['n']}  D 지연 {record['latency_ms_D']}")
    print(f"decoders {record['decoders']}")


if __name__ == "__main__":
    main()
