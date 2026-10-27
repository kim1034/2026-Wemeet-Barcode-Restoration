"""Independent raw-result checks and portable combined experiment reports."""

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import wandb

from scripts.controlled_geometry.common import digest, read_json, write_json
from scripts.mobilenet_followup.controller import audit_job, task_matrix
from scripts.mobilenet_followup.core import POLICIES, choose

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/experiments/2026-10-27-mobilenet-followup"
LABELS = {
    "early_exit": "현재 조기 종료",
    "always_rectify": "항상 보정",
    "agreement": "판독 일치 확인",
}
PRIOR_RUN = "https://wandb.ai/kim-1034/wemeet-barcode/runs/n13l9dno"


def grade(value, truth):
    if value["text"] is None:
        return "U"
    return "C" if value["text"] == truth and value["format"] == "Code128" else "W"


def read_records(directory):
    return [json.loads(x) for x in (directory / "per_sample.jsonl").read_text().splitlines()]


def verify_accuracy(records, summary, task, inputs):
    if task["kind"] == "scene":
        expected = {
            str(r["id"]): r["text"]
            for r in (
                json.loads(x)
                for x in (Path(inputs["scene_root"]) / "manifest.jsonl")
                .read_text()
                .splitlines()[1:]
            )
        }
    else:
        directory = (
            Path(inputs["prior_campaign"])
            / "evaluations/restoration-mobilenetv3-small-presetM-s42-v2"
            / task["dataset"]
            / "results"
        )
        previous = read_records(directory)
        expected = {str(r["id"]): r["gt_text"] for r in previous}
    assert {str(r["id"]) for r in records} == set(expected)
    for r in records:
        if expected[str(r["id"])] is not None:
            assert r["gt_text"] == expected[str(r["id"])]
        assert r["statuses"]["d0"] == grade(r["d0"], r["gt_text"])
        if r.get("detected", True):
            for policy in POLICIES:
                actual = choose(policy, r["d0"], r["warped"])
                assert actual == r["decisions"][policy]
        else:
            assert all(v["text"] is None for v in r["decisions"].values())
        if task["kind"] == "scene":
            assert r["statuses"]["A"] == grade(r["whole_scene"], r["gt_text"])
            assert r["statuses"]["B"] == r["statuses"]["d0"]
            c = r["d0"] if r["d0"]["text"] is not None or not r["warped"] else r["warped"][0]
            assert r["statuses"]["C"] == grade(c, r["gt_text"])
    for path, metrics in summary["paths"].items():
        statuses = [r["statuses"][path] for r in records]
        for metric, label in [("EDR", "C"), ("FDR", "W"), ("NRR", "U")]:
            assert math.isclose(metrics[metric], 100 * statuses.count(label) / len(statuses))
        hard = [r for r in records if r["statuses"]["d0"] != "C"]
        assert metrics["H"] == len(hard)
        if hard:
            assert math.isclose(
                metrics["RR_H"], 100 * sum(r["statuses"][path] == "C" for r in hard) / len(hard)
            )
        for a in ("C", "W", "U"):
            for b in ("C", "W", "U"):
                assert metrics["transitions"][a + "->" + b] == sum(
                    r["statuses"]["d0"] == a and r["statuses"][path] == b for r in records
                )


def verify_timing(summary):
    for metrics in summary["paths"].values():
        for p in metrics["passes"]:
            assert all(math.isfinite(v) and v > 0 for v in p["raw_ms"])
            for name, quantile in [("p50_ms", 0.5), ("p95_ms", 0.95)]:
                assert math.isclose(p[name], float(np.quantile(p["raw_ms"], quantile)))
        for name in ("p50_ms", "p95_ms"):
            assert math.isclose(
                metrics[name], float(np.median([p[name] for p in metrics["passes"]]))
            )


def paired(left, right, left_policy, right_policy):
    """Payload-cluster paired bootstrap, with images retained within each cluster."""
    rhs = {r["id"]: r for r in right}
    assert {r["id"] for r in left} == set(rhs)
    groups = defaultdict(lambda: [0, 0, 0])
    wins = losses = 0
    for a in left:
        b = rhs[a["id"]]
        assert a["gt_text"] == b["gt_text"]
        aa, bb = a["statuses"][left_policy], b["statuses"][right_policy]
        entry = groups[a["gt_text"]]
        entry[0] += 1
        entry[1] += int(bb == "C") - int(aa == "C")
        entry[2] += int(bb == "W") - int(aa == "W")
        wins += bb == "C" and aa != "C"
        losses += aa == "C" and bb != "C"
    values = np.array(list(groups.values()), dtype=float)
    rng = np.random.default_rng(20261027)
    draws = []
    for _ in range(20):
        index = rng.integers(0, len(values), size=(100, len(values)))
        sums = values[index].sum(axis=1)
        draws.append(100 * sums[:, 1:] / sums[:, :1])
    distribution = np.concatenate(draws)
    point = 100 * values[:, 1:].sum(axis=0) / values[:, 0].sum()
    return {
        "n": len(left),
        "payload_clusters": len(values),
        "bootstrap_repeats": 2000,
        "correct_gains": int(wins),
        "correct_losses": int(losses),
        "EDR_delta_pp": float(point[0]),
        "EDR_ci95_pp": np.quantile(distribution[:, 0], [0.025, 0.975]).tolist(),
        "FDR_delta_pp": float(point[1]),
        "FDR_ci95_pp": np.quantile(distribution[:, 1], [0.025, 0.975]).tolist(),
    }


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def metric_cell(metrics):
    return f"{metrics['EDR']:.2f} / {metrics['FDR']:.2f} / {metrics['NRR']:.2f}"


def failure_rows(task, records):
    result = []
    for r in records:
        for policy in POLICIES:
            state = r["statuses"][policy]
            if state == "C":
                continue
            decision = r["decisions"][policy]
            if not r.get("detected", True):
                reason = "detector_no_box"
            elif state == "U":
                reason = decision.get("reason", "decode_no_return")
            elif r["d0"]["text"] == decision["text"] and r["statuses"]["d0"] == "W":
                reason = "same_wrong_as_d0"
            else:
                reason = "wrong_after_rectification"
            result.append(
                {
                    "task": task["id"],
                    "sample_id": r["id"],
                    "policy": policy,
                    "status": state,
                    "reason": reason,
                    "gt_text": r["gt_text"],
                    "predicted_text": decision["text"],
                    "format": decision["format"],
                    "d0_status": r["statuses"]["d0"],
                    "band": r.get("band", "unknown"),
                    "bucket": r.get("bucket", "unknown"),
                    "aspect_family": r.get("aspect_family", "unknown"),
                    "tilt_band": r.get("tilt_band", "unknown"),
                }
            )
    return result


def finalize(root):
    root = Path(root).resolve()
    if (root / "completion.json").exists():
        done = read_json(root / "completion.json")
        for filename, expected in done["files"].items():
            assert digest(root / "report" / filename) == expected
        return done
    inputs = read_json(root / "inputs.json")
    for filename, expected in read_json(root / "execution_lock.json").items():
        assert digest(filename) == expected, filename
    preserved = read_json(root / "preservation_before.json")
    for filename, expected in preserved.items():
        assert digest(filename) == expected, filename
    for filename, expected in inputs["scene_files"].items():
        assert digest(filename) == expected, filename
    tasks = task_matrix()
    assert read_json(root / "tasks.json") == tasks
    accuracy, timing, sources, records_by_id = {}, {}, {}, {}
    rows, failures = [], []
    for task in tasks:
        done = audit_job(root, task)
        assert done is not None, task["id"]
        directory = Path(done["output"])
        summary = read_json(directory / "summary.json")
        sources[task["id"]] = {
            "wandb": done["wandb"],
            "completion_sha256": digest(directory.parent / "completion.json"),
            "files": done["files"],
            "gpu": read_json(directory / "task.json")["gpu"],
        }
        if "timing" in task["kind"]:
            verify_timing(summary)
            timing[task["id"]] = {
                p: {k: v for k, v in m.items() if k != "passes"}
                for p, m in summary["paths"].items()
            }
        else:
            records = read_records(directory)
            verify_accuracy(records, summary, task, inputs)
            accuracy[task["id"]] = summary
            records_by_id[task["id"]] = records
            failures.extend(failure_rows(task, records))
            for p, m in summary["paths"].items():
                rows.append(
                    {
                        "task": task["id"],
                        "model": task["model"],
                        "dataset": task["dataset"],
                        "decoder_clahe": task["enhance"],
                        "detector_imgsz": task.get("imgsz", ""),
                        "path": p,
                        **{k: m[k] for k in ("n", "C", "W", "U", "EDR", "FDR", "NRR", "RR_H")},
                        "wandb": done["wandb"],
                    }
                )
    comparisons = {}

    def compare(a, b, pa, pb):
        comparisons[f"{a}/{pa} -> {b}/{pb}"] = paired(records_by_id[a], records_by_id[b], pa, pb)

    for task in tasks:
        if "timing" in task["kind"]:
            continue
        for policy in ("always_rectify", "agreement"):
            compare(task["id"], task["id"], "early_exit", policy)
        if task["enhance"]:
            for policy in POLICIES:
                compare(task["id"].replace("clahe", "legacy"), task["id"], policy, policy)
    for mode in ("legacy", "clahe"):
        for policy in POLICIES:
            compare(
                f"scene-resnet18-c3-1280-{mode}",
                f"scene-mobilenetv3-small-1280-{mode}",
                policy,
                policy,
            )
            for size in (1600, 1920):
                compare(
                    f"scene-mobilenetv3-small-1280-{mode}",
                    f"scene-mobilenetv3-small-{size}-{mode}",
                    policy,
                    policy,
                )
    report = root / "report"
    report.mkdir(exist_ok=True)
    csv_file(report / "results.csv", rows)
    csv_file(report / "failure_cases.csv", failures)
    csv_file(
        report / "timing.csv",
        [
            {"task": task, "policy": p, "p50_ms": m["p50_ms"], "p95_ms": m["p95_ms"]}
            for task, paths in timing.items()
            for p, m in paths.items()
        ],
    )
    csv_file(
        report / "failure_summary.csv",
        [
            {"task": a, "policy": b, "status": c, "reason": d, "count": n}
            for (a, b, c, d), n in sorted(
                Counter(
                    (r["task"], r["policy"], r["status"], r["reason"]) for r in failures
                ).items()
            )
        ],
    )
    write_json(report / "paired_comparisons.json", comparisons)
    write_json(
        report / "summary.json", {"accuracy": accuracy, "timing": timing, "sources": sources}
    )
    portable = {
        k: v
        for k, v in inputs.items()
        if k
        not in (
            "scene_files",
            "prior_campaign",
            "prior_protocol",
            "crop_data_root",
            "split_manifest",
            "cache_root",
            "scene_root",
            "checkpoints",
            "detector",
        )
    }
    portable["checkpoints"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "path"}
        for k, v in inputs["checkpoints"].items()
    }
    portable["detector"] = {k: v for k, v in inputs["detector"].items() if k != "path"}
    write_json(report / "provenance.json", portable)
    write_json(
        report / "validation.json",
        {
            "status": "passed",
            "tasks": len(tasks),
            "accuracy_tasks": len(accuracy),
            "timing_tasks": len(timing),
            "preserved_files": len(preserved),
            "scene_files": len(inputs["scene_files"]),
            "raw_decisions_rates_transitions_quantiles_checked": True,
            "inputs_sha256": digest(root / "inputs.json"),
            "execution_lock_sha256": digest(root / "execution_lock.json"),
            "sources": sources,
        },
    )
    old_report = Path(inputs["prior_campaign"]) / "report"
    old_rows = list(csv.DictReader((old_report / "results.csv").open()))
    old_table = []
    for model in ("resnet18-c3", "mobilenetv3-small"):
        for preset in ("R", "M"):
            row = [model, preset]
            for dataset in ("validation", "benchmark"):
                subset = [
                    r
                    for r in old_rows
                    if r["model"] == model and r["preset"] == preset and r["dataset"] == dataset
                ]
                assert {int(r["seed"]) for r in subset} == {42, 43, 44}
                for metric in ("EDR", "FDR"):
                    values = [float(r[metric]) for r in subset]
                    row.append(f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}")
            old_table.append(row)
    text = [
        "# MobileNet 선정 및 후속 실험 통합 결과",
        "",
        ("기존 네 조건의 3-seed 비교와, 선택한 MobileNet M/seed42의 후속 평가를 함께 정리했다."),
        (
            "추가 학습 없이 GPU 6·7(A100 40GB)을 사용했다. 후속 결과는 "
            "단일 체크포인트의 개발셋 평가다."
        ),
        "",
        "## 1. 기존 네 조건 비교",
        "",
        "R: lr 1e-4, batch 8, 50 epoch, β=0.02. M: lr 3e-4, batch 64, 100 epoch, β=1/154.",
        (
            "동일한 공통 head·데이터 분할로 각 모델에 R/M을 적용했다. 아래는 "
            "seed42/43/44 평균 ± 표준편차(%)."
        ),
        "",
        table(
            ["모델", "조건", "검증 정답률", "검증 오독률", "benchmark 정답률", "benchmark 오독률"],
            old_table,
        ),
        "",
        ("MobileNet M을 채택하고 검증 RR_H 기준으로 seed42/epoch90을 선택했다(100 epoch 학습)."),
        (
            "이전 ResNet 채택 당시의 서로 다른 데이터·지표로 나온 수치는 이번 "
            "공통 조건 비교와 직접 비교할 수 없다."
        ),
        ("MobileNet의 평균 정답률 우위와 모든 조건에서의 오독률 우위는 별개의 주장이다."),
        "",
        (
            f"[기존 W&B 통합 결과]({PRIOR_RUN}) · "
            "[기존 상세 보고서](../2026-10-26-stage2-controlled-v2/REPORT_ko.md)"
        ),
        "",
        "## 2. 후속 실험 조건",
        "",
        ("1. 정책: 현재 조기 종료 / 항상 보정 / 판독 일치 확인을 동일 입력에서 비교했다."),
        ("2. Full pipeline: 탐지→crop→D0→geometry→TPS→1/1.5/3배 재시도를 평가했다."),
        ("3. 탐지 입력 해상도: MobileNet 1280/1600/1920, ResNet M 대조군 1280을 사용했다."),
        ("4. 오독·미검출·무응답을 나누고, 난도·형태·기울기별 결과와 실제 실패 샘플을 저장했다."),
        "",
        ("일치 확인은 D0와 세 warp에서 서로 다른 이미지 2개 이상이 같은 text/format을 반환하고,"),
        (
            "다른 유효 판독이 없을 때만 채택한다. GT는 판독 결정에 쓰지 않는다. "
            "동일한 오독이 반복되면 통과할 수 있다."
        ),
        "항상 보정은 D0를 채택하지 않고, 첫 유효 warp 판독을 사용한다.",
        "",
        (
            "최신 GitHub 디코더의 CLAHE 재시도(clip=4, grid=4×4) "
            "효과를 분리하기 위해 off/on을 각각 평가했다."
        ),
        (
            "zxing-cpp 단일 백엔드, FP32, TF32 off, CPU "
            "thread 1. 탐지 전 CLAHE는 양쪽 모두 동일하게 적용했다."
        ),
        "",
        "## 3. MobileNet crop 판독 정책",
        "",
        ("각 셀은 정답률 / 오독률 / 무응답률(%). C/W/U는 서로 배타적이며 분모는 전체 입력 수다."),
        "",
        table(
            ["데이터", "디코더 CLAHE", *LABELS.values()],
            [
                [
                    dataset + f" (n={accuracy[key]['n']})",
                    mode,
                    *[metric_cell(accuracy[key]["paths"][p]) for p in POLICIES],
                ]
                for dataset in ("validation", "benchmark")
                for mode in ("legacy", "clahe")
                for key in [f"crop-{dataset}-{mode}"]
            ],
        ),
        "",
        "legacy는 기존 디코더 조건(off), clahe는 최신 디코더 조건(on)이다.",
        ("기존 조건의 4,476개 입력 모두에서 원래 예측과 판독 결과를 재현 검사했다."),
        "",
        "## 4. 전체 파이프라인 및 탐지 해상도",
        "",
        "benchmark/v2의 합성 장면 100개. 각 셀은 정답률 / 오독률 / 무응답률(%).",
        "",
        table(
            ["모델", "imgsz", "디코더 CLAHE", "탐지 수", *LABELS.values()],
            [
                [
                    task["model"],
                    task["imgsz"],
                    task["enhance"],
                    sum(r["detected"] for r in records_by_id[task["id"]]),
                    *[metric_cell(accuracy[task["id"]]["paths"][p]) for p in POLICIES],
                ]
                for task in tasks
                if task["kind"] == "scene"
            ],
        ),
        "",
        ("탐지 수는 박스 반환 횟수이며 mAP가 아니다. A=전체 이미지 직접 판독, B=탐지 crop D0,"),
        "C=B에서 판독 실패 시 1배 warp, D=현재 조기 종료의 3개 scale 재시도다.",
        "A/B/C/D의 상세 수치는 results.csv에 포함했다.",
        "",
        "## 5. 실제 실행 시간",
        "",
        (
            "모든 정확도 평가가 끝난 뒤 시간 측정을 직렬로 실행했다. 50회 "
            "warmup 후 전체 입력을 3회 반복했다."
        ),
        ("셀은 반복별 p50/p95의 중앙값(ms)이다. crop은 탐지 제외, scene은 탐지를 포함한다."),
        (
            "모델·입력을 메모리에 올린 뒤 측정하므로 시작 시 로딩, 파일 I/O, "
            "네트워크 시간은 제외한다."
        ),
        "",
        table(
            ["작업", *LABELS.values()],
            [
                [
                    task.replace("-timing", ""),
                    *[f"{paths[p]['p50_ms']:.2f} / {paths[p]['p95_ms']:.2f}" for p in POLICIES],
                ]
                for task, paths in timing.items()
            ],
        ),
        "",
        "## 6. 실패 분석 및 해석 범위",
        "",
        (
            "failure_cases.csv에는 모든 오독·무응답의 ID, GT, "
            "판독값, D0 상태, 난도와 형태를 기록했다."
        ),
        ("failure_summary.csv는 원인 후보별 건수다. 판독 불일치/부족과 미검출을 별도로 구분한다."),
        ("보정 이후 오독은 관찰된 위치를 뜻하며, TPS나 모델이 원인이라고 단정하는 진단은 아니다."),
        (
            "원시 추론·warp 판독, 난도별 집계, 좌표 오차, D0→최종 전이는 "
            "W&B artifact 및 summary.json에 있다."
        ),
        "",
        ("paired_comparisons.json에는 동일 입력 간 정답 증가/감소와 오독률 차이를 넣었다."),
        (
            "95% 구간은 payload 텍스트를 묶은 paired bootstrap "
            "2,000회이며, 다중 비교 보정은 하지 않았다."
        ),
        ("텍스트는 실제 원본 장면 ID의 대용치이므로 독립성을 완전히 보장하지 않는다."),
        "",
        (
            "검증 2,976개는 체크포인트 선택에 사용했고 crop benchmark "
            "1,500개와 장면 100개는 합성 개발셋이다."
        ),
        ("학습과 payload가 겹칠 수 있어 독립 test/현장 일반화 성능으로 제시하지 않는다."),
        ("이 실험에서 정책을 선택한 뒤에는 별도의 실물 바코드 holdout에서 최종 확인해야 한다."),
        "",
        "## 7. 재현과 공유",
        "",
        f"기반 GitHub commit: `{inputs['upstream_commit']}`.",
        f"장면 dataset revision: `{inputs['scene_dataset_revision']}`.",
        f"MobileNet checkpoint SHA256: `{inputs['checkpoints']['mobilenetv3-small']['sha256']}`.",
        ("패키지·가중치·GPU 식별 정보는 provenance.json, 검증 결과는 validation.json에 있다."),
        (
            "GitHub에는 코드·문서·집계를 공유하고, 데이터/가중치/전체 trace는 "
            "W&B artifact와 고정 HF revision으로 연결한다."
        ),
        ("기존 캠페인과 결과 파일은 수정하지 않았다. README.md에 실행 방법을 정리했다."),
        "",
        table(
            ["실험", "W&B"], [[key, f"[run]({value['wandb']})"] for key, value in sources.items()]
        ),
        "",
    ]
    (report / "REPORT_ko.md").write_text("\n".join(text))
    (report / "NOTION_ko.md").write_text("\n".join(text))
    run = wandb.init(
        entity="kim-1034",
        project="wemeet-barcode",
        group="mobilenet-followup-20261027",
        name="combined-four-way-and-mobilenet-followup-report",
        job_type="report",
        mode="online",
        dir=str(root),
        config=portable,
        settings=wandb.Settings(init_timeout=60),
    )
    run.summary.update(
        {
            "accuracy_tasks": len(accuracy),
            "timing_tasks": len(timing),
            "validated": True,
            "prior_report": PRIOR_RUN,
            "preserved_files": len(preserved),
        }
    )
    for name in ("REPORT_ko.md", "NOTION_ko.md"):
        with (report / name).open("a") as stream:
            stream.write(f"\n[이번 통합 W&B 보고서]({run.url})\n")
    artifact = wandb.Artifact("mobilenet-followup-combined-report", type="report")
    for path in report.iterdir():
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
    artifact.add_file(str(old_report / "results.csv"), name="prior-four-way-results.csv")
    artifact.add_file(str(old_report / "REPORT_ko.md"), name="prior-four-way-REPORT_ko.md")
    run.log_artifact(artifact).wait()
    url = run.url
    run.finish()
    DOCS.mkdir(parents=True, exist_ok=True)
    for path in report.iterdir():
        if path.is_file():
            shutil.copy2(path, DOCS / path.name)
    done = {
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "accuracy_tasks": len(accuracy),
        "timing_tasks": len(timing),
        "wandb": url,
        "files": {p.name: digest(p) for p in report.iterdir() if p.is_file()},
    }
    write_json(root / "completion.json", done)
    return done


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    finalize(parser.parse_args().root)
