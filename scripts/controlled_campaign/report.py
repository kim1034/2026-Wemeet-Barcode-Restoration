"""Verify all completed runs, paired comparisons and the final W&B report."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import wandb

from scripts.controlled_campaign.run import (
    validate_authorization,
    verify_evaluation,
    verify_operations,
    verify_profile,
    verify_training,
)
from scripts.controlled_geometry.common import ROOT, digest, read_json, write_json
from scripts.controlled_geometry.compare import paired_comparison
from scripts.controlled_geometry.tracking import (
    configure_local_directories,
    log_artifact,
    start_run,
)


def mean_std(values):
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("Expected three finite seed results")
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "n": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    verify_operations(campaign)
    auth = validate_authorization(campaign)
    output = campaign / "report"
    output.mkdir(exist_ok=False)
    rows, timings, sources = [], [], {}
    configurations = {}
    for lane in auth["lanes"]:
        timing_dir = campaign / "latency" / f"gpu{lane['gpu_index']}"
        latency = verify_profile(timing_dir)
        if latency["config"] != read_json(lane["configs"][0]):
            raise ValueError("Latency model/config mismatch")
        timings.append(latency)
        for path in lane["configs"]:
            cfg = read_json(path)
            configurations[cfg["model"], cfg["preset"], cfg["seed"]] = cfg
            done = verify_training(cfg)
            if (
                cfg["seed"] == 42
                and latency["checkpoint_sha256"] != done["selected/checkpoint_sha256"]
            ):
                raise ValueError("Latency used a different checkpoint")
            for dataset in ("benchmark", "validation"):
                directory = campaign / "evaluations" / cfg["name"] / dataset
                evaluated = verify_evaluation(directory, cfg, dataset)
                summary = read_json(directory / "results/summary.json")
                if summary["geometry"]["invalid_field_count"]:
                    raise ValueError("Invalid geometry in final results")
                stats = summary["paths"]["crop_policy_model"]
                rows.append(
                    {
                        "model": cfg["model"],
                        "preset": cfg["preset"],
                        "seed": cfg["seed"],
                        "dataset": dataset,
                        "n": summary["n"],
                        "selected_epoch": done["selected/epoch"],
                        **{key: stats[key] for key in ("C", "W", "U")},
                        **{
                            key: stats[key]["pct"]
                            for key in ("EDR", "FDR", "NRR", "RR_H", "FDR_H", "RR_U")
                        },
                        "epe_mean_px": summary["geometry"]["metrics"]["epe_mean_px"],
                        "epe_p95_px": summary["geometry"]["metrics"]["epe_p95_px"],
                        "train_run": done["run_url"],
                        "evaluation_run": evaluated["run_url"],
                    }
                )
                sources[str(directory / "completion.json")] = digest(directory / "completion.json")
    if len(configurations) != 12 or len(rows) != 24:
        raise ValueError("Full experiment matrix not complete")
    comparisons = []
    for preset in ("R", "M"):
        for seed in (42, 43, 44):
            for dataset in ("benchmark", "validation"):
                configs = [
                    configurations[model, preset, seed]
                    for model in ("resnet18-c3", "mobilenetv3-small")
                ]
                samples = []
                for cfg in configs:
                    path = (
                        campaign
                        / "evaluations"
                        / cfg["name"]
                        / dataset
                        / "results/per_sample.jsonl"
                    )
                    samples.append([json.loads(line) for line in path.read_text().splitlines()])
                comparison = {
                    "preset": preset,
                    "seed": seed,
                    "dataset": dataset,
                    "delta_direction": "MobileNet minus ResNet",
                    **paired_comparison(*samples),
                }
                comparisons.append(comparison)
                write_json(
                    output / "comparisons" / f"preset{preset}-s{seed}-{dataset}.json", comparison
                )
    aggregates = []
    for model in ("resnet18-c3", "mobilenetv3-small"):
        for preset in ("R", "M"):
            for dataset in ("benchmark", "validation"):
                selected = [
                    r
                    for r in rows
                    if (r["model"], r["preset"], r["dataset"]) == (model, preset, dataset)
                ]
                aggregates.append(
                    {
                        "model": model,
                        "preset": preset,
                        "dataset": dataset,
                        **{
                            key: mean_std([r[key] for r in selected])
                            for key in ("EDR", "FDR", "RR_H", "FDR_H", "RR_U", "epe_mean_px")
                        },
                    }
                )
    before = read_json(campaign.parent / "preparation/preservation_before.json")
    if any(digest(ROOT / path) != sha for path, sha in before["files"].items()):
        raise ValueError("An original experiment/source file changed")
    write_json(
        output / "preservation.json",
        {
            "status": "passed",
            "files_checked": len(before["files"]),
            "reference_sha256": digest(campaign.parent / "preparation/preservation_before.json"),
        },
    )
    with (output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "complete",
        "protocol_sha256": auth["protocol_sha256"],
        "training_runs": 12,
        "selected_checkpoint_evaluations": 24,
        "seed42_latency_profiles": 4,
        "paired_comparisons": 12,
        "rows": rows,
        "aggregates": aggregates,
        "timings": timings,
        "source_completions": sources,
        "interpretation": [
            "Compare backbones within the same preset; the four preset variables change together.",
            "Historical synthetic development benchmark, not an independent final test.",
            "Payload grouping is a proxy; train/evaluation payload overlap remains disclosed.",
            "MobileNet uses the new shared head; historical checkpoint/source was unavailable.",
            "GPU seed reproducibility is not bitwise guaranteed for adaptive-pooling backward.",
            "Latency excludes Stage1; measured on seed42 selected checkpoints after training.",
        ],
    }
    write_json(output / "summary.json", result)
    lines = [
        "# MobileNet·ResNet 동일 조건 비교 결과",
        "",
        "12개 학습, 24개 선택 체크포인트 평가, 12개 paired 비교 완료.",
        "수치는 3개 seed의 평균 ± 표준편차이며 비율의 단위는 %다.",
        "",
        "| 모델 | 조건 | 데이터 | 정답률 | 오독률 | RR_H | EPE(px) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in aggregates:
        values = [
            f"{r[k]['mean']:.4f} ± {r[k]['std']:.4f}" for k in ("EDR", "FDR", "RR_H", "epe_mean_px")
        ]
        lines.append("| " + " | ".join([r["model"], r["preset"], r["dataset"], *values]) + " |")
    lines += [
        "",
        "R: lr 1e-4 / batch 8 / 50 epoch / beta 0.02.",
        "M: lr 3e-4 / batch 64 / 100 epoch / beta 1/154.",
        "",
        "같은 조건 내에서 모델을 비교한다. 기존 합성 개발셋이며 새 독립 test가 아니다.",
        "MobileNet은 공통 head를 적용한 새 구현이다. GPU 재개의 비트 단위 일치는 보장하지 않는다.",
        "세부 결과와 W&B 링크는 results.csv, paired 구간은 comparisons/에 있다.",
        "시간 측정은 summary.json에 있다.",
    ]
    (output / "REPORT_ko.md").write_text("\n".join(lines) + "\n")
    configure_local_directories()
    with start_run(
        "controlled-v2-final-comparison",
        {
            "protocol_sha256": auth["protocol_sha256"],
            "training_runs": 12,
            "gpu_indices": [4, 5, 6, 7],
        },
        output,
        job_type="report",
    ) as run:
        columns = list(rows[0])
        run.log(
            {
                "comparison/per_seed": wandb.Table(
                    columns=columns, data=[[r[k] for k in columns] for r in rows]
                )
            }
        )
        for r in aggregates:
            for key in ("EDR", "FDR", "RR_H", "FDR_H", "epe_mean_px"):
                run.summary[f"{r['dataset']}/{r['model']}/preset{r['preset']}/{key}_mean"] = r[key][
                    "mean"
                ]
                run.summary[f"{r['dataset']}/{r['model']}/preset{r['preset']}/{key}_std"] = r[key][
                    "std"
                ]
        files = {
            str(p.relative_to(output)): p
            for p in output.rglob("*")
            if p.is_file() and "wandb" not in p.parts
        }
        files.update(
            {
                "authorization.json": campaign / "authorization.json",
                "operations.json": campaign / "operations.json",
            }
        )
        log_artifact(
            run,
            "controlled-v2-final-comparison",
            "report",
            files,
            metadata={"protocol_sha256": auth["protocol_sha256"]},
        )
        run_url = run.url
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "run_url": run_url,
            "files": {name: digest(path) for name, path in files.items()},
            "training_runs": 12,
            "evaluations": 24,
        },
    )
    print(json.dumps({"status": "complete", "run_url": run_url}), flush=True)


if __name__ == "__main__":
    main()
