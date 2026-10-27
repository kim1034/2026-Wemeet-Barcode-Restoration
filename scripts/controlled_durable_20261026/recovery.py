"""Recover committed epochs; retain interrupted outputs and never redo completed work."""

import hashlib
import io
import json
import math
import re
import time
from pathlib import Path

import torch

from scripts.controlled_campaign.hardware import utc_now
from scripts.controlled_geometry.common import (
    ROOT,
    digest,
    object_digest,
    read_json,
    selection_key,
    write_json,
)
from scripts.controlled_geometry.evaluation import selection_metrics
from scripts.controlled_resume_20261026.runtime import verify_revision

EPOCH_FOLDER = re.compile(r"epoch-(\d{4})(?:-attempt-\d+)?$")


class IntegrityError(ValueError):
    """Do not change scientific conditions or overwrite a contradictory completed result."""


def verify_durable(campaign):
    manifest = read_json(campaign / "durable/manifest.json")
    if (
        manifest["user_instruction"]
        != "전체 평가와 집계 완료 될 때까지 끝까지 진행되게 해. 중간에 끊기지 않게!"
    ):
        raise IntegrityError("Unexpected continuation authorization")
    if digest(campaign / "revision.json") != manifest["previous_revision_sha256"]:
        raise IntegrityError("Previous execution amendment changed")
    for relative, expected in manifest["sources"].items():
        if digest(ROOT / relative) != expected:
            raise IntegrityError(f"Durable execution source changed: {relative}")
    for relative, expected in manifest["inputs"].items():
        if digest(campaign / relative) != expected:
            raise IntegrityError(f"Durable execution input changed: {relative}")
    verify_revision(campaign)
    return manifest


def ledger_path(campaign, config):
    return campaign / "durable/validation_exclusions" / (config["name"] + ".json")


def exclusions(campaign, config):
    path = ledger_path(campaign, config)
    data = read_json(path) if path.exists() else {"directories": {}}
    output = Path(config["output"])
    for relative, record in data["directories"].items():
        directory = output / relative
        if (
            not directory.is_dir()
            or directory.parent != output / "validation"
            or not EPOCH_FOLDER.fullmatch(directory.name)
        ):
            raise IntegrityError("Excluded validation path escaped its run")
        actual = {
            str(p.relative_to(directory)): digest(p) for p in directory.rglob("*") if p.is_file()
        }
        if actual != record["files"]:
            raise IntegrityError("Preserved uncommitted validation output changed")
    return data


def metric_rows(output, payload=None):
    path = output / "metrics.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    if [r["epoch"] for r in rows] != list(range(1, len(rows) + 1)):
        raise IntegrityError("Missing or duplicated epoch in local metrics")
    if payload is not None:
        epoch = payload["epoch"]
        if len(rows) == epoch - 1 and payload.get("metrics", {}).get("epoch") == epoch:
            rows.append(payload["metrics"])
        if len(rows) != epoch or rows[-1] != payload["metrics"]:
            raise IntegrityError("Checkpoint/metric commit mismatch")
    return rows


def committed_summaries(campaign, config, epoch, metrics):
    output = Path(config["output"])
    excluded = exclusions(campaign, config)["directories"]
    summaries = {}
    by_epoch = {m["epoch"]: m for m in metrics}
    for number in range(5, epoch + 1, 5):
        candidates = [
            p / "summary.json"
            for p in (output / "validation").glob(f"epoch-{number:04d}*")
            if str(p.relative_to(output)) not in excluded and (p / "summary.json").exists()
        ]
        if len(candidates) != 1:
            raise IntegrityError(f"Ambiguous or absent committed validation at epoch {number}")
        summary = read_json(candidates[0])
        if summary["n"] != 2976 or summary["geometry"]["invalid_field_count"]:
            raise IntegrityError("Incomplete/invalid validation population")
        if any(by_epoch[number].get(k) != v for k, v in selection_metrics(summary).items()):
            raise IntegrityError("Validation file does not match the committed epoch metrics")
        summaries[number] = summary
    return summaries


def checkpoint_state(campaign, config):
    output = Path(config["output"])
    raw = (output / "last.pt").read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        payload["format"] != "controlled-v2"
        or payload["config"] != config
        or not 1 <= payload["epoch"] <= config["epochs"]
    ):
        raise IntegrityError("Wrong checkpoint format, configuration or epoch")
    saved = read_json(output / "run.json")
    if (
        saved["config_sha256"] != object_digest(config)
        or read_json(output / "config.json") != config
    ):
        raise IntegrityError("W&B/local configuration changed")
    rows = metric_rows(output, payload)
    if any(r["train/samples"] != 27024 or r["val/samples"] != 2976 for r in rows):
        raise IntegrityError("Training or validation population changed")
    if payload["optimizer_steps"] != payload["epoch"] * math.ceil(27024 / config["batch_size"]):
        raise IntegrityError("Optimizer step count does not match full completed epochs")
    if not payload["optimizer"]["state"] or len(payload["rng"]["cuda"]) != 1:
        raise IntegrityError("Missing optimizer/assigned CUDA RNG state")
    if not all(torch.isfinite(t).all() for t in payload["model"].values()):
        raise IntegrityError("Nonfinite saved model")
    summaries = committed_summaries(campaign, config, payload["epoch"], rows)
    if summaries:
        chosen = min(summaries, key=lambda n: selection_key(summaries[n], n))
        best = payload["best"]
        if (
            not best
            or best["epoch"] != chosen
            or best["summary"] != summaries[chosen]
            or tuple(best["key"]) != selection_key(summaries[chosen], chosen)
        ):
            raise IntegrityError("Saved selection differs from the frozen rule")
        if not all(torch.isfinite(t).all() for t in best["model"].values()):
            raise IntegrityError("Nonfinite selected model")
    elif payload["best"] is not None:
        raise IntegrityError("Selected checkpoint before the first validation")
    return payload, raw


def verify_training(campaign, config):
    output = Path(config["output"])
    payload, _ = checkpoint_state(campaign, config)
    done = read_json(output / "completion.json")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if (
        done["status"] != "trained"
        or payload["epoch"] != config["epochs"]
        or done["selected/checkpoint_sha256"] != digest(output / "best.pt")
        or best["format"] != "controlled-v2-selected"
        or best["config"] != config
        or best["epoch"] != payload["best"]["epoch"]
        or done["selected/epoch"] != best["epoch"]
        or best["selection"] != payload["best"]["summary"]
    ):
        raise IntegrityError("Incomplete or contradictory selected checkpoint")
    if set(best["model"]) != set(payload["best"]["model"]) or any(
        not torch.equal(t, payload["best"]["model"][k]) for k, t in best["model"].items()
    ):
        raise IntegrityError("Best/last selected model differs")
    return done


def active_config_pids(config_path):
    """Read-only orphan check; never signal another process by matching its name."""
    target = str(Path(config_path).resolve())
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().decode().split("\0")
            if "--config" in args and args[args.index("--config") + 1] == target:
                found.append(int(path.parent.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return found


def archive_incomplete(campaign, path, label):
    """Move an incomplete result intact; completed records must be verified, not replaced."""
    path = Path(path)
    if not path.exists():
        return None
    if (path / "completion.json").exists():
        raise IntegrityError("Never archive a result already claiming completion")
    target = campaign / "durable/interrupted" / label / str(time.time_ns())
    target.parent.mkdir(parents=True, exist_ok=True)
    files = {str(p.relative_to(path)): digest(p) for p in path.rglob("*") if p.is_file()}
    path.rename(target)
    write_json(
        target.parent / (target.name + ".json"),
        {
            "at_utc": utc_now(),
            "original_path": str(path),
            "preserved_path": str(target),
            "files": files,
        },
    )
    return target


def prepare_training(campaign, config, config_path):
    if active_config_pids(config_path):
        raise RuntimeError("Wait for the previous process using this config to release")
    output = Path(config["output"])
    if not output.exists():
        return False, 0
    if (output / "completion.json").exists():
        raise IntegrityError("Completed training must be reused, not relaunched")
    if not (output / "last.pt").exists():
        if (output / "metrics.jsonl").exists() and metric_rows(output):
            raise IntegrityError("Training metrics exist without their checkpoint")
        archive_incomplete(campaign, output, config["name"] + "-before-first-checkpoint")
        return False, 0
    payload, raw = checkpoint_state(campaign, config)
    sha = hashlib.sha256(raw).hexdigest()
    backup = campaign / "durable/checkpoints" / (sha + ".pt")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        temporary = backup.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(backup)
    if digest(backup) != sha:
        raise IntegrityError("Recovery checkpoint copy changed")
    ledger = exclusions(campaign, config)
    for directory in (output / "validation").glob("epoch-*"):
        match = EPOCH_FOLDER.fullmatch(directory.name)
        if not match:
            raise IntegrityError("Unrecognized validation directory")
        if int(match[1]) > payload["epoch"]:
            relative = str(directory.relative_to(output))
            record = {
                "checkpoint_epoch": payload["epoch"],
                "checkpoint_sha256": sha,
                "files": {
                    str(p.relative_to(directory)): digest(p)
                    for p in directory.rglob("*")
                    if p.is_file()
                },
            }
            if relative in ledger["directories"] and ledger["directories"][relative] != record:
                if ledger["directories"][relative]["files"] != record["files"]:
                    raise IntegrityError("Uncommitted output changed between recoveries")
            else:
                ledger["directories"][relative] = record
    write_json(ledger_path(campaign, config), ledger)
    entry = {
        "at_utc": utc_now(),
        "config_sha256": object_digest(config),
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_sha256": sha,
        "preserved_checkpoint": str(backup),
        "wandb": read_json(output / "run.json"),
        "validation_exclusions_sha256": digest(ledger_path(campaign, config)),
    }
    write_json(campaign / "durable/resumes" / config["name"] / f"{time.time_ns()}.json", entry)
    return True, payload["epoch"]


def verify_report(campaign):
    """Independent completion admission: all actual outputs, hashes and arithmetic."""
    import csv

    import numpy as np

    from scripts.controlled_campaign.report import mean_std
    from scripts.controlled_campaign.run import (
        validate_authorization,
        verify_evaluation,
        verify_profile,
    )
    from scripts.controlled_geometry.compare import paired_comparison

    auth = validate_authorization(campaign)
    done = read_json(campaign / "report/completion.json")
    if done["status"] != "complete" or done["training_runs"] != 12 or done["evaluations"] != 24:
        raise IntegrityError("Incomplete final report")
    required = {
        "summary.json",
        "results.csv",
        "REPORT_ko.md",
        "preservation.json",
        "authorization.json",
        "operations.json",
    }
    required.update(
        f"comparisons/preset{p}-s{s}-{d}.json"
        for p in ("R", "M")
        for s in (42, 43, 44)
        for d in ("benchmark", "validation")
    )
    if not required <= set(done["files"]):
        raise IntegrityError("Final completion omits required output hashes")
    for name, expected in done["files"].items():
        path = (
            campaign / name
            if name in ("authorization.json", "operations.json")
            else campaign / "report" / name
        )
        if not path.resolve().is_relative_to(campaign.resolve()) or digest(path) != expected:
            raise IntegrityError("Final report file/hash mismatch")
    summary = read_json(campaign / "report/summary.json")
    if (
        summary["status"] != "complete"
        or summary["training_runs"] != 12
        or summary["selected_checkpoint_evaluations"] != 24
        or summary["seed42_latency_profiles"] != 4
        or summary["paired_comparisons"] != 12
        or summary["protocol_sha256"] != auth["protocol_sha256"]
    ):
        raise IntegrityError("Final summary has the wrong scope or protocol")
    csv_rows = list(csv.DictReader((campaign / "report/results.csv").open()))
    if len(summary["rows"]) != 24 or len(csv_rows) != 24 or len(summary["aggregates"]) != 8:
        raise IntegrityError("Final result matrix incomplete")
    expected_rows, inputs, timings = {}, {}, []
    for lane in auth["lanes"]:
        timing = verify_profile(campaign / "latency" / f"gpu{lane['gpu_index']}")
        timings.append(timing)
        for path in lane["configs"]:
            config = read_json(path)
            trained = verify_training(campaign, config)
            if config["seed"] == 42 and (
                timing["config"] != config
                or timing["checkpoint_sha256"] != trained["selected/checkpoint_sha256"]
            ):
                raise IntegrityError("Timing checkpoint mismatch")
            for dataset in ("benchmark", "validation"):
                directory = campaign / "evaluations" / config["name"] / dataset
                evaluated = verify_evaluation(directory, config, dataset)
                result = read_json(directory / "results/summary.json")
                stats = result["paths"]["crop_policy_model"]
                expected_rows[config["model"], config["preset"], config["seed"], dataset] = {
                    "model": config["model"],
                    "preset": config["preset"],
                    "seed": config["seed"],
                    "dataset": dataset,
                    "n": result["n"],
                    "selected_epoch": trained["selected/epoch"],
                    **{k: stats[k] for k in ("C", "W", "U")},
                    **{k: stats[k]["pct"] for k in ("EDR", "FDR", "NRR", "RR_H", "FDR_H", "RR_U")},
                    "epe_mean_px": result["geometry"]["metrics"]["epe_mean_px"],
                    "epe_p95_px": result["geometry"]["metrics"]["epe_p95_px"],
                    "train_run": trained["run_url"],
                    "evaluation_run": evaluated["run_url"],
                }
                inputs[config["model"], config["preset"], config["seed"], dataset] = [
                    json.loads(x)
                    for x in (directory / "results/per_sample.jsonl").read_text().splitlines()
                ]
    if len(expected_rows) != 24 or summary["rows"] != list(expected_rows.values()):
        raise IntegrityError("Final rows do not reproduce source evaluations")
    if summary["timings"] != timings:
        raise IntegrityError("Final latency summaries differ from their source profiles")
    if csv_rows != [{k: str(v) for k, v in row.items()} for row in summary["rows"]]:
        raise IntegrityError("Final CSV differs from JSON")
    for aggregate in summary["aggregates"]:
        rows = [
            r
            for r in summary["rows"]
            if all(r[k] == aggregate[k] for k in ("model", "preset", "dataset"))
        ]
        for key in ("EDR", "FDR", "RR_H", "FDR_H", "RR_U", "epe_mean_px"):
            if aggregate[key] != mean_std([r[key] for r in rows]):
                raise IntegrityError("Three-seed aggregate arithmetic mismatch")
    for preset in ("R", "M"):
        for seed in (42, 43, 44):
            for dataset in ("benchmark", "validation"):
                compared = read_json(
                    campaign / "report/comparisons" / f"preset{preset}-s{seed}-{dataset}.json"
                )
                actual = paired_comparison(
                    *[
                        inputs[m, preset, seed, dataset]
                        for m in ("resnet18-c3", "mobilenetv3-small")
                    ]
                )
                if any(compared[k] != value for k, value in actual.items()):
                    raise IntegrityError("Paired comparison differs from raw results")
    before = read_json(campaign.parent / "preparation/preservation_before.json")
    if any(digest(ROOT / path) != sha for path, sha in before["files"].items()):
        raise IntegrityError("An original experiment file changed")
    if not np.isfinite([r["EDR"] for r in summary["rows"]]).all():
        raise IntegrityError("Nonfinite final accuracy")
    return {
        "status": "passed",
        "training_runs": 12,
        "evaluations": 24,
        "latency_profiles": 4,
        "paired_comparisons": 12,
        "original_files_preserved": len(before["files"]),
        "report_run_url": done["run_url"],
        "report_completion_sha256": digest(campaign / "report/completion.json"),
    }


def wait_seconds(failures):
    return min(600, 30 * 2 ** min(failures, 5))
