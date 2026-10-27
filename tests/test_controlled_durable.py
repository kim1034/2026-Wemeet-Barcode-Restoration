"""Fault cases for persisted epochs, interrupted stages and service supervision."""

import copy
import json
import threading
from pathlib import Path

import pytest
import torch
import wandb

from scripts.controlled_durable_20261026 import recovery, run, watch, worker
from scripts.controlled_geometry.common import (
    digest,
    object_digest,
    selection_key,
    write_json,
)
from scripts.controlled_geometry.evaluation import selection_metrics


def training_fixture(tmp_path, epoch=6):
    campaign = tmp_path / "campaign"
    output = tmp_path / "training"
    config = {"name": "example", "output": str(output), "seed": 42, "epochs": 10, "batch_size": 8}
    summary = {
        "n": 2976,
        "geometry": {"invalid_field_count": 0, "metrics": {"epe_mean_px": 2.0}},
        "paths": {
            p: {"RR_H": {"pct": 10.0}, "FDR_H": {"pct": 1.0}}
            for p in ("crop_forced_model", "crop_policy_model")
        },
    }
    rows = [
        {
            "epoch": i,
            "train/samples": 27024,
            "val/samples": 2976,
            **(selection_metrics(summary) if i % 5 == 0 else {}),
        }
        for i in range(1, epoch + 1)
    ]
    output.mkdir(parents=True)
    for number in range(5, epoch + 1, 5):
        write_json(output / f"validation/epoch-{number:04d}/summary.json", summary)
    model = {"weight": torch.ones(2)}
    payload = {
        "format": "controlled-v2",
        "config": config,
        "epoch": epoch,
        "model": model,
        "optimizer": {"state": {0: {"step": torch.tensor(3378 * epoch)}}},
        "optimizer_steps": 3378 * epoch,
        "rng": {"cuda": [torch.zeros(1, dtype=torch.uint8)]},
        "best": {"epoch": 5, "summary": summary, "key": selection_key(summary, 5), "model": model},
        "metrics": rows[-1],
    }
    torch.save(payload, output / "last.pt")
    write_json(output / "config.json", config)
    write_json(output / "run.json", {"id": "same-wandb-id", "config_sha256": object_digest(config)})
    (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return campaign, config, payload, rows, summary


@pytest.mark.parametrize(
    "fault",
    [None, "epoch", "config", "rng", "optimizer", "steps", "nonfinite", "metrics", "selection"],
)
def test_checkpoint_resume_accepts_only_committed_scientific_state(tmp_path, fault):
    campaign, config, payload, rows, _ = training_fixture(tmp_path)
    if fault == "epoch":
        payload["epoch"] = 11
    elif fault == "config":
        payload["config"] = {**config, "batch_size": 64}
    elif fault == "rng":
        payload["rng"]["cuda"] = []
    elif fault == "optimizer":
        payload["optimizer"]["state"] = {}
    elif fault == "steps":
        payload["optimizer_steps"] -= 1
    elif fault == "nonfinite":
        payload["model"] = {"weight": torch.tensor([float("nan")])}
    elif fault == "metrics":
        (Path(config["output"]) / "metrics.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows[1:])
        )
    elif fault == "selection":
        payload["best"]["key"] = (0, 0, 0, 5)
    torch.save(payload, Path(config["output"]) / "last.pt")
    if fault:
        with pytest.raises(recovery.IntegrityError):
            recovery.checkpoint_state(campaign, config)
    else:
        saved, _ = recovery.checkpoint_state(campaign, config)
        assert saved["epoch"] == 6


def test_checkpoint_saved_before_metric_log_is_recoverable(tmp_path):
    campaign, config, payload, rows, _ = training_fixture(tmp_path)
    path = Path(config["output"]) / "metrics.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    saved, _ = recovery.checkpoint_state(campaign, config)
    assert saved["metrics"] == payload["metrics"]
    assert len(recovery.metric_rows(Path(config["output"]), saved)) == 6
    assert len(path.read_text().splitlines()) == 5  # admission is read-only


def test_partial_validation_is_preserved_and_excluded_before_resume(tmp_path, monkeypatch):
    campaign, config, payload, rows, summary = training_fixture(tmp_path)
    output = Path(config["output"])
    interrupted = output / "validation/epoch-0010"
    write_json(interrupted / "summary.json", summary)
    original_hash = digest(interrupted / "summary.json")
    monkeypatch.setattr(recovery, "active_config_pids", lambda _: [])
    assert recovery.prepare_training(campaign, config, "fixture") == (True, 6)
    assert digest(interrupted / "summary.json") == original_hash
    assert "validation/epoch-0010" in recovery.exclusions(campaign, config)["directories"]
    assert list((campaign / "durable/checkpoints").glob("*.pt"))
    successful = output / "validation/epoch-0010-attempt-02"
    write_json(successful / "summary.json", summary)
    later = rows + [
        {
            "epoch": i,
            "train/samples": 27024,
            "val/samples": 2976,
            **(selection_metrics(summary) if i == 10 else {}),
        }
        for i in range(7, 11)
    ]
    selected = recovery.committed_summaries(campaign, config, 10, later)
    assert selected[10] == summary
    (interrupted / "summary.json").write_text("changed")
    with pytest.raises(recovery.IntegrityError, match="Preserved"):
        recovery.exclusions(campaign, config)


def test_two_unexplained_complete_validation_attempts_are_not_cherry_picked(tmp_path):
    campaign, config, payload, rows, summary = training_fixture(tmp_path)
    write_json(Path(config["output"]) / "validation/epoch-0005-attempt-02/summary.json", summary)
    with pytest.raises(recovery.IntegrityError, match="Ambiguous"):
        recovery.checkpoint_state(campaign, config)


def test_no_checkpoint_start_preserves_the_unstarted_run(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    output = tmp_path / "training"
    output.mkdir()
    (output / "startup.log").write_text("network failed before training")
    config = {"name": "new-run", "output": str(output)}
    monkeypatch.setattr(recovery, "active_config_pids", lambda _: [])
    assert recovery.prepare_training(campaign, config, "fixture") == (False, 0)
    assert not output.exists()
    logs = list((campaign / "durable/interrupted").rglob("startup.log"))
    assert len(logs) == 1 and logs[0].read_text() == "network failed before training"


def test_live_previous_process_blocks_checkpoint_recovery(tmp_path, monkeypatch):
    campaign, config, *_ = training_fixture(tmp_path)
    monkeypatch.setattr(recovery, "active_config_pids", lambda _: [123])
    with pytest.raises(RuntimeError, match="release"):
        recovery.prepare_training(campaign, config, "fixture")
    assert not (campaign / "durable/checkpoints").exists()


def test_claimed_completion_is_never_archived(tmp_path):
    output = tmp_path / "results"
    write_json(output / "completion.json", {"status": "evaluated"})
    with pytest.raises(recovery.IntegrityError):
        recovery.archive_incomplete(tmp_path, output, "evaluation")
    assert (output / "completion.json").exists()


@pytest.mark.parametrize("fault", [None, "early", "selected_model", "wrong_hash"])
def test_training_closeout_revalidates_selection_and_full_epochs(tmp_path, fault):
    campaign, config, payload, _, summary = training_fixture(tmp_path, epoch=10)
    output = Path(config["output"])
    selected = {
        "format": "controlled-v2-selected",
        "config": config,
        "epoch": 5,
        "selection": summary,
        "model": copy.deepcopy(payload["best"]["model"]),
    }
    if fault == "early":
        payload["epoch"] = 9
        torch.save(payload, output / "last.pt")
    if fault == "selected_model":
        selected["model"]["weight"].zero_()
    torch.save(selected, output / "best.pt")
    write_json(
        output / "completion.json",
        {
            "status": "trained",
            "selected/epoch": 5,
            "selected/checkpoint_sha256": "wrong"
            if fault == "wrong_hash"
            else digest(output / "best.pt"),
        },
    )
    if fault:
        with pytest.raises(recovery.IntegrityError):
            recovery.verify_training(campaign, config)
    else:
        assert recovery.verify_training(campaign, config)["selected/epoch"] == 5


def test_completed_training_and_evaluation_are_reused(tmp_path, monkeypatch):
    campaign = object.__new__(run.DurableCampaign)
    campaign.directory = tmp_path
    campaign.command = lambda *a: pytest.fail("completed work must not launch a process")
    config = {"name": "already-complete", "output": str(tmp_path / "training")}
    write_json(Path(config["output"]) / "completion.json", {"status": "trained"})
    write_json(
        tmp_path / "evaluations/already-complete/benchmark/completion.json", {"status": "evaluated"}
    )
    monkeypatch.setattr(run, "verify_training", lambda *a: "verified-training")
    monkeypatch.setattr(run, "verify_evaluation", lambda *a: "verified-evaluation")
    assert campaign.train({}, "config", config) == "verified-training"
    assert campaign.evaluate({}, config, "benchmark") == "verified-evaluation"


def test_failed_stage_waits_and_continues_without_ending_the_lane(monkeypatch):
    campaign = object.__new__(run.DurableCampaign)
    campaign.mutex = threading.RLock()
    campaign.state = {"recovery_events": 0}
    events, delays, attempts = [], [], []
    campaign.event = events.append
    campaign.publish = lambda: None
    monkeypatch.setattr(run.time, "sleep", delays.append)

    def action():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return "finished"

    assert campaign.retry("evaluation", action) == "finished"
    assert len(events) == 2 and delays == [60, 120]
    assert campaign.state["recovery_events"] == 2


def test_wandb_publication_retry_preserves_the_same_operation(monkeypatch):
    attempts, delays = [], []
    monkeypatch.setattr(worker.time, "sleep", delays.append)

    def upload():
        attempts.append(1)
        if len(attempts) == 1:
            raise wandb.errors.CommError("network interruption")
        return "artifact-uploaded"

    assert worker.network_retry(upload) == "artifact-uploaded"
    assert delays == [60]
    with pytest.raises(ValueError):
        worker.network_retry(lambda: (_ for _ in ()).throw(ValueError("wrong result")))


@pytest.mark.parametrize(
    "active,pid,report,held,expected",
    [
        ("active", 1, False, False, "monitor"),
        ("active", 1, True, False, "monitor"),
        ("active", 1, False, True, "recover_failed_lane"),
        ("activating", 0, False, False, "monitor"),
        ("deactivating", 1, False, False, "monitor"),
        ("failed", 0, False, False, "start_continuation"),
        ("inactive", 0, False, False, "start_continuation"),
        ("inactive", 0, True, False, "verify_completion"),
    ],
)
def test_supervisor_acts_only_after_real_service_state(active, pid, report, held, expected):
    progress = {"lanes": {"4": {"stage": "held_error" if held else "training"}}}
    assert (
        watch.decision({"ActiveState": active, "MainPID": str(pid)}, report, progress) == expected
    )
