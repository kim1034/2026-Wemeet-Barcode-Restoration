"""Regression coverage for real NumPy boolean output and bounded checkpoint recovery."""

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.controlled_geometry import evaluation
from scripts.controlled_geometry.common import digest, read_json, write_json
from scripts.controlled_resume_20261026 import runtime
from wemeet.ai.geometry.model import destination_grid


@pytest.mark.parametrize("valid", [True, False])
def test_real_evaluation_serializes_boolean_without_changing_results(tmp_path, monkeypatch, valid):
    grid = destination_grid().numpy()
    prediction = grid.copy() if valid else grid.copy() + 2
    monkeypatch.setattr(evaluation, "predict", lambda *a: prediction)
    wrong = {"text": "wrong", "format": "Code128", "backend": "fixture"}
    monkeypatch.setattr(evaluation, "observe_decode", lambda *a: (None, wrong))
    monkeypatch.setattr(
        evaluation,
        "crop_attempts",
        lambda *a: {
            "final": {"text": "right", "format": "Code128"},
            "attempts": [],
        },
    )
    samples = [
        ("one", {"text": "right", "bucket": "low"}, np.zeros((20, 40), np.uint8), grid, grid)
    ]
    summary, records = evaluation.evaluate_samples(nn.Identity(), samples)
    assert isinstance(records[0]["geometry_valid"], np.bool_)
    assert bool(records[0]["geometry_valid"]) is valid
    with pytest.raises(TypeError, match="not JSON serializable"):
        runtime.ORIGINAL_SAVE_RESULTS(tmp_path / "failed", records, summary)
    assert (tmp_path / "failed/per_sample.jsonl").read_bytes() == b""
    original = copy.deepcopy(records)
    monkeypatch.setattr(evaluation, "save_results", runtime.save_results)
    after_summary, after = evaluation.evaluate_samples(
        nn.Identity(), samples, output=tmp_path / "fixed"
    )
    assert after_summary == summary
    assert after == original
    stored = [json.loads(x) for x in (tmp_path / "fixed/per_sample.jsonl").read_text().splitlines()]
    assert stored == after
    assert type(stored[0]["geometry_valid"]) is bool
    assert evaluation.summarize(stored) == summary
    assert after[0]["paths"]["crop_policy_model"]["status"] == "W"
    assert (tmp_path / "failed/per_sample.jsonl").read_bytes() == b""


def test_serialization_does_not_hide_nonfinite_or_unknown_values(tmp_path):
    for index, row in enumerate(
        [
            {"geometry_valid": np.bool_(True), "other": float("nan")},
            {"geometry_valid": np.bool_(True), "other": object()},
            {"geometry_valid": "true"},
        ]
    ):
        with pytest.raises((ValueError, TypeError)):
            runtime.save_results(tmp_path / str(index), [row], {})


def test_validation_path_only_admits_the_reviewed_empty_epoch5(tmp_path):
    old = tmp_path / "validation/epoch-0005"
    old.mkdir(parents=True)
    (old / "per_sample.jsonl").write_text("")
    resumed = tmp_path / "validation/epoch-0005-attempt-02"
    resumed.mkdir()
    assert runtime.validation_summary_path(tmp_path, 5, 42) == resumed / "summary.json"
    (old / "per_sample.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        runtime.validation_summary_path(tmp_path, 5, 42)
    (old / "per_sample.jsonl").write_text("")
    (tmp_path / "validation/epoch-0005-attempt-03").mkdir()
    with pytest.raises(ValueError):
        runtime.validation_summary_path(tmp_path, 5, 42)
    with pytest.raises(ValueError):
        runtime.validation_summary_path(tmp_path, 5, 43)


@pytest.mark.parametrize(
    "fault",
    [None, "epoch", "config", "optimizer", "rng", "metrics", "selected", "started_again", "wandb"],
)
def test_resume_is_restricted_to_the_exact_saved_epoch4(tmp_path, fault):
    config = {"seed": 42, "name": "reviewed", "output": str(tmp_path)}
    rows = [{"epoch": epoch} for epoch in range(1, 5)]
    payload = {
        "format": "controlled-v2",
        "config": config,
        "epoch": 4,
        "best": None,
        "metrics": rows[-1],
        "optimizer": {"state": {0: {"step": 1}}},
        "rng": {"cuda": [torch.zeros(1, dtype=torch.uint8)]},
    }
    if fault == "epoch":
        payload["epoch"] = 5
    elif fault == "config":
        payload["config"] = {**config, "seed": 43}
    elif fault == "optimizer":
        payload["optimizer"]["state"] = {}
    elif fault == "rng":
        payload["rng"]["cuda"] = []
    elif fault == "metrics":
        rows = rows[:-1]
    elif fault == "selected":
        (tmp_path / "best.pt").touch()
    elif fault == "started_again":
        (tmp_path / "validation/epoch-0005-attempt-02").mkdir(parents=True)
    torch.save(payload, tmp_path / "last.pt")
    write_json(tmp_path / "run.json", {"id": "original"})
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    entry = {
        "name": config["name"],
        "wandb_id": "changed" if fault == "wandb" else "original",
        "resume_files": {
            name: digest(tmp_path / name) for name in ("last.pt", "run.json", "metrics.jsonl")
        },
    }
    if fault:
        with pytest.raises(ValueError):
            runtime.validate_resume(config, entry)
    else:
        assert runtime.validate_resume(config, entry) == 4
        (tmp_path / "metrics.jsonl").write_text("changed")
        with pytest.raises(ValueError):
            runtime.validate_resume(config, entry)


@pytest.mark.parametrize(
    "fault", [None, "wrong_selection", "incomplete_epochs", "changed_best", "missing_validation"]
)
def test_resume_closeout_keeps_full_epochs_and_selection_rule(tmp_path, fault):
    config = {"output": str(tmp_path), "epochs": 10, "seed": 42}
    old = tmp_path / "validation/epoch-0005"
    old.mkdir(parents=True)
    (old / "per_sample.jsonl").write_text("")
    summaries = {}
    for epoch, rate in ((5, 10), (10, 5)):
        summary = {
            "geometry": {"invalid_field_count": 0, "metrics": {"epe_mean_px": 1}},
            "paths": {"crop_policy_model": {"RR_H": {"pct": rate}, "FDR_H": {"pct": 0}}},
        }
        summaries[epoch] = summary
        folder = "epoch-0005-attempt-02" if epoch == 5 else "epoch-0010"
        write_json(tmp_path / "validation" / folder / "summary.json", summary)
    selected = 10 if fault == "wrong_selection" else 5
    model = {"weight": torch.ones(2)}
    best = {"config": config, "epoch": selected, "selection": summaries[selected], "model": model}
    last = {
        "config": config,
        "epoch": 9 if fault == "incomplete_epochs" else 10,
        "best": {"epoch": selected, "model": model},
    }
    torch.save(best, tmp_path / "best.pt")
    torch.save(last, tmp_path / "last.pt")
    write_json(
        tmp_path / "completion.json",
        {
            "status": "trained",
            "selected/epoch": selected,
            "selected/checkpoint_sha256": digest(tmp_path / "best.pt"),
        },
    )
    (tmp_path / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"epoch": epoch, "train/samples": 27024, "val/samples": 2976}) + "\n"
            for epoch in range(1, 11)
        )
    )
    if fault == "changed_best":
        best["model"] = {"weight": torch.zeros(2)}
        torch.save(best, tmp_path / "best.pt")
        done = read_json(tmp_path / "completion.json")
        done["selected/checkpoint_sha256"] = digest(tmp_path / "best.pt")
        write_json(tmp_path / "completion.json", done)
    if fault == "missing_validation":
        (tmp_path / "validation/epoch-0010/summary.json").unlink()
    if fault:
        with pytest.raises((ValueError, FileNotFoundError)):
            runtime.verify_training(config)
    else:
        assert runtime.verify_training(config)["selected/epoch"] == 5


def test_report_dispatch_retains_its_campaign_argument(monkeypatch, tmp_path):
    captured = []
    module = SimpleNamespace(main=lambda: captured.extend(sys.argv[1:]), write_json=lambda *a: None)
    monkeypatch.setattr(runtime, "install", lambda _: None)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runtime", "--amendment", str(tmp_path), "--task", "report", "--campaign", str(tmp_path)],
    )
    runtime.main()
    assert captured == ["--campaign", str(tmp_path)]
    assert module.verify_training is runtime.verify_training
