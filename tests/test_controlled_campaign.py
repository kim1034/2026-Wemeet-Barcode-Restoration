"""GPU ownership, matrix integrity and complete-result admission on CPU."""

import copy
import os

import pytest
import torch

from scripts.controlled_campaign import hardware
from scripts.controlled_campaign import run as campaign
from scripts.controlled_geometry.common import digest, read_json, write_json


def observation():
    return {
        "gpus": [{"index": "4", "uuid": "GPU-test", "memory.used": "0", "utilization.gpu": "0"}],
        "processes": [],
    }


@pytest.mark.parametrize("fault", ["uuid", "compute_process", "graphics_memory", "utilization"])
def test_busy_or_changed_gpu_cannot_launch(fault):
    data = observation()
    if fault == "uuid":
        with pytest.raises(RuntimeError):
            hardware.idle(data, 4, "GPU-different")
        return
    if fault == "compute_process":
        data["processes"].append({"pid": "999999", "gpu_uuid": "GPU-test"})
    elif fault == "graphics_memory":
        data["gpus"][0]["memory.used"] = "1024"
    else:
        data["gpus"][0]["utilization.gpu"] = "90"
    assert not hardware.idle(data, 4, "GPU-test")


def test_pid_proof_requires_the_actual_assigned_process(monkeypatch):
    data = observation()
    monkeypatch.setattr(hardware, "snapshot", lambda: data)
    with pytest.raises(RuntimeError):
        hardware.prove_own_cuda(4, "GPU-test")
    data["processes"] = [{"pid": str(os.getpid()), "gpu_uuid": "GPU-test"}]
    assert hardware.prove_own_cuda(4, "GPU-test")["pid"] == os.getpid()
    data["processes"].append({"pid": "999999", "gpu_uuid": "GPU-test"})
    with pytest.raises(RuntimeError):
        hardware.prove_own_cuda(4, "GPU-test")


def matrix_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "validate_lock", lambda _: {})
    write_json(tmp_path / "protocol.json", {})
    auth = {
        "protocol": str(tmp_path / "protocol.json"),
        "protocol_sha256": digest(tmp_path / "protocol.json"),
        "lanes": [],
        "config_sha256": {},
    }
    for index, model, preset in (
        (4, "resnet18-c3", "R"),
        (5, "resnet18-c3", "M"),
        (6, "mobilenetv3-small", "R"),
        (7, "mobilenetv3-small", "M"),
    ):
        lane = {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "model": model,
            "preset": preset,
            "configs": [],
        }
        for seed in (42, 43, 44):
            path = tmp_path / f"{model}-{preset}-{seed}.json"
            write_json(path, {"name": path.stem, "model": model, "preset": preset, "seed": seed})
            lane["configs"].append(str(path))
            auth["config_sha256"][str(path)] = digest(path)
        auth["lanes"].append(lane)
    write_json(tmp_path / "authorization.json", auth)
    return auth


@pytest.mark.parametrize("fault", ["extra_gpu", "duplicate_gpu", "seed", "changed_config"])
def test_matrix_drift_rejected(tmp_path, monkeypatch, fault):
    auth = matrix_fixture(tmp_path, monkeypatch)
    assert campaign.validate_authorization(tmp_path) == auth
    if fault == "extra_gpu":
        auth["lanes"][0]["gpu_index"] = 0
    elif fault == "duplicate_gpu":
        auth["lanes"][0]["gpu_uuid"] = auth["lanes"][1]["gpu_uuid"]
    elif fault == "seed":
        auth["lanes"][0]["configs"][0] = auth["lanes"][0]["configs"][1]
    else:
        path = auth["lanes"][0]["configs"][0]
        write_json(path, {**read_json(path), "seed": 99})
    write_json(tmp_path / "authorization.json", auth)
    with pytest.raises(ValueError):
        campaign.validate_authorization(tmp_path)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "ground_truth", "d0"])
def test_evaluation_population_cannot_change(fault):
    records = [
        {"id": "a", "gt_text": "one", "d0_status": "U"},
        {"id": "b", "gt_text": "two", "d0_status": "C"},
    ]
    reference = {r["id"]: copy.deepcopy(r) for r in records}
    campaign.check_population(records, reference)
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif fault == "ground_truth":
        records[0]["gt_text"] = "other"
    else:
        records[0]["d0_status"] = "C"
    with pytest.raises(ValueError):
        campaign.check_population(records, reference)


def test_execution_source_mutation_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    source = tmp_path / "runner.py"
    source.write_text("original\n")
    write_json(tmp_path / "operations.json", {"files": {"runner.py": digest(source)}})
    campaign.verify_operations(tmp_path)
    source.write_text("changed\n")
    with pytest.raises(ValueError):
        campaign.verify_operations(tmp_path)


def test_profile_completion_requires_matching_result_hash(tmp_path):
    write_json(tmp_path / "result.json", {"status": "passed"})
    write_json(
        tmp_path / "completion.json",
        {"status": "passed", "result_sha256": digest(tmp_path / "result.json")},
    )
    campaign.verify_profile(tmp_path)
    write_json(tmp_path / "result.json", {"status": "failed"})
    with pytest.raises(ValueError):
        campaign.verify_profile(tmp_path)


@pytest.mark.parametrize("fault", ["incomplete_epochs", "wrong_selection", "changed_best"])
def test_training_closeout_rejects_partial_or_wrong_checkpoint(tmp_path, fault):
    cfg = {"output": str(tmp_path), "epochs": 10}
    summaries = {
        epoch: {
            "geometry": {"invalid_field_count": 0, "metrics": {"epe_mean_px": 1}},
            "paths": {"crop_policy_model": {"RR_H": {"pct": rate}, "FDR_H": {"pct": 0}}},
        }
        for epoch, rate in ((5, 10), (10, 5))
    }
    for epoch, summary in summaries.items():
        write_json(tmp_path / f"validation/epoch-{epoch:04d}/summary.json", summary)
    selected = 10 if fault == "wrong_selection" else 5
    model = {"weight": torch.ones(2)}
    best = {"config": cfg, "epoch": selected, "selection": summaries[selected], "model": model}
    last = {
        "config": cfg,
        "epoch": 9 if fault == "incomplete_epochs" else 10,
        "best": {"epoch": selected, "model": model},
    }
    torch.save(best, tmp_path / "best.pt")
    torch.save(last, tmp_path / "last.pt")
    write_json(
        tmp_path / "completion.json",
        {
            "status": "trained",
            "selected/checkpoint_sha256": digest(tmp_path / "best.pt"),
            "selected/epoch": selected,
        },
    )
    import json

    (tmp_path / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"epoch": epoch, "train/samples": 27024, "val/samples": 2976}) + "\n"
            for epoch in range(1, 11)
        )
    )
    if fault == "changed_best":
        campaign.verify_training(cfg)
        best["model"] = {"weight": torch.zeros(2)}
        torch.save(best, tmp_path / "best.pt")
    with pytest.raises(ValueError):
        campaign.verify_training(cfg)
