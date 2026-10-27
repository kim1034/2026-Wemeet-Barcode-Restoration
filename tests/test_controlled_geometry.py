"""CPU checks for model parity, fixed populations and restartable comparisons."""

import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.controlled_geometry.common import (
    PRESETS,
    configure,
    digest,
    read_json,
    selection_key,
    write_json,
)
from scripts.controlled_geometry.compare import paired_comparison
from scripts.controlled_geometry.data import GeometryDataset, RenderCache, load_split, preprocess
from scripts.controlled_geometry.evaluation import evaluate_samples
from scripts.controlled_geometry.prepare import generate_configs
from scripts.controlled_geometry.train import (
    make_checkpoint,
    next_validation_path,
    record_metrics,
    restore_checkpoint,
    run_epoch,
)
from scripts.train_geometry import RecipeDataset
from wemeet.ai.geometry.controlled import BACKBONES, ControlledGeometry
from wemeet.ai.geometry.model import ResNet18C3, destination_grid


@pytest.fixture(autouse=True)
def cpu_only():
    configure(42, threads=1)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("backbone,channels", [(BACKBONES[0], 256), (BACKBONES[1], 48)])
def test_spatial_contract_and_training_gradients(backbone, channels):
    model = ControlledGeometry(backbone)
    image = torch.randn(2, 3, 160, 384)
    with torch.no_grad():
        assert model.backbone(image).shape == (2, channels, 10, 24)
        assert torch.equal(model(image), destination_grid().expand(2, -1, -1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = destination_grid().expand(2, -1, -1) + 0.02
    for _ in range(2):
        optimizer.zero_grad()
        nn.functional.smooth_l1_loss(model(image), target, beta=0.02).backward()
        optimizer.step()
    assert (
        model.backbone[0][0].weight.grad is not None
        if backbone == BACKBONES[1]
        else model.backbone[0].weight.grad is not None
    )
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.equal(model(image), destination_grid().expand(2, -1, -1))


def test_resnet_legacy_state_and_prediction_are_exact():
    legacy = ResNet18C3(pretrained=False).eval()
    nn.init.normal_(legacy.output.weight, std=0.02)
    controlled = ControlledGeometry(BACKBONES[0]).eval()
    controlled.load_state_dict(legacy.state_dict(), strict=True)
    image = torch.randn(1, 3, 160, 384)
    with torch.inference_mode():
        assert torch.equal(legacy(image), controlled(image))


def test_common_head_shapes_are_identical():
    a, b = (ControlledGeometry(name) for name in BACKBONES)
    for name in ("head", "output"):
        assert {k: v.shape for k, v in getattr(a, name).state_dict().items()} == {
            k: v.shape for k, v in getattr(b, name).state_dict().items()
        }


def fake_record(name, seed):
    return {
        "id": name,
        "bucket": "L",
        "band": "target",
        "recipe": {"symbology": "Code128", "seed": seed},
    }


def test_split_is_independent_of_learning_seed(tmp_path):
    path = tmp_path / "split.json"
    value = {"train": [fake_record("one", 1)], "validation": [fake_record("two", 2)]}
    write_json(path, value)
    original = digest(path)
    for seed in (42, 43, 44):
        configure(seed)
        assert load_split(path) == value
        assert digest(path) == original


@pytest.mark.parametrize("fault", ["duplicate", "overlap", "recipe", "symbology"])
def test_split_rejects_invalid_population(tmp_path, fault):
    value = {"train": [fake_record("one", 1)], "validation": [fake_record("two", 2)]}
    if fault == "duplicate":
        value["train"].append(copy.deepcopy(value["train"][0]))
    elif fault == "overlap":
        value["validation"][0]["id"] = "one"
    elif fault == "recipe":
        value["validation"][0]["recipe"] = value["train"][0]["recipe"]
    else:
        value["train"][0]["recipe"]["symbology"] = "QR"
    write_json(tmp_path / "split.json", value)
    with pytest.raises(ValueError):
        load_split(tmp_path / "split.json")


def test_epe_is_global_2d_and_not_batch_mean():
    inputs = torch.zeros(3, 48, 2)
    target = inputs.clone()
    target[0, :, 0], target[0, :, 1] = 0.03, 0.04
    dataset = TensorDataset(inputs, target, torch.full((3, 2), 101.0), torch.arange(3))
    small = run_epoch(nn.Identity(), DataLoader(dataset, batch_size=2), "cpu", 0.02)
    large = run_epoch(nn.Identity(), DataLoader(dataset, batch_size=3), "cpu", 0.02)
    assert small["epe_mean_px"] == pytest.approx(5 / 3)
    assert small["epe_p95_px"] == pytest.approx(5)
    for key in ("epe_mean_px", "epe_p95_px", "epe_rmse_px"):
        assert small[key] == large[key]


def test_checkpoint_resume_preserves_shuffle_dropout_and_optimizer():
    dataset = TensorDataset(
        torch.randn(7, 4), torch.randn(7, 48, 2), torch.full((7, 2), 101.0), torch.arange(7)
    )
    model = nn.Sequential(nn.Linear(4, 96), nn.Dropout(0.2), nn.Unflatten(1, (48, 2)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=3, shuffle=True, generator=generator)
    run_epoch(model, loader, "cpu", 0.02, optimizer)
    buffer = io.BytesIO()
    torch.save(
        make_checkpoint(model, optimizer, generator, 1, {"seed": 42}, None, 3, "cpu"), buffer
    )
    run_epoch(model, loader, "cpu", 0.02, optimizer)
    expected = copy.deepcopy(model.state_dict())
    torch.randn(10)
    restored = nn.Sequential(nn.Linear(4, 96), nn.Dropout(0.2), nn.Unflatten(1, (48, 2)))
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_generator = torch.Generator()
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    with pytest.raises(ValueError, match="configuration"):
        restore_checkpoint(payload, {"seed": 43}, restored, restored_optimizer, restored_generator)
    restore_checkpoint(payload, {"seed": 42}, restored, restored_optimizer, restored_generator)
    restored_loader = DataLoader(dataset, batch_size=3, shuffle=True, generator=restored_generator)
    run_epoch(restored, restored_loader, "cpu", 0.02, restored_optimizer)
    assert all(torch.equal(expected[k], v) for k, v in restored.state_dict().items())


def test_selection_uses_decode_before_geometry():
    def result(rr, fdr, epe):
        return {
            "geometry": {"invalid_field_count": 0, "metrics": {"epe_mean_px": epe}},
            "paths": {"crop_policy_model": {"RR_H": {"pct": rr}, "FDR_H": {"pct": fdr}}},
        }

    assert selection_key(result(90, 2, 3), 10) < selection_key(result(80, 0, 1), 5)
    assert selection_key(result(90, 1, 3), 10) < selection_key(result(90, 2, 1), 5)
    assert selection_key(result(90, 1, 1), 5) < selection_key(result(90, 1, 1), 10)
    with pytest.raises(ValueError):
        selection_key(result(None, None, 1), 5)


def test_policy_keeps_d0_wrong_even_when_forced_is_correct(monkeypatch):
    import scripts.controlled_geometry.evaluation as evaluation

    monkeypatch.setattr(evaluation, "predict", lambda *args: destination_grid().numpy())
    wrong = {"text": "wrong", "format": "Code128", "backend": "fake"}
    monkeypatch.setattr(evaluation, "observe_decode", lambda *args: (None, wrong))
    monkeypatch.setattr(
        evaluation,
        "crop_attempts",
        lambda *args: {"final": {"text": "right", "format": "Code128"}, "attempts": []},
    )
    samples = [
        (
            "one",
            {"text": "right", "bucket": "low"},
            np.zeros((20, 40), np.uint8),
            destination_grid().numpy(),
            destination_grid().numpy(),
        )
    ]
    summary, records = evaluate_samples(nn.Identity(), samples)
    assert summary["paths"]["crop_forced_model"]["C"] == 1
    assert summary["paths"]["crop_policy_model"]["W"] == 1
    assert records[0]["paths"]["crop_policy_model"]["early_exit"]
    with pytest.raises(ValueError, match="Frozen"):
        evaluate_samples(nn.Identity(), samples, reference={})


def test_cache_preserves_historical_preprocessing_and_detects_corruption(tmp_path):
    path = Path("runs/restoration-resnet18-c3-v1/split.json")
    if not path.exists():
        pytest.skip("Local recipe fixture absent")
    record = read_json(path)["train"][0]
    cache = RenderCache(tmp_path, "test-renderer")
    actual = GeometryDataset([record], cache)[0]
    expected = RecipeDataset([record])[0]
    for a, b in zip(actual[:3], expected[:3], strict=True):
        assert torch.equal(a, b)
    second = cache.get(record)
    assert torch.equal(preprocess(second["gray"]), actual[0])
    next(tmp_path.glob("*.npz")).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        cache.get(record)


def test_matrix_has_exactly_four_conditions_for_each_seed(tmp_path):
    write_json(tmp_path / "protocol.json", {"selection": "fixed"})
    configs = generate_configs(tmp_path / "protocol.json", tmp_path)
    assert len(configs) == 12
    assert len({(c["model"], c["preset"], c["seed"]) for c in configs}) == 12
    for c in configs:
        assert all(c[k] == v for k, v in PRESETS[c["preset"]].items())
        assert c["protocol_lock_sha256"] == digest(tmp_path / "protocol.json")


@pytest.mark.parametrize(
    "key,value",
    [
        ("precision", "fp16"),
        ("scheduler", "cosine"),
        ("cpu_threads", 8),
        ("num_workers", 4),
        ("name", "mislabelled"),
        ("wandb_group", "wrong"),
        ("weight_decay", 0.01),
        ("smooth_l1_beta", 0.1),
    ],
)
def test_config_rejects_silent_condition_changes(tmp_path, monkeypatch, key, value):
    import scripts.controlled_geometry.train as training
    from scripts.controlled_geometry.common import PROTOCOL

    lock = {
        "selection": "fixed",
        "presets": PRESETS,
        "pretrained": dict.fromkeys(BACKBONES),
        "protocol": PROTOCOL,
    }
    write_json(tmp_path / "protocol.json", lock)
    config = generate_configs(tmp_path / "protocol.json", tmp_path)[0]
    monkeypatch.setattr(training, "validate_lock", lambda *a, **k: lock)
    assert training.validate_config(config) == lock
    config[key] = value
    with pytest.raises(ValueError):
        training.validate_config(config)


def test_comparison_refuses_modified_output_files(tmp_path, monkeypatch):
    from scripts.controlled_geometry.compare import main

    directories = [tmp_path / "left", tmp_path / "right"]
    for directory in directories:
        directory.mkdir()
        write_json(
            directory / "run_config.json",
            {
                "protocol_lock_sha256": "same",
                "dataset": "benchmark",
                "preset": "R",
                "seed": 42,
                "n": 2,
            },
        )
        write_json(
            directory / "completion.json",
            {
                "status": "evaluated",
                "n": 2,
                "files": {"run_config.json": digest(directory / "run_config.json")},
            },
        )
    (directories[0] / "run_config.json").write_text(
        (directories[0] / "run_config.json").read_text() + " "
    )
    monkeypatch.setattr(
        "sys.argv",
        ["compare", *map(str, directories), "--output", str(tmp_path / "comparison.json")],
    )
    with pytest.raises(ValueError, match="artifact changed"):
        main()


def test_recovery_preserves_previous_validation_and_deduplicates_metrics(tmp_path):
    first = next_validation_path(tmp_path, 5)
    first.mkdir(parents=True)
    assert next_validation_path(tmp_path, 5) != first
    record_metrics(tmp_path, {"epoch": 5, "loss": 0.1})
    record_metrics(tmp_path, {"epoch": 5, "loss": 0.1})
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        record_metrics(tmp_path, {"epoch": 5, "loss": 0.2})


def test_paired_comparison_rejects_unpaired_data():
    def row(i, status):
        return {
            "id": str(i),
            "group_id_proxy": str(i),
            "gt_text": str(i),
            "d0_status": "U",
            "gt_src_norm": [],
            "crop_hw": [20, 40],
            "paths": {
                path: {"status": status} for path in ("crop_forced_model", "crop_policy_model")
            },
        }

    rows = [row(1, "C"), row(2, "W")]
    result = paired_comparison(rows, copy.deepcopy(rows), draws=100)
    assert result["paths"]["crop_policy_model"]["RR_H"]["ci95_pp"] == [0, 0]
    with pytest.raises(ValueError):
        paired_comparison(rows, rows[:1], draws=100)
    changed = copy.deepcopy(rows)
    changed[0]["d0_status"] = "W"
    with pytest.raises(ValueError):
        paired_comparison(rows, changed, draws=100)


def test_offline_wandb_roundtrip(tmp_path, monkeypatch):
    from scripts.controlled_geometry.tracking import log_artifact, start_run

    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"technical_smoke_only": True, "gpu_used": False}))
    with start_run(
        "cpu-technical-smoke",
        {"candidate_result": False},
        tmp_path,
        mode="offline",
        job_type="preparation",
    ) as run:
        run.log({"checks/cpu": 1})
        log_artifact(run, "cpu-smoke", "checks", {"evidence.json": evidence})
        assert run.offline
    assert list((tmp_path / "wandb").glob("offline-run-*/*.wandb"))
