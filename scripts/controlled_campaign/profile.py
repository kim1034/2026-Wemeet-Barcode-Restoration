"""Real GPU profile on disposable weights; never a training/selection checkpoint."""

import argparse
import copy
import itertools
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.controlled_campaign.hardware import prove_own_cuda
from scripts.controlled_geometry.common import configure, digest, read_json, write_json
from scripts.controlled_geometry.data import (
    GeometryDataset,
    RenderCache,
    load_split,
    preprocess,
    validation_samples,
)
from scripts.controlled_geometry.evaluation import evaluate_samples, measure_timing
from scripts.controlled_geometry.tracking import (
    configure_local_directories,
    log_artifact,
    start_run,
)
from scripts.controlled_geometry.train import run_epoch, validate_config
from wemeet.ai.geometry.controlled import ControlledGeometry


def require_device(uuid):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("Expected exactly the authorized physical UUID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device is required")


def loader(records, config, cache):
    return DataLoader(
        GeometryDataset(records, cache),
        batch_size=config["batch_size"],
        num_workers=1,
        multiprocessing_context="spawn",
        pin_memory=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )


def timed_batches(model, records, config, cache, optimizer, warmup, measured):
    iterator = iter(loader(records[: (warmup + measured) * config["batch_size"]], config, cache))
    tick = time.perf_counter()
    run_epoch(
        model, itertools.islice(iterator, warmup), "cuda:0", config["smooth_l1_beta"], optimizer
    )
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - tick
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    metrics = run_epoch(
        model, itertools.islice(iterator, measured), "cuda:0", config["smooth_l1_beta"], optimizer
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - tick
    peak = torch.cuda.max_memory_allocated()
    if metrics["samples"] != measured * config["batch_size"]:
        raise ValueError("Incomplete measured batch population")
    del iterator
    return {
        "seconds": seconds,
        "samples": metrics["samples"],
        "batches": measured,
        "seconds_per_sample": seconds / metrics["samples"],
        "warmup_seconds": warmup_seconds,
        "warmup_batches": warmup,
        "startup_seconds_estimate": max(0, warmup_seconds - warmup * seconds / measured),
        "peak_allocated_bytes": peak,
        "finite_loss_and_gradients": True,
    }


def training_profile(args, config, lock):
    split = load_split(lock["split_manifest"])
    cache = RenderCache(lock["cache_root"], lock["renderer_fingerprint"])
    model = ControlledGeometry(config["model"], lock["pretrained"][config["model"]]["path"]).to(
        "cuda:0"
    )
    torch.cuda.synchronize()
    before = prove_own_cuda(args.gpu_index, args.gpu_uuid)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        train = timed_batches(model, split["train"], config, cache, optimizer, 5, 40)
        validation = timed_batches(model, split["validation"], config, cache, None, 2, 10)
    # Compare the now nonzero full head, not just the zero-initialized identity output.
    records = [
        split["validation"][i] for i in np.linspace(0, len(split["validation"]) - 1, 30, dtype=int)
    ]
    samples = list(validation_samples(records, cache))
    image = torch.stack([preprocess(s[2]) for s in samples[:6]])
    model.eval()
    cpu_model = copy.deepcopy(model).cpu().eval()
    with torch.inference_mode():
        expected = cpu_model(image)
        actual = model(image.to("cuda:0")).cpu()
        expected_features = cpu_model.backbone(image)
        actual_features = model.backbone(image.to("cuda:0")).cpu()
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(actual_features, expected_features, atol=2e-4, rtol=2e-4)
    tick = time.perf_counter()
    summary, _ = evaluate_samples(
        model, samples, "cuda:0", reference=read_json(lock["d0_manifest"])["validation"]
    )
    decode_seconds = time.perf_counter() - tick
    if summary["geometry"]["invalid_field_count"]:
        raise ValueError("Nonfinite profile geometry")
    after = prove_own_cuda(args.gpu_index, args.gpu_uuid)
    epoch_seconds = (
        train["seconds_per_sample"] * len(split["train"])
        + validation["seconds_per_sample"] * len(split["validation"])
        + train["startup_seconds_estimate"]
        + validation["startup_seconds_estimate"]
    )
    full_decode_estimate = decode_seconds / len(samples) * len(split["validation"])
    return {
        "status": "passed",
        "purpose": "technical profile only; discard updated weights",
        "config": config,
        "protocol_sha256": config["protocol_lock_sha256"],
        "hardware_before": before,
        "hardware_after": after,
        "train": train,
        "validation": validation,
        "decode": {
            "samples": len(samples),
            "seconds": decode_seconds,
            "full_validation_seconds_estimate": full_decode_estimate,
        },
        "cpu_gpu_prediction_max_abs_delta": float((actual - expected).abs().max()),
        "cpu_gpu_feature_max_abs_delta": float((actual_features - expected_features).abs().max()),
        "prediction_tolerance": {"atol": 5e-5, "rtol": 5e-5},
        "feature_tolerance": {"atol": 2e-4, "rtol": 2e-4},
        "warnings": sorted({str(w.message) for w in captured}),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epoch_without_decode_seconds_estimate": epoch_seconds,
        "run_seconds_estimate": epoch_seconds * config["epochs"]
        + full_decode_estimate * (config["epochs"] // 5),
        "estimate_limit": (
            "short warm-cache profile; revise from actual epoch timings; "
            "decoder cost depends on learned predictions"
        ),
        "full_training_started": False,
    }


def latency_profile(args, config, lock, payload):
    from scripts.controlled_geometry.data import benchmark_samples

    model = ControlledGeometry(config["model"])
    model.load_state_dict(payload["model"], strict=True)
    model.to("cuda:0").eval()
    torch.cuda.synchronize()
    before = prove_own_cuda(args.gpu_index, args.gpu_uuid)
    samples = list(benchmark_samples(lock["data_root"]))
    if len(samples) != 1500:
        raise ValueError("Incomplete benchmark population")
    timing = measure_timing(model, samples, "cuda:0")
    return {
        "status": "passed",
        "purpose": "selected-checkpoint latency; Stage1 excluded",
        "checkpoint_sha256": digest(args.checkpoint),
        "selected_epoch": payload["epoch"],
        "config": config,
        "hardware_before": before,
        "hardware_after": prove_own_cuda(args.gpu_index, args.gpu_uuid),
        "timing": timing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--checkpoint", type=Path, help="Time an already selected checkpoint")
    args = parser.parse_args()
    config = read_json(args.config)
    lock = validate_config(config)
    configure(config["seed"], threads=2, device="cuda:0")
    require_device(args.gpu_uuid)
    args.output.mkdir(parents=True, exist_ok=False)
    configure_local_directories()
    job = "latency" if args.checkpoint else "profile"
    with start_run(
        config["name"] + "-" + job,
        {**config, "physical_gpu": args.gpu_index, "gpu_uuid": args.gpu_uuid, "profile_only": True},
        args.output,
        job_type=job,
    ) as run:
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if payload["format"] != "controlled-v2-selected" or payload["config"] != config:
                raise ValueError("Selected checkpoint/config mismatch")
            result = latency_profile(args, config, lock, payload)
            run.summary.update(
                {
                    f"latency/{key}/p50_ms": value["p50_ms"]
                    for key, value in result["timing"].items()
                }
            )
            run.summary.update(
                {
                    f"latency/{key}/p95_ms": value["p95_ms"]
                    for key, value in result["timing"].items()
                }
            )
        else:
            result = training_profile(args, config, lock)
            run.log(
                {
                    "profile/train_samples_per_second": 1 / result["train"]["seconds_per_sample"],
                    "profile/peak_allocated_bytes": result["train"]["peak_allocated_bytes"],
                    "profile/cpu_gpu_max_abs_delta": result["cpu_gpu_prediction_max_abs_delta"],
                    "profile/run_hours_estimate": result["run_seconds_estimate"] / 3600,
                }
            )
        result["run_url"] = run.url
        write_json(args.output / "result.json", result)
        log_artifact(
            run, config["name"] + "-" + job, "profile", {"result.json": args.output / "result.json"}
        )
    write_json(
        args.output / "completion.json",
        {
            "status": "passed",
            "result_sha256": digest(args.output / "result.json"),
            "run_url": result["run_url"],
        },
    )
    print(
        json.dumps(
            {"status": "passed", "job": job, "gpu": args.gpu_index, "run_url": result["run_url"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
