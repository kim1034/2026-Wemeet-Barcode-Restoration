"""Evaluate a selected v2 checkpoint on a frozen population, optionally timing it."""

import argparse
import json
from pathlib import Path

import torch

from scripts.controlled_geometry.common import configure, digest, read_json, write_json
from scripts.controlled_geometry.data import (
    RenderCache,
    benchmark_samples,
    load_split,
    validation_samples,
)
from scripts.controlled_geometry.evaluation import (
    evaluate_samples,
    measure_timing,
    selection_metrics,
)
from scripts.controlled_geometry.tracking import (
    configure_local_directories,
    log_artifact,
    log_examples,
    start_run,
)
from scripts.controlled_geometry.train import validate_config
from wemeet.ai.geometry.controlled import ControlledGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("validation", "benchmark"), default="benchmark")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--timing", action="store_true", help="Run separately without competing workloads"
    )
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["format"] != "controlled-v2-selected":
        raise ValueError("Expected the validation-selected v2 checkpoint")
    config = payload["config"]
    lock = validate_config(config)
    if not args.execute:
        print(
            json.dumps({"status": "validated_only", "checkpoint_sha256": digest(args.checkpoint)})
        )
        return
    if args.device.startswith("cuda") and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("Expose exactly one assigned CUDA GPU; no CPU fallback")
    configure(42, threads=1, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    configure_local_directories()
    metadata = {
        **config,
        "dataset": args.dataset,
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_sha256": digest(args.checkpoint),
        "device": args.device,
        "evaluation_scope": "synthetic crop; Stage1 excluded; not independent final test",
        "timing_requested": args.timing,
    }
    with start_run(
        config["name"] + "-" + args.dataset, metadata, args.output, job_type="evaluation"
    ) as run:
        model = ControlledGeometry(config["model"])
        model.load_state_dict(payload["model"], strict=True)
        model.to(args.device).eval()
        if args.dataset == "benchmark":
            samples = benchmark_samples(lock["data_root"])
        else:
            split = load_split(lock["split_manifest"])
            samples = validation_samples(
                split["validation"], RenderCache(lock["cache_root"], lock["renderer_fingerprint"])
            )
        samples = list(samples)
        reference = read_json(lock["d0_manifest"])[args.dataset]
        if {s[0] for s in samples} != set(reference):
            raise ValueError("Evaluation population changed")
        summary, records = evaluate_samples(
            model, samples, args.device, args.output / "results", reference
        )
        metadata["n"] = len(records)
        write_json(args.output / "run_config.json", metadata)
        run.summary.update(selection_metrics(summary, "eval"))
        log_examples(run, records)
        if args.timing:
            timing = measure_timing(model, samples, args.device)
            write_json(args.output / "timing.json", timing)
            for path, values in timing.items():
                run.summary.update(
                    {
                        f"latency/{path}/p50_ms": values["p50_ms"],
                        f"latency/{path}/p95_ms": values["p95_ms"],
                    }
                )
        paths = {
            str(p.relative_to(args.output)): p
            for p in args.output.rglob("*")
            if p.is_file() and "wandb" not in p.parts
        }
        log_artifact(
            run,
            config["name"] + "-" + args.dataset,
            "evaluation",
            paths,
            metadata={"checkpoint_sha256": metadata["checkpoint_sha256"]},
        )
        hashes = {name: digest(path) for name, path in paths.items()}
        write_json(
            args.output / "completion.json",
            {"status": "evaluated", "n": len(records), "run_url": run.url, "files": hashes},
        )


if __name__ == "__main__":
    main()
