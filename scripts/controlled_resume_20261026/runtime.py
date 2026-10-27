"""Keep the original experiment source intact; amend only JSON output and recovery."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.controlled_campaign.run import verify_operations
from scripts.controlled_geometry import evaluation
from scripts.controlled_geometry.common import ROOT, digest, read_json, selection_key

ORIGINAL_SAVE_RESULTS = evaluation.save_results


def save_results(output, records, summary):
    """NumPy comparisons return np.bool_; preserve its value as a JSON boolean."""
    for row in records:
        value = row["geometry_valid"]
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("geometry_valid must be a boolean")
        row["geometry_valid"] = bool(value)
    ORIGINAL_SAVE_RESULTS(output, records, summary)


def verify_revision(directory):
    manifest = read_json(directory / "revision.json")
    for relative, expected in manifest["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Resume source changed: {relative}")
    for relative, expected in manifest["preserved"].items():
        if digest(directory / relative) != expected:
            raise ValueError(f"Preserved interruption evidence changed: {relative}")
    for relative, expected in manifest["inputs"].items():
        if digest(directory / relative) != expected:
            raise ValueError(f"Resume input changed: {relative}")
    verify_operations(directory)
    return manifest


def validate_resume(config, entry):
    output = Path(config["output"])
    if config["seed"] != 42 or config["name"] != entry["name"]:
        raise ValueError("Only the four interrupted seed42 runs may resume")
    if (output / "completion.json").exists() or (output / "best.pt").exists():
        raise ValueError("Completed/selected run cannot enter this recovery")
    for relative, expected in entry["resume_files"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Interrupted run changed before resume: {relative}")
    payload = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    rows = [json.loads(x) for x in (output / "metrics.jsonl").read_text().splitlines()]
    if (
        payload["format"] != "controlled-v2"
        or payload["config"] != config
        or payload["epoch"] != 4
        or payload["best"] is not None
        or [x["epoch"] for x in rows] != [1, 2, 3, 4]
        or rows[-1] != payload["metrics"]
        or read_json(output / "run.json")["id"] != entry["wandb_id"]
        or (output / "validation/epoch-0005-attempt-02").exists()
    ):
        raise ValueError("Checkpoint is not the reviewed epoch4 interruption")
    if not payload["optimizer"]["state"] or len(payload["rng"]["cuda"]) != 1:
        raise ValueError("Missing optimizer or assigned CUDA RNG state")
    return payload["epoch"]


def validation_summary_path(output, epoch, seed):
    """Only the observed, empty epoch5 write has a specifically admitted successor."""
    output = Path(output)
    stem = f"epoch-{epoch:04d}"
    folders = sorted((output / "validation").glob(stem + "*"))
    if seed == 42 and epoch == 5:
        original = output / "validation" / stem
        resumed = output / "validation" / (stem + "-attempt-02")
        if (
            set(folders) != {original, resumed}
            or sorted(p.name for p in original.iterdir()) != ["per_sample.jsonl"]
            or (original / "per_sample.jsonl").stat().st_size != 0
        ):
            raise ValueError("Unexpected original epoch5 data or additional attempts")
        return resumed / "summary.json"
    expected = output / "validation" / stem
    if folders != [expected]:
        raise ValueError("Unexpected validation retry outside the reviewed epoch5 failure")
    return expected / "summary.json"


def verify_training(config):
    """Original completion/selection checks, resolving the preserved epoch5 interruption."""
    output = Path(config["output"])
    done = read_json(output / "completion.json")
    if done["status"] != "trained" or done["selected/checkpoint_sha256"] != digest(
        output / "best.pt"
    ):
        raise ValueError("Incomplete/changed selected checkpoint")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    if best["config"] != config or last["config"] != config or last["epoch"] != config["epochs"]:
        raise ValueError("Wrong checkpoint configuration/epoch")
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    if [m["epoch"] for m in metrics] != list(range(1, config["epochs"] + 1)):
        raise ValueError("Missing or duplicated training epoch")
    if any(m["train/samples"] != 27024 or m["val/samples"] != 2976 for m in metrics):
        raise ValueError("Wrong training/validation population")
    if done["selected/epoch"] != best["epoch"] or best["epoch"] != last["best"]["epoch"]:
        raise ValueError("Selected checkpoint provenance mismatch")
    summaries = {
        epoch: read_json(validation_summary_path(output, epoch, config["seed"]))
        for epoch in range(5, config["epochs"] + 1, 5)
    }
    chosen = min(summaries, key=lambda epoch: selection_key(summaries[epoch], epoch))
    if best["epoch"] != chosen or best["selection"] != summaries[chosen]:
        raise ValueError("Checkpoint did not follow the frozen selection rule")
    if any(
        not torch.equal(value, last["best"]["model"][key]) for key, value in best["model"].items()
    ):
        raise ValueError("Best/last selected model states differ")
    return done


def install(directory):
    """Apply the narrow correction before the original entry point imports its helpers."""
    from scripts.controlled_geometry import tracking

    manifest = verify_revision(directory)
    evaluation.save_results = save_results
    original_start = tracking.start_run
    original_log_artifact = tracking.log_artifact
    revision_files = {
        "runtime-amendment/revision.json": directory / "revision.json",
        "runtime-amendment/README_ko.md": directory / "README_ko.md",
        "runtime-amendment/source.zip": directory / "source.zip",
    }

    def start_run(*args, **kwargs):
        run = original_start(*args, **kwargs)
        try:
            run.summary.update(
                {
                    "runtime_amendment/id": manifest["id"],
                    "runtime_amendment/sha256": digest(directory / "revision.json"),
                    "runtime_amendment/reason": manifest["reason"],
                    "runtime_amendment/resumed_from_epoch": 4 if kwargs.get("resume") else 0,
                }
            )
            original_log_artifact(run, "controlled-v2-json-bool-amendment", "code", revision_files)
        except BaseException:
            run.finish(exit_code=1)
            raise
        return run

    def log_artifact(run, name, kind, paths, metadata=None):
        return original_log_artifact(
            run,
            name,
            kind,
            {**paths, **revision_files},
            metadata={
                **(metadata or {}),
                "runtime_amendment_sha256": digest(directory / "revision.json"),
            },
        )

    tracking.start_run = start_run
    tracking.log_artifact = log_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--task", choices=("train", "evaluate", "profile", "report"), required=True)
    args, remaining = parser.parse_known_args()
    install(args.amendment.resolve())
    module_name = {
        "train": "scripts.controlled_geometry.train",
        "evaluate": "scripts.controlled_geometry.evaluate",
        "profile": "scripts.controlled_campaign.profile",
        "report": "scripts.controlled_campaign.report",
    }[args.task]
    module = importlib.import_module(module_name)
    if args.task == "report":
        module.verify_training = verify_training
        # Include the incident in the original report's artifact before completion.
        original_write = module.write_json

        def write_json(path, value):
            if Path(path).name == "summary.json" and "interpretation" in value:
                value["interpretation"].append(
                    "All four seed42 runs resumed from epoch4 after the epoch5 JSON failure; "
                    "original failures/checkpoints are preserved and runtime amendment is attached."
                )
            original_write(path, value)

        module.write_json = write_json
    sys.argv = [module_name, *remaining]
    module.main()


if __name__ == "__main__":
    main()
