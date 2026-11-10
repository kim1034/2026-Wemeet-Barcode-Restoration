"""Original numerical entry points with preserved recovery and retryable W&B publication."""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import wandb

from scripts.controlled_durable_20261026.recovery import (
    checkpoint_state,
    verify_durable,
    verify_training,
    wait_seconds,
)
from scripts.controlled_geometry.common import digest, read_json
from scripts.controlled_resume_20261026.runtime import install as install_json_correction


def network_retry(function):
    failures = 0
    while True:
        try:
            return function()
        except (wandb.errors.CommError, TimeoutError, ConnectionError) as exc:
            failures += 1
            delay = wait_seconds(failures)
            print(
                json.dumps(
                    {"event": "wandb_retry", "error": type(exc).__name__, "retry_in_seconds": delay}
                ),
                flush=True,
            )
            time.sleep(delay)


def install(campaign, resumed_epoch):
    from scripts.controlled_geometry import tracking

    verify_durable(campaign)
    install_json_correction(campaign)
    original_start = tracking.start_run
    original_artifact = tracking.log_artifact
    manifest = campaign / "durable/manifest.json"
    extra = {
        "durable-continuation/manifest.json": manifest,
        "durable-continuation/README_ko.md": campaign / "durable/README_ko.md",
        "durable-continuation/source.zip": campaign / "durable/source.zip",
    }

    def start_run(*args, **kwargs):
        run = network_retry(lambda: original_start(*args, **kwargs))
        run.summary.update(
            {
                "durable_continuation/sha256": digest(manifest),
                "durable_continuation/resumed_from_epoch": resumed_epoch,
                "runtime_amendment/resumed_from_epoch": resumed_epoch,
            }
        )
        network_retry(
            lambda: original_artifact(run, "controlled-v2-durable-continuation", "code", extra)
        )
        return run

    def log_artifact(run, name, kind, paths, metadata=None):
        return network_retry(
            lambda: original_artifact(
                run,
                name,
                kind,
                {**paths, **extra},
                metadata={**(metadata or {}), "durable_continuation_sha256": digest(manifest)},
            )
        )

    tracking.start_run = start_run
    tracking.log_artifact = log_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--task", choices=("train", "evaluate", "profile", "report"), required=True)
    args, remaining = parser.parse_known_args()
    campaign = args.continuation.resolve()
    resumed_epoch = 0
    if args.task == "train" and "--resume" in remaining:
        config = read_json(remaining[remaining.index("--config") + 1])
        payload, _ = checkpoint_state(campaign, config)
        resumed_epoch = payload["epoch"]
    install(campaign, resumed_epoch)
    module_name = {
        "train": "scripts.controlled_geometry.train",
        "evaluate": "scripts.controlled_geometry.evaluate",
        "profile": "scripts.controlled_campaign.profile",
        "report": "scripts.controlled_campaign.report",
    }[args.task]
    module = importlib.import_module(module_name)
    if args.task == "report":
        module.verify_training = lambda config: verify_training(campaign, config)
        original_write = module.write_json

        def write_json(path, value):
            if Path(path).name == "summary.json" and "interpretation" in value:
                value["interpretation"].append(
                    "The epoch5 JSON correction preserves numerical conditions. "
                    "Saved checkpoints resume; interrupted outputs and recovery records remain."
                )
            original_write(path, value)

        module.write_json = write_json
    sys.argv = [module_name, *remaining]
    module.main()


if __name__ == "__main__":
    main()
