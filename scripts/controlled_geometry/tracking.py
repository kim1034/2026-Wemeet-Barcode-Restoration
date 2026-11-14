"""Explicit W&B lifecycle; offline checks never claim online publication."""

import os
from pathlib import Path

import wandb

from scripts.controlled_geometry.common import PROTOCOL

ENTITY = "kim-1034"
PROJECT = "wemeet-barcode"


def probe():
    """Read account/project access without ever returning credentials."""
    try:
        api = wandb.Api(timeout=15)
        project = api.project(PROJECT, entity=ENTITY)
        # Force an authenticated read; merely constructing an API is not proof.
        exists = project is not None
        return {"online_read_verified": exists, "entity": ENTITY, "project": PROJECT}
    except Exception as exc:
        return {
            "online_read_verified": False,
            "entity": ENTITY,
            "project": PROJECT,
            "error_type": type(exc).__name__,
            "action": "wandb login --verify in barcode environment",
        }


def start_run(name, config, output, mode="online", job_type="train", run_id=None, resume=False):
    if mode not in ("online", "offline"):
        raise ValueError("Explicit online or offline mode required")
    Path(output).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=PROTOCOL,
        name=name,
        config=config,
        job_type=job_type,
        tags=["stage2", "controlled-v2", job_type],
        mode=mode,
        dir=str(output),
        id=run_id,
        resume="must" if resume and mode == "online" else None,
        settings=wandb.Settings(init_timeout=45, x_disable_stats=job_type == "preparation"),
    )


def log_artifact(run, name, kind, paths, metadata=None):
    artifact = wandb.Artifact(name, type=kind, metadata=metadata or {})
    for label, path in paths.items():
        artifact.add_file(str(path), name=label)
    result = run.log_artifact(artifact)
    if not run.offline:
        result.wait()
    return result


def log_examples(run, records, limit=30):
    selected = sorted(records, key=lambda r: r["paths"]["crop_policy_model"]["status"] != "W")[
        :limit
    ]
    table = wandb.Table(
        columns=["id", "bucket", "ground_truth", "d0", "forced", "policy", "status"]
    )
    for r in selected:
        table.add_data(
            r["id"],
            r["bucket"],
            r["gt_text"],
            r["d0"]["text"],
            r["paths"]["crop_forced_model"]["final"]["text"],
            r["paths"]["crop_policy_model"]["final"]["text"],
            r["paths"]["crop_policy_model"]["status"],
        )
    run.log({"examples/decode_outcomes": table})


def configure_local_directories():
    etc = Path(__file__).resolve().parents[3]
    for key, relative in {
        "WANDB_CACHE_DIR": ".cache/wandb",
        "WANDB_CONFIG_DIR": ".cache/wandb-config",
        "WANDB_DATA_DIR": ".cache/wandb-staging",
    }.items():
        os.environ.setdefault(key, str(etc / relative))
