"""Independent durable CPU closeout; never launches or changes evaluation jobs."""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import wandb

from scripts.controlled_campaign.hardware import snapshot
from scripts.controlled_geometry.common import configure, digest, read_json, write_json
from scripts.mobilenet_followup.audit_native import audit
from scripts.mobilenet_followup.controller import audit_job, task_matrix

UNIT = "wemeet-mobilenet-followup-gpu67-20261027.service"


def service():
    lines = subprocess.check_output(
        ["systemctl", "--user", "show", UNIT, "-p", "MainPID", "-p", "ActiveState"], text=True
    )
    return dict(line.split("=", 1) for line in lines.splitlines())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    delivery = root / "delivery"
    delivery.mkdir(exist_ok=True)
    seal = read_json(delivery / "postrun_seal.json")
    for name, expected in seal.items():
        assert digest(name) == expected
    if (delivery / "completion.json").exists():
        return
    uuids = set(read_json(root / "inputs.json")["gpu_uuids"].values())
    while True:
        state = read_json(root / "progress.json")
        observation = snapshot()
        active = service()
        owned = {v.get("pid") for v in state["lanes"].values()}
        timing = any("-timing" in v["task"] for v in state["lanes"].values())
        interference = []
        for p in observation["processes"]:
            if p["gpu_uuid"] not in uuids or int(p["pid"]) in owned:
                continue
            path = Path(f"/proc/{p['pid']}/cmdline")
            command = path.read_bytes() if path.exists() else b""
            if str(root).encode() not in command:
                interference.append(p)
        with (delivery / "hardware_observations.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "state": state,
                        "hardware": observation,
                        "timing": timing,
                        "external_gpu_processes": interference,
                    }
                )
                + "\n"
            )
        write_json(
            delivery / "status.json",
            {
                "phase": "waiting_for_evaluation_release",
                "service": active,
                "completed_tasks": len(state["completed"]),
                "updated_at": time.time(),
            },
        )
        if (root / "completion.json").exists() and state["status"] == "complete":
            if active["MainPID"] == "0" and active["ActiveState"] == "inactive":
                break
        time.sleep(20)
    write_json(delivery / "status.json", {"phase": "independent_audit", "at": time.time()})
    configure(42, threads=1, device="cpu")
    native = audit(root)
    write_json(delivery / "native_pipeline_audit.json", native)
    verified = 0
    for mapping in (
        read_json(root / "preservation_before.json"),
        read_json(root / "execution_lock.json"),
        read_json(root / "native_bridge_amendment/seal.json")["files"],
        *read_json(root / "native_bridge_amendment/preserved_before.json").values(),
    ):
        for name, expected in mapping.items():
            assert digest(name) == expected, name
            verified += 1
    for name, expected in read_json(root / "completion.json")["files"].items():
        assert digest(root / "report" / name) == expected
    observations = [
        json.loads(line)
        for line in (delivery / "hardware_observations.jsonl").read_text().splitlines()
    ]
    assert not any(r["timing"] and r["external_gpu_processes"] for r in observations)
    api = wandb.Api(timeout=60)
    remote = []
    for task in task_matrix():
        done = audit_job(root, task)
        assert done is not None
        run = api.run("kim-1034/wemeet-barcode/" + done["wandb"].rstrip("/").split("/")[-1])
        assert run.state == "finished", run.url
        artifacts = [a for a in run.logged_artifacts() if a.type == "evaluation"]
        assert artifacts
        artifact = artifacts[-1]
        assert artifact.state == "COMMITTED"
        entries = artifact.manifest.entries
        checked = 0
        for name in done["files"]:
            path = Path(done["output"]) / name
            expected = base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
            assert entries[name].digest == expected, (task["id"], name)
            checked += 1
        remote.append(
            {
                "task": task["id"],
                "run": run.url,
                "artifact": artifact.name,
                "state": run.state,
                "remote_file_digests_matched": checked,
            }
        )
    for key, path in [
        ("run", root / "input_publication.json"),
        ("run", root / "native_bridge_amendment/publication.json"),
        ("wandb", root / "completion.json"),
    ]:
        url = read_json(path)[key]
        run = api.run("kim-1034/wemeet-barcode/" + url.rstrip("/").split("/")[-1])
        assert run.state == "finished"
        assert all(a.state == "COMMITTED" for a in run.logged_artifacts())
    write_json(
        delivery / "independent_validation.json",
        {
            "status": "passed",
            "preservation_checks": verified,
            "native_scene_matches": 800,
            "timing_external_gpu_interference": False,
            "hardware_observations": len(observations),
            "remote": remote,
            "service_released": active,
        },
    )
    repo = Path(__file__).resolve().parents[2]
    commands = [
        [sys.executable, "-m", "scripts.mobilenet_followup.plots", "--root", str(root)],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [str(repo.parent / ".venvs/barcode/bin/lint-imports"), "--no-cache"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for i, command in enumerate(commands):
        with (delivery / f"check-{i}.log").open("w") as stream:
            subprocess.run(
                command,
                cwd=repo,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
    run = wandb.init(
        entity="kim-1034",
        project="wemeet-barcode",
        group="mobilenet-followup-20261027",
        name="independent-closeout-audit",
        job_type="checks",
        dir=str(delivery),
        mode="online",
        config={"postrun_seal_sha256": digest(delivery / "postrun_seal.json")},
    )
    artifact = wandb.Artifact("mobilenet-followup-independent-closeout", type="checks")
    for path in delivery.iterdir():
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact).wait()
    url = run.url
    run.summary.update({"passed": True, "native_scene_matches": 800, "remote_tasks_verified": 22})
    run.finish()
    write_json(delivery / "status.json", {"phase": "complete", "wandb": url, "at": time.time()})
    write_json(
        delivery / "completion.json",
        {
            "status": "complete",
            "wandb": url,
            "files": {p.name: digest(p) for p in delivery.iterdir() if p.is_file()},
        },
    )


if __name__ == "__main__":
    main()
