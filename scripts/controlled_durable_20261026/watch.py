"""CPU supervision until verified evaluation, aggregation and W&B publication complete."""

import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from scripts.controlled_campaign.hardware import snapshot, utc_now
from scripts.controlled_durable_20261026.recovery import verify_durable, verify_report
from scripts.controlled_geometry import tracking
from scripts.controlled_geometry.common import digest, read_json, write_json

SERVICE = "wemeet-controlled-v2-gpu4567-20261026.service"
WATCH_SERVICE = "wemeet-controlled-v2-supervision-20261026.service"


def service_state():
    text = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            SERVICE,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "NRestarts",
            "-p",
            "ExecMainStatus",
            "-p",
            "ControlGroup",
        ],
        text=True,
        timeout=20,
    )
    return dict(line.split("=", 1) for line in text.splitlines())


def decision(service, report_exists, progress=None):
    if service["ActiveState"] in ("active", "activating", "deactivating") or int(
        service["MainPID"]
    ):
        if (
            service["ActiveState"] == "active"
            and progress
            and any(
                lane.get("stage") == "held_error" for lane in progress.get("lanes", {}).values()
            )
        ):
            return "recover_failed_lane"
        return "monitor"
    if report_exists:
        return "verify_completion"
    return "start_continuation"


def publish_completion(campaign, audit):
    tracking.configure_local_directories()
    output = campaign / "durable/final_wandb"
    saved = read_json(output / "run.json") if (output / "run.json").exists() else None
    with tracking.start_run(
        "controlled-v2-independent-final-audit",
        {
            "durable_manifest_sha256": digest(campaign / "durable/manifest.json"),
            "report_completion_sha256": audit["report_completion_sha256"],
        },
        output,
        job_type="preparation",
        run_id=saved["id"] if saved else None,
        resume=bool(saved),
    ) as run:
        write_json(output / "run.json", {"id": run.id, "url": run.url})
        run.summary.update(audit)
        files = {
            "independent_audit.json": campaign / "durable/final_audit.json",
            "manifest.json": campaign / "durable/manifest.json",
            "README_ko.md": campaign / "durable/README_ko.md",
            "source.zip": campaign / "durable/source.zip",
            "final_summary.json": campaign / "report/summary.json",
            "results.csv": campaign / "report/results.csv",
            "REPORT_ko.md": campaign / "report/REPORT_ko.md",
            "report_completion.json": campaign / "report/completion.json",
        }
        tracking.log_artifact(run, "controlled-v2-independent-final-audit", "validation", files)
        url = run.url
    return url


def complete(campaign, service):
    verify_durable(campaign)
    # Both locks are also held by every writer. No audit can race a new recovery writer.
    original = Path(read_json(campaign / "revision.json")["original_campaign"])
    with (
        (original / "writer.lock").open("a") as old_lock,
        (campaign / "writer.lock").open("a") as lock,
    ):
        for stream in (old_lock, lock):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = service_state()
        if decision(current, True) != "verify_completion":
            raise RuntimeError("Writer restarted before final audit")
        group = current.get("ControlGroup")
        if group:
            group_path = Path("/sys/fs/cgroup") / group.lstrip("/")
            if any(p.read_text().strip() for p in group_path.rglob("cgroup.procs")):
                raise RuntimeError("Owned service descendants have not released")
        audit = {
            **verify_report(campaign),
            "at_utc": utc_now(),
            "service": service,
            "hardware_after_writer_exit": snapshot(),
        }
        write_json(campaign / "durable/final_audit.json", audit)
        publication = publish_completion(campaign, audit)
        result = {
            **audit,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "independent_audit_wandb": publication,
            "durable_manifest_sha256": digest(campaign / "durable/manifest.json"),
        }
        write_json(campaign / "durable/completion.json", result)
        state_path = campaign.parent / "CURRENT_STATE.json"
        state = read_json(state_path)
        state.update(
            status="GPU_CAMPAIGN_COMPLETE",
            full_training="complete",
            independent_completion=str(campaign / "durable/completion.json"),
            final_report_run=audit["report_run_url"],
            independent_audit_run=publication,
            updated_at_utc=utc_now(),
        )
        write_json(state_path, state)
    # Completed campaigns should not start again on a future login/reboot.
    subprocess.run(["systemctl", "--user", "disable", SERVICE, WATCH_SERVICE], check=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    campaign = args.campaign.resolve()
    verify_durable(campaign)
    with (campaign / "durable/supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                service = service_state()
                progress = read_json(campaign / "progress.json")
                action = decision(service, (campaign / "report/completion.json").exists(), progress)
                state = {
                    "at_utc": utc_now(),
                    "status": action,
                    "service": service,
                    "campaign_status": progress["status"],
                    "completed_training": len(progress["completed_training"]),
                    "completed_evaluations": len(progress["completed_evaluations"]),
                    "lanes": {
                        g: {k: v.get(k) for k in ("name", "last_epoch", "epochs", "stage", "retry")}
                        for g, v in progress["lanes"].items()
                    },
                }
                write_json(campaign / "durable/status.json", state)
                if action == "verify_completion":
                    result = complete(campaign, service)
                    write_json(campaign / "durable/status.json", result)
                    return
                if action == "start_continuation":
                    verify_durable(campaign)
                    subprocess.run(
                        ["systemctl", "--user", "start", SERVICE], check=True, timeout=30
                    )
                if action == "recover_failed_lane":
                    verify_durable(campaign)
                    write_json(
                        campaign / "durable/legacy_handoffs" / f"{time.time_ns()}.json",
                        {
                            "at_utc": utc_now(),
                            "service": service,
                            "progress": progress,
                            "reason": "Failed lane; resume the service from saved checkpoints",
                        },
                    )
                    subprocess.run(
                        ["systemctl", "--user", "restart", SERVICE], check=True, timeout=90
                    )
                if args.once:
                    print(json.dumps(state), flush=True)
                    return
            except Exception as exc:
                write_json(
                    campaign / "durable/status.json",
                    {
                        "at_utc": utc_now(),
                        "status": "supervisor_retry",
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                if args.once:
                    raise
            time.sleep(20)


if __name__ == "__main__":
    main()
