"""One durable writer: four independent GPU lanes, then comparisons and a report."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.controlled_campaign.hardware import assigned_gpu, idle, snapshot, utc_now
from scripts.controlled_geometry.common import (
    ROOT,
    digest,
    read_json,
    selection_key,
    validate_lock,
    write_json,
)


def verify_operations(campaign):
    manifest = read_json(campaign / "operations.json")
    for relative, expected in manifest["files"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Execution source changed: {relative}")


def validate_authorization(campaign):
    auth = read_json(campaign / "authorization.json")
    if digest(auth["protocol"]) != auth["protocol_sha256"]:
        raise ValueError("Protocol changed")
    validate_lock(auth["protocol"])
    expected = [
        (4, "resnet18-c3", "R"),
        (5, "resnet18-c3", "M"),
        (6, "mobilenetv3-small", "R"),
        (7, "mobilenetv3-small", "M"),
    ]
    actual = [(x["gpu_index"], x["model"], x["preset"]) for x in auth["lanes"]]
    if actual != expected or len({x["gpu_uuid"] for x in auth["lanes"]}) != 4:
        raise ValueError("Unexpected GPU assignment")
    names = []
    for lane in auth["lanes"]:
        if len(lane["configs"]) != 3:
            raise ValueError("Three seeds required per lane")
        for seed, path in zip((42, 43, 44), lane["configs"], strict=True):
            cfg = read_json(path)
            if digest(path) != auth["config_sha256"][path] or (
                cfg["seed"],
                cfg["model"],
                cfg["preset"],
            ) != (seed, lane["model"], lane["preset"]):
                raise ValueError("Run matrix changed")
            names.append(cfg["name"])
    if len(set(names)) != 12:
        raise ValueError("Duplicate run in matrix")
    return auth


def verify_profile(directory):
    done = read_json(directory / "completion.json")
    if done["status"] != "passed" or done["result_sha256"] != digest(directory / "result.json"):
        raise ValueError("Incomplete or changed GPU profile")
    result = read_json(directory / "result.json")
    if result["status"] != "passed":
        raise ValueError("GPU profile failed")
    return result


def verify_training(config):
    import torch

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
        epoch: read_json(output / f"validation/epoch-{epoch:04d}/summary.json")
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


def verify_evaluation(directory, config, dataset):
    from scripts.controlled_geometry.evaluation import summarize

    done = read_json(directory / "completion.json")
    metadata = read_json(directory / "run_config.json")
    expected_n = 1500 if dataset == "benchmark" else 2976
    required = {
        "run_config.json",
        "results/per_sample.jsonl",
        "results/summary.json",
        "results/hard_ids.json",
        "results/outcomes.csv",
    }
    if (
        done["status"] != "evaluated"
        or done["n"] != expected_n
        or not required <= set(done["files"])
    ):
        raise ValueError("Incomplete evaluation population/files")
    for key in ("name", "model", "preset", "seed", "protocol_lock_sha256"):
        if metadata[key] != config[key]:
            raise ValueError("Evaluation config mismatch")
    if metadata["dataset"] != dataset or metadata["checkpoint_sha256"] != digest(
        Path(config["output"]) / "best.pt"
    ):
        raise ValueError("Evaluation checkpoint mismatch")
    for relative, expected in done["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError("Evaluation output changed")
    records = [
        json.loads(line)
        for line in (directory / "results/per_sample.jsonl").read_text().splitlines()
    ]
    reference = read_json(read_json(config["protocol_lock"])["d0_manifest"])[dataset]
    check_population(records, reference)
    if len(records) != expected_n or summarize(records) != read_json(
        directory / "results/summary.json"
    ):
        raise ValueError("Raw results do not reproduce the complete evaluation summary")
    return done


def check_population(records, reference):
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(reference):
        raise ValueError("Evaluation sample IDs do not match the frozen population")
    for row in records:
        expected = reference[row["id"]]
        if (row["gt_text"], row["d0_status"]) != (expected["gt_text"], expected["d0_status"]):
            raise ValueError("Evaluation ground truth/D0 differs from the input freeze")


class Campaign:
    def __init__(self, directory):
        self.directory = directory
        self.auth = validate_authorization(directory)
        verify_operations(directory)
        self.mutex = threading.RLock()
        self.state = {
            "status": "STARTING",
            "started_at_utc": utc_now(),
            "lanes": {},
            "completed_training": [],
            "completed_evaluations": [],
            "errors": [],
        }
        self.prepared_state = read_json(directory / "cpu_preparation_state.json")
        self.parent_pid = os.getpid()

    def publish(self):
        with self.mutex:
            self.state["updated_at_utc"] = utc_now()
            for lane in self.state["lanes"].values():
                output = lane.get("training_output")
                if output:
                    path = Path(output)
                    if (path / "run.json").exists():
                        lane["wandb_train"] = read_json(path / "run.json")["url"]
                    if (path / "metrics.jsonl").exists():
                        rows = [
                            json.loads(x) for x in (path / "metrics.jsonl").read_text().splitlines()
                        ]
                        if rows:
                            lane["last_epoch"] = rows[-1]["epoch"]
                            lane["last_epoch_seconds"] = rows[-1]["epoch_seconds"]
                            lane["recent_epoch_mean_seconds"] = sum(
                                x["epoch_seconds"] for x in rows[-10:]
                            ) / len(rows[-10:])
            write_json(self.directory / "progress.json", self.state)
            write_json(
                self.directory.parent / "CURRENT_STATE.json",
                {
                    **self.prepared_state,
                    "status": self.state["status"],
                    "updated_at_utc": self.state["updated_at_utc"],
                    "scope": self.auth["scope"],
                    "campaign": str(self.directory),
                    "gpu_profile": "passed",
                    "full_training": "complete"
                    if len(self.state["completed_training"]) == 12
                    else "in_progress",
                    "automatic_training_service_created": True,
                    "gpu_jobs_started": True,
                    "cuda_initialized": True,
                    "progress": self.state,
                },
            )

    def event(self, value):
        with self.mutex:
            with (self.directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"at_utc": utc_now(), **value}, allow_nan=False) + "\n")

    def command(self, lane, label, arguments):
        verify_operations(self.directory)
        index, uuid = lane["gpu_index"], lane["gpu_uuid"]
        while True:
            observed = snapshot()
            if idle(observed, index, uuid):
                break
            self.event(
                {
                    "type": "resource_wait",
                    "gpu_index": index,
                    "label": label,
                    "observation": observed,
                }
            )
            time.sleep(10)
        if shutil.disk_usage(self.directory).free < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB free for checkpoints and evaluation records")
        with (self.directory / "logs" / f"{label}.log").open("x") as stream:
            child = subprocess.Popen(
                [sys.executable, "-u", "-m", *arguments],
                cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": uuid},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self.mutex:
                current = self.state["lanes"].setdefault(str(index), {})
                current.update(
                    pid=child.pid,
                    gpu_index=index,
                    gpu_uuid=uuid,
                    stage=label,
                    actual_gpu_observed=False,
                )
            self.event(
                {
                    "type": "launch",
                    "label": label,
                    "gpu_index": index,
                    "gpu_uuid": uuid,
                    "pid": child.pid,
                    "command": arguments,
                }
            )
            seen = False
            try:
                while child.poll() is None:
                    try:
                        observed = snapshot()
                    except subprocess.SubprocessError as exc:
                        self.event(
                            {
                                "type": "hardware_observation_error",
                                "label": label,
                                "error": type(exc).__name__,
                            }
                        )
                        time.sleep(5)
                        continue
                    assigned_gpu(observed, index, uuid)
                    own = [p for p in observed["processes"] if int(p["pid"]) == child.pid]
                    if own:
                        if any(p["gpu_uuid"] != uuid for p in own):
                            raise RuntimeError("Child used an unauthorized GPU")
                        seen = True
                    with self.mutex:
                        self.state["lanes"][str(index)]["actual_gpu_observed"] = seen
                    self.event(
                        {
                            "type": "hardware",
                            "label": label,
                            "pid": child.pid,
                            "observation": observed,
                        }
                    )
                    self.publish()
                    time.sleep(5)
            finally:
                if child.poll() is None:
                    # This group was created by this writer; never signal unrelated jobs.
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
            code = child.wait()
        self.event(
            {
                "type": "exit",
                "label": label,
                "pid": child.pid,
                "exit_code": code,
                "gpu_observed": seen,
            }
        )
        with self.mutex:
            self.state["lanes"][str(index)]["pid"] = None
        if code or not seen:
            raise RuntimeError(
                f"{label}: exit={code}, actual CUDA observation={seen}; preserve outputs"
            )

    def lane(self, lane):
        try:
            for path in lane["configs"]:
                config = read_json(path)
                name = config["name"]
                with self.mutex:
                    self.state["lanes"][str(lane["gpu_index"])] = {
                        "name": name,
                        "seed": config["seed"],
                        "epochs": config["epochs"],
                        "training_output": config["output"],
                    }
                # Never silently rerun or overwrite an existing training directory.
                if Path(config["output"]).exists():
                    raise FileExistsError(config["output"])
                self.command(
                    lane,
                    name + "-train",
                    [
                        "scripts.controlled_geometry.train",
                        "--config",
                        path,
                        "--execute",
                        "--device",
                        "cuda:0",
                    ],
                )
                verify_training(config)
                with self.mutex:
                    self.state["completed_training"].append(name)
                for dataset in ("benchmark", "validation"):
                    output = self.directory / "evaluations" / name / dataset
                    self.command(
                        lane,
                        name + "-" + dataset,
                        [
                            "scripts.controlled_geometry.evaluate",
                            "--checkpoint",
                            str(Path(config["output"]) / "best.pt"),
                            "--output",
                            str(output),
                            "--dataset",
                            dataset,
                            "--execute",
                            "--device",
                            "cuda:0",
                        ],
                    )
                    verify_evaluation(output, config, dataset)
                    with self.mutex:
                        self.state["completed_evaluations"].append(name + "/" + dataset)
                self.publish()
            with self.mutex:
                self.state["lanes"][str(lane["gpu_index"])]["stage"] = (
                    "training_and_evaluation_complete"
                )
        except Exception as exc:
            error = {
                "gpu_index": lane["gpu_index"],
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            self.event({"type": "error", **error})
            with self.mutex:
                self.state["errors"].append(error)
                self.state["lanes"].setdefault(str(lane["gpu_index"]), {})["stage"] = "held_error"
                self.state["status"] = "GPU_CAMPAIGN_RUNNING_WITH_ERRORS"
        finally:
            self.publish()

    def execute(self):
        cache = read_json(self.directory / "cache.json")
        if (
            cache["status"] != "passed"
            or cache["train_samples"] != 27024
            or cache["protocol_sha256"] != self.auth["protocol_sha256"]
        ):
            raise ValueError("Training cache preflight incomplete")
        profiles = []
        for lane in self.auth["lanes"]:
            profile = verify_profile(self.directory / "profiles" / f"gpu{lane['gpu_index']}")
            if (
                profile["config"] != read_json(lane["configs"][0])
                or profile["hardware_before"]["gpu_uuid"] != lane["gpu_uuid"]
            ):
                raise ValueError("Profile/assigned training config mismatch")
            profiles.append(profile)
        self.state.update(
            status="GPU_CAMPAIGN_RUNNING",
            profile_hours_first_four_estimate=max(x["run_seconds_estimate"] for x in profiles)
            / 3600,
            profile_hours_all_twelve_estimate=3
            * max(x["run_seconds_estimate"] for x in profiles)
            / 3600,
        )
        self.publish()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.lane, self.auth["lanes"]))
        if self.state["errors"]:
            self.state["status"] = "GPU_CAMPAIGN_INCOMPLETE"
            self.publish()
            return 1
        # Latency measurements are serial after all four training lanes release.
        self.state["status"] = "GPU_CAMPAIGN_LATENCY"
        self.publish()
        for lane in self.auth["lanes"]:
            config = read_json(lane["configs"][0])
            self.command(
                lane,
                config["name"] + "-latency",
                [
                    "scripts.controlled_campaign.profile",
                    "--config",
                    lane["configs"][0],
                    "--checkpoint",
                    str(Path(config["output"]) / "best.pt"),
                    "--output",
                    str(self.directory / "latency" / f"gpu{lane['gpu_index']}"),
                    "--gpu-index",
                    str(lane["gpu_index"]),
                    "--gpu-uuid",
                    lane["gpu_uuid"],
                ],
            )
        self.state["status"] = "GPU_CAMPAIGN_REPORTING"
        self.publish()
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.controlled_campaign.report",
                "--campaign",
                str(self.directory),
            ],
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            check=True,
        )
        if read_json(self.directory / "report/completion.json")["status"] != "complete":
            raise ValueError("Final report did not complete")
        self.state["status"] = "GPU_CAMPAIGN_COMPLETE"
        self.publish()
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with (args.campaign / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.campaign / "progress.json").exists():
            raise RuntimeError(
                "Campaign already started; inspect preserved progress before any continuation"
            )
        campaign = Campaign(args.campaign.resolve())
        try:
            code = campaign.execute()
        except Exception as exc:
            campaign.state["status"] = "GPU_CAMPAIGN_INCOMPLETE"
            campaign.state["errors"].append(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            campaign.publish()
            raise
        raise SystemExit(code)


if __name__ == "__main__":
    main()
