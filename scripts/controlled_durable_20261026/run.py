"""One restartable writer; completed stages are reused and interrupted stages resume."""

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.controlled_campaign.run import Campaign, verify_evaluation, verify_profile
from scripts.controlled_durable_20261026.recovery import (
    archive_incomplete,
    prepare_training,
    verify_durable,
    verify_report,
    verify_training,
    wait_seconds,
)
from scripts.controlled_geometry.common import ROOT, read_json, write_json


class DurableCampaign(Campaign):
    def __init__(self, directory):
        super().__init__(directory)
        verify_durable(directory)
        self.state.update(runtime_amendment="durable-continuation-20261026", recovery_events=0)

    def arguments(self, task, arguments):
        return [
            "scripts.controlled_durable_20261026.worker",
            "--continuation",
            str(self.directory),
            "--task",
            task,
            *arguments,
        ]

    def command(self, lane, label, arguments):
        verify_durable(self.directory)
        return super().command(lane, label + f"-attempt-{time.time_ns()}", arguments)

    def retry(self, label, function, lane=None):
        failures = 0
        while True:
            try:
                result = function()
                with self.mutex:
                    if lane is not None:
                        self.state["lanes"][str(lane["gpu_index"])].pop("retry", None)
                return result
            except Exception as exc:
                failures += 1
                delay = wait_seconds(failures)
                error = {
                    "label": label,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "retry_in_seconds": delay,
                }
                self.event({"type": "durable_retry", **error})
                with self.mutex:
                    self.state["recovery_events"] += 1
                    if lane is not None:
                        self.state["lanes"][str(lane["gpu_index"])]["retry"] = error
                    else:
                        self.state["final_stage_retry"] = error
                self.publish()
                time.sleep(delay)

    def train(self, lane, path, config):
        output = Path(config["output"])
        if (output / "completion.json").exists():
            return verify_training(self.directory, config)
        resume, epoch = prepare_training(self.directory, config, path)
        with self.mutex:
            self.state["lanes"][str(lane["gpu_index"])]["resumed_from_epoch"] = epoch
        self.command(
            lane,
            config["name"] + "-train",
            self.arguments(
                "train",
                [
                    "--config",
                    path,
                    "--execute",
                    "--device",
                    "cuda:0",
                    *(["--resume"] if resume else []),
                ],
            ),
        )
        return verify_training(self.directory, config)

    def evaluate(self, lane, config, dataset):
        output = self.directory / "evaluations" / config["name"] / dataset
        if (output / "completion.json").exists():
            return verify_evaluation(output, config, dataset)
        archive_incomplete(self.directory, output, config["name"] + "-" + dataset)
        self.command(
            lane,
            config["name"] + "-" + dataset,
            self.arguments(
                "evaluate",
                [
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
            ),
        )
        return verify_evaluation(output, config, dataset)

    def lane(self, lane):
        for path in lane["configs"]:
            config = read_json(path)
            name = config["name"]
            with self.mutex:
                self.state["lanes"][str(lane["gpu_index"])] = {
                    "name": name,
                    "seed": config["seed"],
                    "epochs": config["epochs"],
                    "training_output": config["output"],
                    "stage": "checking_existing_outputs",
                }
            self.retry(name + "-train", lambda: self.train(lane, path, config), lane)
            with self.mutex:
                self.state["completed_training"].append(name)
            for dataset in ("benchmark", "validation"):
                self.retry(name + "-" + dataset, lambda: self.evaluate(lane, config, dataset), lane)
                with self.mutex:
                    self.state["completed_evaluations"].append(name + "/" + dataset)
            self.publish()
        with self.mutex:
            self.state["lanes"][str(lane["gpu_index"])]["stage"] = (
                "training_and_evaluation_complete"
            )
        self.publish()

    def latency(self, lane):
        config = read_json(lane["configs"][0])
        output = self.directory / "latency" / f"gpu{lane['gpu_index']}"
        if (output / "completion.json").exists():
            result = verify_profile(output)
            from scripts.controlled_geometry.common import digest

            if result["checkpoint_sha256"] != digest(Path(config["output"]) / "best.pt"):
                raise ValueError("Completed latency used a different selected checkpoint")
            return result
        archive_incomplete(self.directory, output, f"latency-gpu{lane['gpu_index']}")
        self.command(
            lane,
            config["name"] + "-latency",
            self.arguments(
                "profile",
                [
                    "--config",
                    lane["configs"][0],
                    "--checkpoint",
                    str(Path(config["output"]) / "best.pt"),
                    "--output",
                    str(output),
                    "--gpu-index",
                    str(lane["gpu_index"]),
                    "--gpu-uuid",
                    lane["gpu_uuid"],
                ],
            ),
        )
        return verify_profile(output)

    def report(self):
        output = self.directory / "report"
        if not (output / "completion.json").exists():
            archive_incomplete(self.directory, output, "final-report")
            log = self.directory / "logs" / f"durable-report-{time.time_ns()}.log"
            with log.open("x") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        *self.arguments("report", ["--campaign", str(self.directory)]),
                    ],
                    cwd=ROOT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        return verify_report(self.directory)

    def execute(self):
        if (self.directory / "report/completion.json").exists():
            self.retry("final-report-verification", lambda: verify_report(self.directory))
            self.state["status"] = "GPU_CAMPAIGN_COMPLETE"
            self.state["completed_training"] = [
                read_json(p)["name"] for lane in self.auth["lanes"] for p in lane["configs"]
            ]
            self.state["completed_evaluations"] = [
                n + "/" + d
                for n in self.state["completed_training"]
                for d in ("benchmark", "validation")
            ]
            self.publish()
            return 0
        self.state["status"] = "GPU_CAMPAIGN_RUNNING"
        self.publish()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.lane, self.auth["lanes"]))
        self.state["status"] = "GPU_CAMPAIGN_LATENCY"
        self.publish()
        for lane in self.auth["lanes"]:
            self.retry(f"latency-gpu{lane['gpu_index']}", lambda: self.latency(lane), lane)
        self.state["status"] = "GPU_CAMPAIGN_REPORTING"
        self.publish()
        verified = self.retry("final-report", self.report)
        write_json(self.directory / "durable/writer_completion.json", verified)
        self.state.pop("final_stage_retry", None)
        self.state["status"] = "GPU_CAMPAIGN_COMPLETE"
        self.publish()
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    directory = args.campaign.resolve()
    verify_durable(directory)
    original = Path(read_json(directory / "revision.json")["original_campaign"])
    with (
        (original / "writer.lock").open("a") as old_lock,
        (directory / "writer.lock").open("a") as lock,
    ):
        for stream in (old_lock, lock):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "progress.json").exists():
            snapshot = directory / "durable/previous_progress" / f"{time.time_ns()}.json"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory / "progress.json", snapshot)
        raise SystemExit(DurableCampaign(directory).execute())


if __name__ == "__main__":
    main()
