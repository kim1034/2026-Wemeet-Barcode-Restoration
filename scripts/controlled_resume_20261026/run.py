"""Continue only the four reviewed epoch4 checkpoints, then the eight unstarted runs."""

import argparse
import fcntl
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.controlled_campaign.run import Campaign, verify_evaluation, verify_profile
from scripts.controlled_geometry.common import ROOT, read_json
from scripts.controlled_resume_20261026.runtime import (
    validate_resume,
    verify_revision,
    verify_training,
)


class ResumeCampaign(Campaign):
    def __init__(self, directory):
        super().__init__(directory)
        self.revision = verify_revision(directory)
        if self.revision["status"] != "validated_for_epoch4_resume":
            raise ValueError("Resume revision is not validated")
        original = read_json(Path(self.revision["original_campaign"]) / "progress.json")
        if original["status"] != "GPU_CAMPAIGN_INCOMPLETE" or len(original["errors"]) != 4:
            raise ValueError("Original campaign is not the reviewed four-run interruption")
        if set(self.revision["resume_runs"]) != {
            read_json(lane["configs"][0])["name"] for lane in self.auth["lanes"]
        }:
            raise ValueError("Unexpected resume population")
        for lane in self.auth["lanes"]:
            config = read_json(lane["configs"][0])
            validate_resume(config, self.revision["resume_runs"][config["name"]])
            for path in lane["configs"][1:]:
                if Path(read_json(path)["output"]).exists():
                    raise ValueError("Previously unstarted seed43/44 output exists")
        self.state.update(
            original_campaign=self.revision["original_campaign"],
            runtime_amendment=self.revision["id"],
            recovery="Four seed42 runs resume from saved epoch4; original failures preserved",
        )

    def arguments(self, task, arguments):
        return [
            "scripts.controlled_resume_20261026.runtime",
            "--amendment",
            str(self.directory),
            "--task",
            task,
            *arguments,
        ]

    def command(self, lane, label, arguments):
        verify_revision(self.directory)
        return super().command(lane, label, arguments)

    def lane(self, lane):
        try:
            for path in lane["configs"]:
                config = read_json(path)
                name = config["name"]
                resume = config["seed"] == 42
                with self.mutex:
                    self.state["lanes"][str(lane["gpu_index"])] = {
                        "name": name,
                        "seed": config["seed"],
                        "epochs": config["epochs"],
                        "training_output": config["output"],
                        "resumed_from_epoch": 4 if resume else 0,
                    }
                if resume:
                    validate_resume(config, self.revision["resume_runs"][name])
                elif Path(config["output"]).exists():
                    raise FileExistsError(config["output"])
                self.command(
                    lane,
                    name + ("-resume-epoch4" if resume else "-train"),
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
                verify_training(config)
                with self.mutex:
                    self.state["completed_training"].append(name)
                for dataset in ("benchmark", "validation"):
                    output = self.directory / "evaluations" / name / dataset
                    self.command(
                        lane,
                        name + "-" + dataset,
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
        for lane in self.auth["lanes"]:
            profile = verify_profile(self.directory / "profiles" / f"gpu{lane['gpu_index']}")
            if (
                profile["config"] != read_json(lane["configs"][0])
                or profile["hardware_before"]["gpu_uuid"] != lane["gpu_uuid"]
            ):
                raise ValueError("Profile/assigned training config mismatch")
        self.state["status"] = "GPU_CAMPAIGN_RUNNING"
        self.publish()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.lane, self.auth["lanes"]))
        if self.state["errors"]:
            self.state["status"] = "GPU_CAMPAIGN_INCOMPLETE"
            self.publish()
            return 1
        self.state["status"] = "GPU_CAMPAIGN_LATENCY"
        self.publish()
        for lane in self.auth["lanes"]:
            config = read_json(lane["configs"][0])
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
                        str(self.directory / "latency" / f"gpu{lane['gpu_index']}"),
                        "--gpu-index",
                        str(lane["gpu_index"]),
                        "--gpu-uuid",
                        lane["gpu_uuid"],
                    ],
                ),
            )
        self.state["status"] = "GPU_CAMPAIGN_REPORTING"
        self.publish()
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                *self.arguments("report", ["--campaign", str(self.directory)]),
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
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    directory = args.campaign.resolve()
    manifest = verify_revision(directory)
    original = Path(manifest["original_campaign"])
    # The original and continuation share one ownership lock; never admit two writers.
    with (
        (original / "writer.lock").open("a") as original_lock,
        (directory / "writer.lock").open("a") as lock,
    ):
        for stream in (original_lock, lock):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "progress.json").exists():
            raise RuntimeError("Continuation already started; inspect before another action")
        campaign = ResumeCampaign(directory)
        if args.validate_only:
            print("Validated four epoch4 checkpoints, eight fresh runs, inputs and correction.")
            return
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
