"""Exercise the original report and the independent audit with a small source matrix."""

import json
import sys

import pytest

from scripts.controlled_campaign import report
from scripts.controlled_campaign import run as original_run
from scripts.controlled_durable_20261026 import recovery
from scripts.controlled_geometry.common import digest, read_json, write_json
from scripts.controlled_geometry.evaluation import summarize


@pytest.mark.parametrize("fault", [None, "aggregate", "csv", "missing_hash"])
def test_original_report_and_independent_aggregation_check(tmp_path, monkeypatch, fault):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    auth = {"protocol_sha256": "fixture-protocol", "lanes": []}
    for index, model, preset in (
        (4, "resnet18-c3", "R"),
        (5, "resnet18-c3", "M"),
        (6, "mobilenetv3-small", "R"),
        (7, "mobilenetv3-small", "M"),
    ):
        lane = {"gpu_index": index, "configs": []}
        for seed in (42, 43, 44):
            config = {
                "model": model,
                "preset": preset,
                "seed": seed,
                "name": f"{model}-{preset}-{seed}",
            }
            path = campaign / "configs" / (config["name"] + ".json")
            write_json(path, config)
            lane["configs"].append(str(path))
            for dataset in ("benchmark", "validation"):
                output = campaign / "evaluations" / config["name"] / dataset
                records = []
                for number in range(4):
                    status = "C" if (number + seed + index) % 3 else "U"
                    records.append(
                        {
                            "id": str(number),
                            "gt_text": str(number),
                            "group_id_proxy": str(number),
                            "gt_src_norm": [[0.0, 0.0]] * 48,
                            "crop_hw": [20, 40],
                            "bucket": "low",
                            "band": "fixture",
                            "preset": "fixture",
                            "d0_status": "U",
                            "geometry_valid": True,
                            "error_xy_px": [[1.0, 1.0]] * 48,
                            "monotonicity_violations": 0,
                            "paths": {
                                p: {
                                    "status": "U" if p == "d0" else status,
                                    "attempts": [],
                                    "final": {"text": None, "format": None},
                                }
                                for p in ("d0", "crop_forced_model", "crop_policy_model")
                            },
                        }
                    )
                write_json(output / "results/summary.json", summarize(records))
                (output / "results/per_sample.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in records)
                )
                write_json(
                    output / "completion.json", {"run_url": "https://example.test/evaluation"}
                )
        first = read_json(lane["configs"][0])
        write_json(
            campaign / "latency" / f"gpu{index}/result.json",
            {"config": first, "checkpoint_sha256": first["name"]},
        )
        auth["lanes"].append(lane)
    write_json(campaign / "authorization.json", auth)
    write_json(campaign / "operations.json", {})
    write_json(tmp_path / "preparation/preservation_before.json", {"files": {}})

    def trained(config):
        return {
            "selected/epoch": 5,
            "selected/checkpoint_sha256": config["name"],
            "run_url": "https://example.test/training",
        }

    for module in (report, original_run):
        monkeypatch.setattr(module, "validate_authorization", lambda _: auth)
        monkeypatch.setattr(module, "verify_profile", lambda p: read_json(p / "result.json"))
        monkeypatch.setattr(
            module, "verify_evaluation", lambda p, *a: read_json(p / "completion.json")
        )
    monkeypatch.setattr(report, "verify_operations", lambda _: None)
    monkeypatch.setattr(report, "verify_training", trained)
    monkeypatch.setattr(recovery, "verify_training", lambda _, cfg: trained(cfg))

    class FakeRun:
        summary = {}
        url = "https://example.test/report"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def log(self, value):
            pass

    monkeypatch.setattr(report, "start_run", lambda *a, **k: FakeRun())
    monkeypatch.setattr(report, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["report", "--campaign", str(campaign)])
    report.main()
    output = campaign / "report"
    done = read_json(output / "completion.json")
    if fault == "aggregate":
        summary = read_json(output / "summary.json")
        summary["aggregates"][0]["EDR"]["mean"] += 1
        write_json(output / "summary.json", summary)
        done["files"]["summary.json"] = digest(output / "summary.json")
    elif fault == "csv":
        csv_path = output / "results.csv"
        csv_path.write_text(csv_path.read_text().replace("resnet18-c3", "wrong-model", 1))
        done["files"]["results.csv"] = digest(csv_path)
    elif fault == "missing_hash":
        del done["files"]["REPORT_ko.md"]
    write_json(output / "completion.json", done)
    if fault:
        with pytest.raises(recovery.IntegrityError):
            recovery.verify_report(campaign)
    else:
        checked = recovery.verify_report(campaign)
        assert checked["training_runs"] == 12 and checked["paired_comparisons"] == 12
