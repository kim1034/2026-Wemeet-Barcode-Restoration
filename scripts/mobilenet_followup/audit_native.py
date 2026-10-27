"""CPU-only full scene audit using recorded detector and model observations."""

import argparse
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.controlled_geometry.common import configure, digest, read_json, write_json
from scripts.evaluate_stage2_v1 import field_from
from scripts.mobilenet_followup.controller import audit_job, task_matrix
from scripts.mobilenet_followup.core import image_hash
from scripts.mobilenet_followup.native_bridge import decoder_mode
from scripts.mobilenet_followup.report import read_records
from scripts.mobilenet_followup.worker import scene_samples
from wemeet.ai.detection.geometry import CROP_PAD, crop_upright, decide_ratio, fit_ratio, upright
from wemeet.ai.geometry.model import destination_grid
from wemeet.schemas import DetectedBarcode
from wemeet.sw import pipeline


def audit(root):
    inputs = read_json(root / "inputs.json")
    samples = {sid: image for sid, _, image in scene_samples(inputs)}
    dst = destination_grid().numpy().astype(np.float64)
    summaries = {}
    for task in task_matrix():
        if task["kind"] != "scene":
            continue
        done = audit_job(root, task)
        assert done is not None
        records = read_records(Path(done["output"]))
        detected = 0
        passthrough = 0
        with decoder_mode(task["enhance"]):
            for r in records:
                image = samples[r["id"]]
                assert image_hash(image) == r["input_sha256"]
                target, field = None, None
                if r["detected"]:
                    cx, cy, w, h, rad = upright(*r["detector_observation"]["xywhr"])
                    ratio, _ = decide_ratio(w, h)
                    w, h = fit_ratio(w, h, ratio)
                    crop, angle = crop_upright(image, cx, cy, w, h, rad, pad=CROP_PAD)
                    assert (
                        image_hash(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                        == r["d0"]["image_sha256"]
                    )
                    assert angle == r["crop_angle_deg"]
                    target = DetectedBarcode(
                        np.ascontiguousarray(crop, dtype=np.uint8), angle, r["detector_confidence"]
                    )
                    field = field_from(np.asarray(r["prediction"], dtype=np.float64), dst)
                    detected += 1
                    passthrough += sum(v["passthrough"] for v in r["warped"])
                with (
                    patch.object(pipeline, "detect", return_value=target),
                    patch.object(pipeline, "estimate_geometry", return_value=field),
                ):
                    result = pipeline.run(image)
                assert not result.decode.degraded
                assert result.decode.text == r["decisions"]["early_exit"]["text"], r["id"]
        summaries[task["id"]] = {
            "n": len(records),
            "detected_crop_pixel_matches": detected,
            "native_pipeline_text_matches": len(records),
            "warp_passthrough_count": passthrough,
            "source_completion_sha256": digest(root / "jobs" / task["id"] / "completion.json"),
        }
    assert len(summaries) == 8
    return {"status": "passed", "gpu_inference": False, "tasks": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    configure(42, threads=1, device="cpu")
    result = audit(args.root.resolve())
    write_json(args.root / "delivery/native_pipeline_audit.json", result)
    print({"status": result["status"], "tasks": len(result["tasks"]), "native_matches": 800})
