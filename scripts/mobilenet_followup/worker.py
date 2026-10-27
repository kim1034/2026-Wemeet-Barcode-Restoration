"""Audited evaluation or serial timing for a single frozen task."""

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import wandb

from scripts.controlled_campaign.hardware import prove_own_cuda, snapshot
from scripts.controlled_geometry.common import configure, digest, read_json, write_json
from scripts.controlled_geometry.data import (
    RenderCache,
    benchmark_samples,
    load_split,
    validation_samples,
)
from scripts.mobilenet_followup.core import (
    POLICIES,
    Predictor,
    audit_crop,
    counts,
    decode_view,
    decoder_mode,
    empty,
    execute_crop,
    image_hash,
    plain,
    status,
    target_from_gray,
)
from wemeet.ai import detection
from wemeet.ai.detection import yolo_obb
from wemeet.sw import pipeline


def crop_samples(inputs, dataset):
    if dataset == "benchmark":
        return list(benchmark_samples(inputs["crop_data_root"]))
    split = load_split(inputs["split_manifest"])
    cache = RenderCache(inputs["cache_root"], inputs["renderer_fingerprint"])
    return list(validation_samples(split["validation"], cache))


def scene_samples(inputs):
    root = Path(inputs["scene_root"])
    rows = [json.loads(x) for x in (root / "manifest.jsonl").read_text().splitlines()][1:]
    result = []
    for row in rows:
        path = root / row["file"]
        if digest(path) != inputs["scene_files"][str(path)]:
            raise ValueError("Scene image hash mismatch")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Unreadable scene")
        result.append((str(row["id"]), row, image))
    return result


def load_detector(inputs, size):
    if digest(inputs["detector"]["path"]) != inputs["detector"]["sha256"]:
        raise ValueError("Detector hash mismatch")
    os.environ["WEMEET_DETECTION_WEIGHTS"] = inputs["detector"]["path"]
    detection.DEFAULT_IMGSZ = size
    model = yolo_obb.load_model()
    model.to("cuda:0")
    model.overrides.update(device="cuda:0", half=False)
    return model


def summarized(records):
    names = list(records[0]["statuses"])

    def stats(rows):
        hard = [r for r in rows if r["statuses"]["d0"] != "C"]
        paths = {}
        for p in names:
            paths[p] = counts([r["statuses"][p] for r in rows])
            paths[p]["RR_H"] = (
                100 * sum(r["statuses"][p] == "C" for r in hard) / len(hard) if hard else None
            )
            paths[p]["H"] = len(hard)
            paths[p]["transitions"] = {
                a + "->" + b: sum(r["statuses"]["d0"] == a and r["statuses"][p] == b for r in rows)
                for a in ("C", "W", "U")
                for b in ("C", "W", "U")
            }
        return paths

    result = {"n": len(records), "paths": stats(records)}
    for key in ("bucket", "band", "preset", "aspect_family", "tilt_band"):
        groups = sorted({str(r.get(key, "unknown")) for r in records})
        result["by_" + key] = {
            g: stats([r for r in records if str(r.get(key, "unknown")) == g]) for g in groups
        }
    result["failures"] = {}
    for p in POLICIES:
        result["failures"][p] = {
            "not_detected": sum(not r.get("detected", True) for r in records),
            "wrong_early_return": sum(
                r["statuses"][p] == "W" and r["statuses"]["d0"] == "W" for r in records
            ),
            "new_wrong_after_d0_none": sum(
                r["statuses"][p] == "W" and r["statuses"]["d0"] == "U" for r in records
            ),
            "decode_no_return_after_detection": sum(
                r.get("detected", True) and r["statuses"][p] == "U" for r in records
            ),
        }
    errors = [r["error_xy_px"] for r in records if "error_xy_px" in r]
    if errors:
        values = np.linalg.norm(np.asarray(errors), axis=2).ravel()
        result["geometry"] = {
            "epe_mean_px": float(values.mean()),
            "epe_p95_px": float(np.quantile(values, 0.95)),
        }
    return result


def historical_records(inputs, task):
    p = (
        Path(inputs["prior_campaign"])
        / "evaluations"
        / "restoration-mobilenetv3-small-presetM-s42-v2"
        / task["dataset"]
        / "results/per_sample.jsonl"
    )
    return {r["id"]: r for r in (json.loads(x) for x in p.read_text().splitlines())}


def evaluate(inputs, task, predictor, out, gpu):
    records = []
    source = (
        crop_samples(inputs, task["dataset"]) if task["kind"] == "crop" else scene_samples(inputs)
    )
    old = historical_records(inputs, task) if task["kind"] == "crop" else None
    native_checks = 0
    max_delta = 0.0
    detector = load_detector(inputs, task["imgsz"]) if task["kind"] == "scene" else None
    trace_path = out / "per_sample.jsonl"
    if trace_path.exists():
        raise ValueError("Never overwrite an incomplete evaluation trace")
    with decoder_mode(task["enhance"]), trace_path.open("x") as stream:
        for index, sample in enumerate(source):
            sid, row, image = sample[:3]
            truth = row["text"]
            item = {
                "id": sid,
                "gt_text": truth,
                "input_sha256": image_hash(image),
                **{k: row.get(k, "unknown") for k in ("bucket", "band", "preset", "aspect_family")},
            }
            if task["kind"] == "scene":
                lower = min(int(abs(float(row.get("tilt_deg", 0))) // 5), 2) * 5
                item["tilt_band"] = f"{lower}-{lower + 5}deg"
                full_target = target_from_gray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                a = decode_view(plain(full_target), "whole_scene")
                item["whole_scene"] = a
                captured = {}
                original_predict = detector.predict

                def capture(*args, **kwargs):
                    outputs = original_predict(*args, **kwargs)
                    obb = outputs[0].obb
                    if obb is not None and len(obb):
                        best = int(obb.conf.argmax())
                        captured["quad"] = obb.xyxyxyxy[best].cpu().numpy().tolist()
                        captured["xywhr"] = obb.xywhr[best].cpu().numpy().tolist()
                    return outputs

                with patch.object(detector, "predict", capture):
                    target = detection.detect(image)
                item.update(detected=target is not None, detector_observation=captured)
                if captured and "crop_quad" in row:
                    center = tuple(float(v) for v in captured["xywhr"][:2])
                    item["detector_center_in_gt_crop"] = (
                        cv2.pointPolygonTest(np.array(row["crop_quad"], np.float32), center, False)
                        >= 0
                    )
                if target is None:
                    d0 = empty("not_detected")
                    decisions = {p: empty("not_detected") for p in POLICIES}
                    item.update(
                        d0=d0,
                        warped=[],
                        decisions=decisions,
                        statuses={
                            "A": status(a, truth),
                            "d0": "U",
                            "B": "U",
                            "C": "U",
                            **{p: "U" for p in POLICIES},
                        },
                    )
                else:
                    trace = audit_crop(target, predictor)
                    c = trace["d0"] if trace["d0"]["text"] is not None else trace["warped"][0]
                    item.update(trace)
                    item["statuses"] = {
                        "A": status(a, truth),
                        "d0": status(trace["d0"], truth),
                        "B": status(trace["d0"], truth),
                        "C": status(c, truth),
                        **{p: status(d, truth) for p, d in trace["decisions"].items()},
                    }
                    item["crop_hw"] = list(target.crop_bgr_uint8.shape[:2])
                    item["detector_confidence"] = target.confidence
                    item["crop_angle_deg"] = target.angle_deg_ccw
                    if index < 12:
                        with (
                            patch.object(pipeline, "detect", return_value=target),
                            patch.object(pipeline, "estimate_geometry", predictor.field),
                        ):
                            result = pipeline.run(image)
                        if (
                            result.decode.degraded
                            or result.decode.text != trace["decisions"]["early_exit"]["text"]
                        ):
                            raise ValueError(
                                "Observed pipeline disagrees with upstream native execution"
                            )
                        native_checks += 1
            else:
                target = target_from_gray(image)
                trace = audit_crop(target, predictor)
                item.update(trace)
                item["statuses"] = {
                    "d0": status(trace["d0"], truth),
                    **{p: status(d, truth) for p, d in trace["decisions"].items()},
                }
                delta = float(
                    np.max(
                        np.abs(
                            np.asarray(trace["prediction"])
                            - np.asarray(old[sid]["prediction_src_norm"])
                        )
                    )
                )
                max_delta = max(max_delta, delta)
                if delta > 2e-6:
                    raise ValueError(f"Prior prediction numerical mismatch: {sid}: {delta}")
                if not task["enhance"]:
                    for key, old_path in [
                        ("always_rectify", "crop_forced_model"),
                        ("early_exit", "crop_policy_model"),
                    ]:
                        if (item["decisions"][key]["text"], item["decisions"][key]["format"]) != (
                            old[sid]["paths"][old_path]["final"]["text"],
                            old[sid]["paths"][old_path]["final"]["format"],
                        ):
                            raise ValueError(f"Legacy decoder bridge mismatch: {sid}")
                    native_checks += 1
                item["error_xy_px"] = (
                    (np.asarray(trace["prediction"]) - sample[3])
                    * [image.shape[1] - 1, image.shape[0] - 1]
                ).tolist()
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(item)
            if index == 0:
                write_json(
                    out / "hardware_start.json", prove_own_cuda(gpu, inputs["gpu_uuids"][str(gpu)])
                )
            if (index + 1) % 25 == 0:
                write_json(
                    out / "progress.json",
                    {"processed": index + 1, "total": len(source), "at": time.time()},
                )
    if len(records) != len(source) or len({r["id"] for r in records}) != len(records):
        raise ValueError("Missing or duplicate evaluation records")
    summary = summarized(records)
    summary.update(native_parity_checks=native_checks, prior_prediction_max_abs_delta=max_delta)
    return summary


def timing(inputs, task, predictor, out, gpu):
    is_crop = task["kind"] == "crop_timing"
    samples = crop_samples(inputs, "benchmark") if is_crop else scene_samples(inputs)
    if not is_crop:
        load_detector(inputs, task["imgsz"])
    images = [target_from_gray(x[2]) if is_crop else x[2] for x in samples]
    result = {"n": len(images), "warmup": 50, "repeats": 3, "paths": {}}
    with decoder_mode(task["enhance"]):
        for policy in POLICIES:

            def execute(image):
                target = image if is_crop else detection.detect(image)
                return (
                    execute_crop(target, predictor, policy)
                    if target is not None
                    else empty("not_detected")
                )

            for i in range(50):
                execute(images[i % len(images)])
            hardware = prove_own_cuda(gpu, inputs["gpu_uuids"][str(gpu)])
            write_json(out / f"hardware-{policy}-start.json", hardware)
            if any(
                p["gpu_uuid"] in inputs["gpu_uuids"].values() and int(p["pid"]) != os.getpid()
                for p in hardware["observation"]["processes"]
            ):
                raise RuntimeError("Timing requires exclusive follow-up GPUs")
            passes = []
            expected = None
            for repeat in range(3):
                values, decisions = [], []
                for image in images:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    decision = execute(image)
                    torch.cuda.synchronize()
                    values.append((time.perf_counter() - start) * 1000)
                    decisions.append([decision["text"], decision["format"]])
                if expected is None:
                    expected = decisions
                elif expected != decisions:
                    raise RuntimeError("Timing repeats changed decisions")
                passes.append(
                    {
                        "raw_ms": values,
                        "p50_ms": float(np.quantile(values, 0.5)),
                        "p95_ms": float(np.quantile(values, 0.95)),
                    }
                )
                write_json(
                    out / "progress.json",
                    {"policy": policy, "repeat": repeat + 1, "at": time.time()},
                )
            # Check actual timed decisions against corresponding full audited output.
            evaluation = read_json(out.parents[1] / task["accuracy_task"] / "completion.json")
            records_path = Path(evaluation["output"]) / "per_sample.jsonl"
            audited = [json.loads(x) for x in records_path.read_text().splitlines()]
            observed = [
                [r["decisions"][policy]["text"], r["decisions"][policy]["format"]] for r in audited
            ]
            if observed != expected:
                raise RuntimeError("Timing decisions differ from evaluated decisions")
            result["paths"][policy] = {
                "passes": passes,
                "p50_ms": float(np.median([x["p50_ms"] for x in passes])),
                "p95_ms": float(np.median([x["p95_ms"] for x in passes])),
                "decision_parity": True,
            }
            write_json(
                out / f"hardware-{policy}-end.json",
                prove_own_cuda(gpu, inputs["gpu_uuids"][str(gpu)]),
            )
    return result


def publish(task, out, summary):
    run = wandb.init(
        entity="kim-1034",
        project="wemeet-barcode",
        group="mobilenet-followup-20261027",
        name=task["id"],
        job_type="timing" if "timing" in task["kind"] else "evaluation",
        config=task,
        mode="online",
        dir=str(out),
        tags=["mobilenet-followup", task["kind"]],
        settings=wandb.Settings(init_timeout=60),
    )
    for p, values in summary.get("paths", {}).items():
        for k, v in values.items():
            if isinstance(v, (int, float)):
                run.summary[f"{p}/{k}"] = v
    run.summary["n"] = summary["n"]
    artifact = wandb.Artifact(task["id"], type="evaluation", metadata={"scope": task["kind"]})
    for p in out.glob("*.json*"):
        artifact.add_file(str(p), name=p.name)
    run.log_artifact(artifact).wait()
    url = run.url
    run.finish()
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--attempt", type=int, default=1)
    args = ap.parse_args()
    inputs = read_json(args.root / "inputs.json")
    task = next(t for t in read_json(args.root / "tasks.json") if t["id"] == args.task)
    out = args.root / "jobs" / task["id"] / f"attempt-{args.attempt:02d}"
    out.mkdir(parents=True, exist_ok=True)
    configure(42, threads=1, device="cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != inputs["gpu_uuids"][str(args.gpu)]:
        raise RuntimeError("Worker GPU authorization mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one assigned GPU required")
    if not (out / "results.done.json").exists():
        predictor = Predictor(inputs["checkpoints"][task["model"]], task["model"])
        summary = (
            timing(inputs, task, predictor, out, args.gpu)
            if "timing" in task["kind"]
            else evaluate(inputs, task, predictor, out, args.gpu)
        )
        write_json(out / "summary.json", summary)
        write_json(
            out / "task.json",
            {**task, "inputs_sha256": digest(args.root / "inputs.json"), "gpu": args.gpu},
        )
        write_json(out / "hardware_end.json", snapshot())
        write_json(out / "results.done.json", {p.name: digest(p) for p in out.glob("*.json*")})
    else:
        for n, h in read_json(out / "results.done.json").items():
            if digest(out / n) != h:
                raise RuntimeError("Stored result changed before publication retry")
        summary = read_json(out / "summary.json")
    url = publish(task, out, summary)
    write_json(
        out.parent / "completion.json",
        {
            "status": "complete",
            "output": str(out),
            "wandb": url,
            "n": summary["n"],
            "task": task,
            "files": {p.name: digest(p) for p in out.glob("*.json*")},
        },
    )


if __name__ == "__main__":
    main()
