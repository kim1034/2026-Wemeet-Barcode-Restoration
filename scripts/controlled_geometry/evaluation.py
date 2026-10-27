"""Model-independent evaluation using the unchanged v1 native crop policy."""

import copy
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.controlled_geometry.common import PROTOCOL, array_digest, write_json
from scripts.controlled_geometry.data import preprocess
from scripts.evaluate_stage2_v1 import (
    crop_attempts,
    field_from,
    geometry_stats,
    observe_decode,
    outcome_stats,
    rate,
    verdict,
)
from wemeet.ai.geometry.model import destination_grid
from wemeet.schemas import DetectedBarcode, RectifiedBarcode
from wemeet.sw import decoding, pipeline

PATHS = ("d0", "crop_forced_model", "crop_policy_model")


def predict(model, gray, device):
    with torch.inference_mode():
        return model(preprocess(gray).unsqueeze(0).to(device))[0].cpu().numpy().astype(np.float64)


def evaluate_samples(model, samples, device="cpu", output=None, reference=None):
    model.eval()
    records = []
    dst = destination_grid().numpy().astype(np.float64)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for sid, row, gray, gt, stored_dst in samples:
            if not np.allclose(stored_dst, dst, atol=1e-6):
                raise ValueError("Destination grid mismatch")
            pred = predict(model, gray, device)
            valid = (
                pred.shape == (48, 2)
                and bool(np.isfinite(pred).all())
                and pred.min() >= -0.5
                and pred.max() <= 1.5
            )
            target = DetectedBarcode(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0, 1)
            _, d0 = observe_decode(RectifiedBarcode(gray, target, None))
            if reference is not None:
                actual = reference_record(row, gray, gt, d0)
                if reference.get(sid) != actual:
                    raise ValueError(f"Frozen sample or D0 decision changed: {sid}")
            forced = (
                crop_attempts(target, pred, dst)
                if valid
                else {
                    "final": {"text": None, "format": None, "reason": "invalid_geometry"},
                    "attempts": [],
                }
            )
            policy = (
                {"final": d0, "attempts": [], "early_exit": True}
                if d0["text"] is not None
                else {
                    **copy.deepcopy(forced),
                    "early_exit": False,
                }
            )
            item = {
                "id": sid,
                "group_id_proxy": row["text"],
                "bucket": row["bucket"],
                "band": row.get("band", "unknown"),
                "preset": row.get("preset", "unknown"),
                "gt_text": row["text"],
                "crop_hw": list(gray.shape),
                "d0": d0,
                "d0_status": verdict(d0, row["text"]),
                "geometry_valid": valid,
                "prediction_src_norm": pred.tolist() if valid else None,
                "gt_src_norm": np.asarray(gt).tolist(),
                "error_xy_px": ((pred - gt) * [gray.shape[1] - 1, gray.shape[0] - 1]).tolist()
                if valid
                else None,
                "monotonicity_violations": int((np.diff(pred[:, 0].reshape(3, 16)) <= 0).sum())
                if valid
                else None,
                "paths": {
                    "d0": {"final": d0, "attempts": [d0]},
                    "crop_forced_model": forced,
                    "crop_policy_model": policy,
                },
            }
            for path in item["paths"].values():
                path["status"] = verdict(path["final"], row["text"])
                for attempt in path["attempts"]:
                    attempt["status"] = verdict(attempt, row["text"])
            records.append(item)
    finally:
        torch.set_num_threads(old_threads)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty evaluation or duplicate sample IDs")
    summary = summarize(records)
    if output:
        save_results(output, records, summary)
    return summary, records


def reference_record(row, gray, gt, d0):
    return {
        "gray_sha256": array_digest(gray),
        "gt_sha256": array_digest(gt),
        "d0_text": d0["text"],
        "d0_format": d0["format"],
        "d0_status": verdict(d0, row["text"]),
        "gt_text": row["text"],
    }


def summarize(records):
    def stats(rows):
        paths = {p: outcome_stats(rows, p) for p in PATHS}
        unreturned = [r for r in rows if r["d0_status"] == "U"]
        paths["crop_policy_model"]["RR_U"] = rate(
            sum(r["paths"]["crop_policy_model"]["status"] == "C" for r in unreturned),
            len(unreturned),
        )
        return {"n": len(rows), "geometry": geometry_stats(rows), "paths": paths}

    result = {"protocol": PROTOCOL, **stats(records)}
    for key in ("bucket", "band", "preset"):
        result["by_" + key] = {
            value: stats([r for r in records if r[key] == value])
            for value in sorted({r[key] for r in records})
        }
    return result


def save_results(output, records, summary):
    import json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_sample.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "summary.json", summary)
    write_json(
        output / "hard_ids.json",
        {
            "H": [r["id"] for r in records if r["d0_status"] != "C"],
            "H_U": [r["id"] for r in records if r["d0_status"] == "U"],
        },
    )
    columns = ["id", "bucket", "gt_text", "d0_status", "forced_status", "policy_status"]
    with (output / "outcomes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    **{k: row[k] for k in columns[:4]},
                    "forced_status": row["paths"]["crop_forced_model"]["status"],
                    "policy_status": row["paths"]["crop_policy_model"]["status"],
                }
            )


def selection_metrics(summary, prefix="val"):
    metrics = {f"{prefix}/{k}": v for k, v in (summary["geometry"]["metrics"] or {}).items()}
    for path in ("crop_forced_model", "crop_policy_model"):
        for metric in ("EDR", "FDR", "NRR", "RR_H", "FDR_H", "RR_U"):
            value = summary["paths"][path].get(metric)
            if value and value["pct"] is not None:
                metrics[f"{prefix}/{path}/{metric}_pct"] = value["pct"]
    return metrics


def measure_timing(model, samples, device, warmup=50, repeats=3):
    """Native, uninstrumented paths. No GT enters decoding or early exit."""
    model.eval()
    dst = destination_grid().numpy().astype(np.float64)
    targets = [DetectedBarcode(cv2.cvtColor(s[2], cv2.COLOR_GRAY2BGR), 0, 1) for s in samples]

    def stage2(target):
        gray = cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY)
        return field_from(predict(model, gray, device), dst)

    def policy(target):
        plain = RectifiedBarcode(
            cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY), target, None
        )
        first = decoding.decode(plain)
        if first.text is not None:
            return first
        return pipeline._rectify_and_decode(target, stage2(target), {})[1]

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    results = {}
    try:
        for name, function in {"stage2": stage2, "crop_policy_excluding_stage1": policy}.items():
            for index in range(warmup):
                function(targets[index % len(targets)])
            passes = []
            for _ in range(repeats):
                values = []
                for target in targets:
                    sync()
                    start = time.perf_counter()
                    function(target)
                    sync()
                    values.append((time.perf_counter() - start) * 1000)
                passes.append(
                    {
                        "raw_ms": values,
                        "p50_ms": float(np.quantile(values, 0.5)),
                        "p95_ms": float(np.quantile(values, 0.95)),
                    }
                )
            results[name] = {
                "n": len(targets),
                "warmup": warmup,
                "repeats": repeats,
                "passes": passes,
                "p50_ms": float(np.median([p["p50_ms"] for p in passes])),
                "p95_ms": float(np.median([p["p95_ms"] for p in passes])),
            }
    finally:
        torch.set_num_threads(old_threads)
    return results
