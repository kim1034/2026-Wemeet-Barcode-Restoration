"""Frozen ResNet18-C3 evaluation under docs/decisions/0005 (stage2-eval-v1).

Run as python -m scripts.evaluate_stage2_v1. Does not select or train a model.
The historical benchmark is NOT an untouched final test set. Crop-policy results
exclude Stage 1 and must not be called full E2E results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from scripts.label_recipes import rectify
from wemeet.ai import geometry
from wemeet.ai.geometry.model import destination_grid
from wemeet.data.synthesis import build, recipe_from_dict
from wemeet.schemas import DetectedBarcode, GeometryField, RectifiedBarcode
from wemeet.sw import decoding, pipeline
from wemeet.sw.rectify import _FLOW_STEP_PX

PROTOCOL = "stage2-eval-v1"
DATA_REVISION = "a833a3a5944148cb9815ccbd1f05aced50f6c464"
WEIGHT_SHA = "038559beb75efc6935d2431c0fdb6bfa355862cd4e3fdf6c422cd5c84bbae18c"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def rate(k, n):
    if not n:
        return {"numerator": k, "denominator": n, "pct": None, "wilson95_pct": None}
    p, z = k / n, 1.96
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {
        "numerator": k,
        "denominator": n,
        "pct": 100 * p,
        "wilson95_pct": [100 * max(0, center - half), 100 * min(1, center + half)],
    }


def verdict(result, text):
    if result["text"] is None:
        return "U"
    return "C" if result["text"] == text and result["format"] == "Code128" else "W"


def observe_decode(image):
    """Observe native decoder backend results; never pass GT into selection logic."""
    found = []

    def wrap_zxing(original):
        def capture(*args, **kwargs):
            values = original(*args, **kwargs)
            for value in values:
                if value.valid and value.text:
                    fmt = (
                        "Code128"
                        if value.format == decoding._zxingcpp.BarcodeFormat.Code128
                        else str(value.format)
                    )
                    found.append((value.text, fmt, "zxing-cpp"))
                    break
            return values

        return capture

    def wrap_zbar(original):
        def capture(*args, **kwargs):
            values = original(*args, **kwargs)
            if values:
                try:
                    fmt = "Code128" if values[0].type == "CODE128" else values[0].type
                    found.append((values[0].data.decode(), fmt, "pyzbar"))
                except UnicodeDecodeError:
                    pass
            return values

        return capture

    with ExitStack() as stack:
        if decoding._pyzbar_decode is not None:
            stack.enter_context(
                patch.object(decoding, "_pyzbar_decode", wrap_zbar(decoding._pyzbar_decode))
            )
        if decoding._zxingcpp is not None:
            stack.enter_context(
                patch.object(
                    decoding._zxingcpp,
                    "read_barcodes",
                    wrap_zxing(decoding._zxingcpp.read_barcodes),
                )
            )
        result = decoding.decode(image)
    if result.text is None:
        return result, {
            "text": None,
            "format": None,
            "backend": None,
            "reason": result.failure_reason,
        }
    match = next((x for x in found if x[0] == result.text), None)
    if match is None:
        raise RuntimeError("Could not audit native decoder symbology")
    return result, {"text": result.text, "format": match[1], "backend": match[2], "reason": None}


def field_from(src, dst):
    return GeometryField(dst, np.asarray(src, dtype=np.float64), "tps", 1.0)


def crop_attempts(target, src, dst):
    attempts = []
    stage_ms = {}
    previous_warp_ms = 0.0

    def observed(rectified):
        nonlocal previous_warp_ms
        started = time.perf_counter()
        result, record = observe_decode(rectified)
        record["decode_ms_diagnostic"] = (time.perf_counter() - started) * 1000
        record["warp_ms"] = stage_ms.get("warp", 0.0) - previous_warp_ms
        previous_warp_ms = stage_ms.get("warp", 0.0)
        record["scale"] = pipeline._RETRY_OUT_SCALES[len(attempts)]
        record["out_hw"] = list(rectified.image_gray_uint8.shape)
        record["passthrough"] = rectified.field is None
        attempts.append(record)
        return result

    started = time.perf_counter()
    try:
        field = field_from(src, dst)
        with patch.object(pipeline, "decode", observed):
            pipeline._rectify_and_decode(target, field, stage_ms)
        final = attempts[-1].copy()
    except (ValueError, AssertionError, np.linalg.LinAlgError, cv2.error) as exc:
        final = {"text": None, "format": None, "backend": None, "reason": type(exc).__name__}
    return {
        "final": final,
        "attempts": attempts,
        "diagnostic_ms": (time.perf_counter() - started) * 1000,
        "stage_ms_diagnostic": stage_ms,
    }


def oracle_attempt(target, src, dst, shape, homography=False):
    gray = cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY)
    try:
        if homography:
            corners = [0, 15, 47, 32]
            src_px = (src * np.array([gray.shape[1] - 1, gray.shape[0] - 1]))[corners].astype(
                np.float32
            )
            dst_px = (dst * np.array([shape[1] - 1, shape[0] - 1]))[corners].astype(np.float32)
            matrix = cv2.getPerspectiveTransform(src_px, dst_px)
            fixed = cv2.warpPerspective(
                gray,
                matrix,
                (shape[1], shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            fixed = rectify(gray, dst, src, shape)
        _, record = observe_decode(RectifiedBarcode(fixed, target, None))
        return {"final": record, "attempts": [record]}
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return {
            "final": {"text": None, "format": None, "backend": None, "reason": type(exc).__name__},
            "attempts": [],
        }


def geometry_stats(records):
    valid = [r for r in records if r["geometry_valid"]]
    if not valid:
        return {"n": len(records), "invalid_field_count": len(records), "metrics": None}
    errors = np.asarray([r["error_xy_px"] for r in valid], dtype=np.float64)
    epe = np.linalg.norm(errors, axis=2).ravel()
    metrics = {
        "epe_mean_px": float(epe.mean()),
        "epe_p95_px": float(np.quantile(epe, 0.95)),
        "epe_rmse_px": float(np.sqrt(np.mean(epe**2))),
        "mae_x_px": float(np.abs(errors[:, :, 0]).mean()),
        "mae_y_px": float(np.abs(errors[:, :, 1]).mean()),
        "monotonicity_violation_rate": sum(r["monotonicity_violations"] for r in valid)
        / (len(valid) * 45),
    }
    return {
        "n": len(records),
        "points": len(valid) * 48,
        "invalid_field_count": len(records) - len(valid),
        "metrics": metrics if len(valid) == len(records) else None,
        "valid_subset_diagnostic": metrics if len(valid) != len(records) else None,
    }


def outcome_stats(records, path):
    n = len(records)
    statuses = [r["paths"][path]["status"] for r in records]
    count = Counter(statuses)
    hard = [r for r in records if r["d0_status"] != "C"]
    unreturned = [r for r in records if r["d0_status"] == "U"]
    return {
        "n": n,
        "C": count["C"],
        "W": count["W"],
        "U": count["U"],
        "EDR": rate(count["C"], n),
        "FDR": rate(count["W"], n),
        "NRR": rate(count["U"], n),
        "wrong_among_returns": rate(count["W"], count["C"] + count["W"]),
        "RR_H": rate(sum(r["paths"][path]["status"] == "C" for r in hard), len(hard)),
        "FDR_H": rate(sum(r["paths"][path]["status"] == "W" for r in hard), len(hard)),
        "RR_U": rate(sum(r["paths"][path]["status"] == "C" for r in unreturned), len(unreturned))
        if path == "crop_policy_resnet"
        else None,
        "transitions": {
            a + "->" + b: sum(
                r["d0_status"] == a and r["paths"][path]["status"] == b for r in records
            )
            for a in "CWU"
            for b in "CWU"
        },
    }


def paired_bootstrap(records, model_path, reference_path, draws=10000):
    """Conservative payload groups, not an assertion of true source independence."""
    groups = {}
    for r in records:
        h = int(r["d0_status"] != "C")
        diff = int(r["paths"][model_path]["status"] == "C") - int(
            r["paths"][reference_path]["status"] == "C"
        )
        group = groups.setdefault(r["group_id_proxy"], [0, 0])
        group[0] += h * diff
        group[1] += h
    arr = np.asarray(list(groups.values()))
    rng = np.random.default_rng(42)
    values = []
    for start in range(0, draws, 100):
        ix = rng.integers(0, len(arr), size=(min(100, draws - start), len(arr)))
        sums = arr[ix].sum(axis=1)
        values.extend((100 * sums[sums[:, 1] > 0, 0] / sums[sums[:, 1] > 0, 1]).tolist())
    return {
        "method": "paired payload-group bootstrap (proxy), seed42",
        "draws": draws,
        "valid_draws": len(values),
        "empty_H_draws": draws - len(values),
        "delta_RR_H_pp": None
        if arr[:, 1].sum() == 0
        else float(100 * arr[:, 0].sum() / arr[:, 1].sum()),
        "ci95_pp": np.quantile(values, [0.025, 0.975]).tolist() if len(values) == draws else None,
    }


def load_samples(args):
    root = args.data_root.resolve()
    split = json.loads(args.split_manifest.read_text())
    train_ids = {r["id"] for r in split["train"]}
    val_ids = {r["id"] for r in split["validation"]}
    if (
        train_ids & val_ids
        or len(train_ids) != len(split["train"])
        or len(val_ids) != len(split["validation"])
    ):
        raise ValueError("Duplicate or overlapping training/validation IDs")
    groups = {k: {r["recipe"]["text"] for r in rows} for k, rows in split.items()}
    samples = []
    hashes = {}
    if args.dataset == "benchmark":
        expected = json.loads((root.parents[1] / "sha256.json").read_text())
        for bucket in ("low", "mid", "high"):
            directory = root / "eval" / bucket
            manifest = directory / "manifest.jsonl"
            hashes[str(manifest.relative_to(root))] = digest(manifest)
            if expected[str(manifest.relative_to(root.parents[1]))] != digest(manifest):
                raise ValueError(f"Manifest hash mismatch: {manifest}")
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            if len(rows) != 500:
                raise ValueError("Expected 500 samples per benchmark bucket")
            for row in rows[: args.limit]:
                for key in ("file", "npz"):
                    p = directory / row[key]
                    h = digest(p)
                    if expected[str(p.relative_to(root.parents[1]))] != h:
                        raise ValueError(f"Dataset hash mismatch: {p}")
                    hashes[str(p.relative_to(root))] = h
                gray = cv2.imread(str(directory / row["file"]), cv2.IMREAD_GRAYSCALE)
                with np.load(directory / row["npz"]) as npz:
                    gt = npz["src_norm"].copy()
                    dst = npz["dst_norm"].copy()
                samples.append((f"benchmark/{bucket}/{row['file']}", row, gray, gt, dst))
    else:
        selected = split["validation"][: args.limit]
        for i, record in enumerate(selected):
            sample = build(recipe_from_dict(record["recipe"]))
            row = {
                **record["recipe"],
                "band": record["band"],
                "h_flat": sample.h_flat,
                "w_flat": sample.w_flat,
                "m_min": sample.m_min,
                "sat_ratio": sample.sat_ratio,
            }
            samples.append(
                (f"validation/{record['id']}", row, sample.obs, sample.src_norm, sample.dst_norm)
            )
            if (i + 1) % 100 == 0:
                print(f"validation generated {i + 1}/{len(selected)}", flush=True)
    ids = [s[0] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    for sid, row, gray, gt, dst in samples:
        if gray is None or gray.ndim != 2 or not isinstance(row["text"], str) or not row["text"]:
            raise ValueError(f"Invalid sample {sid}")
        if (
            row["symbology"].lower() != "code128"
            or gt.shape != (48, 2)
            or not np.isfinite(gt).all()
        ):
            raise ValueError(f"Invalid GT {sid}")
        if not np.allclose(dst, destination_grid().numpy(), atol=1e-6):
            raise ValueError(f"Invalid destination grid {sid}")
        hashes[sid + "/rendered_gray"] = hashlib.sha256(gray.tobytes()).hexdigest()
        hashes[sid + "/gt_array"] = hashlib.sha256(gt.tobytes()).hexdigest()
    text_set = {s[1]["text"] for s in samples}
    # Fingerprint all recipe fields, not just the seed or reused text identifier.
    fields = set(split["train"][0]["recipe"])

    def recipe_hash(row):
        return hashlib.sha256(
            json.dumps({k: row[k] for k in sorted(fields)}, sort_keys=True).encode()
        ).hexdigest()

    training_recipes = {recipe_hash(r["recipe"]) for r in split["train"]}
    evaluation_recipes = {recipe_hash(s[1]) for s in samples}
    if training_recipes & evaluation_recipes:
        raise ValueError("Exact training recipes found in evaluation set")
    audit = {
        "train_n": len(split["train"]),
        "validation_n": len(split["validation"]),
        "train_validation_payload_overlap": len(groups["train"] & groups["validation"]),
        "exact_train_eval_recipe_overlap": len(training_recipes & evaluation_recipes),
        "train_validation_id_overlap": len(train_ids & val_ids),
        "evaluated_unique_payloads": len(text_set),
        "evaluated_payloads_in_train": len(text_set & groups["train"]),
        "group_definition": "proxy=payload; canonical group_id not provided",
        "independence": "not established; payload overlaps exist, not a fresh final test",
        "hashes_checked": len(hashes),
        "split_manifest_sha256": digest(args.split_manifest),
    }
    return samples, hashes, audit


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timing(samples, args, out):
    """Full ordered list, 50 warmups and 3 passes, native production functions."""
    targets = [DetectedBarcode(cv2.cvtColor(s[2], cv2.COLOR_GRAY2BGR), 0, 1) for s in samples]
    results = {}
    for device in ("cpu", args.device) if args.device != "cpu" else ("cpu",):
        os.environ["WEMEET_GEOMETRY_DEVICE"] = device
        geometry.load_model(device=device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        funcs = {"stage2": geometry.estimate_geometry}
        if device == args.device:

            def crop_policy(target):
                plain = RectifiedBarcode(
                    cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY), target, None
                )
                first = decoding.decode(plain)
                if first.text is not None:
                    return first
                try:
                    field = geometry.estimate_geometry(target)
                except Exception:
                    return first
                if field.confidence < pipeline._MIN_GEOMETRY_CONFIDENCE:
                    return first
                return pipeline._rectify_and_decode(target, field, {})[1]

            funcs["crop_policy_excluding_stage1"] = crop_policy
        for name, func in funcs.items():
            for i in range(50):
                func(targets[i % len(targets)])
            passes = []
            for repeat in range(3):
                times = []
                for i, target in enumerate(targets):
                    sync(device)
                    start = time.perf_counter()
                    func(target)
                    sync(device)
                    times.append((time.perf_counter() - start) * 1000)
                    if (i + 1) % 500 == 0:
                        print(
                            f"timing {device}/{name} pass{repeat + 1} {i + 1}/{len(targets)}",
                            flush=True,
                        )
                passes.append(
                    {
                        "p50_ms": float(np.quantile(times, 0.5)),
                        "p95_ms": float(np.quantile(times, 0.95)),
                        "raw_ms": times,
                    }
                )
            results[f"{device}/{name}"] = {
                "n": len(targets),
                "warmup": 50,
                "passes": passes,
                "representative_p50_ms": float(np.median([p["p50_ms"] for p in passes])),
                "representative_p95_ms": float(np.median([p["p95_ms"] for p in passes])),
            }
        if device.startswith("cuda"):
            results["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            results["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(out / "timing.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("runs/restoration-resnet18-c3-v1/split.json")
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("downloads/geometry/resnet18-c3/model_state_dict.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["benchmark", "validation"], default="benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if digest(args.weights) != WEIGHT_SHA:
        raise ValueError("Not the frozen published baseline weights")
    os.environ["WEMEET_GEOMETRY_WEIGHTS"] = str(args.weights.resolve())
    os.environ["WEMEET_GEOMETRY_DEVICE"] = args.device
    start = time.perf_counter()
    model = geometry.load_model(device=args.device)
    sync(args.device)
    load_ms = (time.perf_counter() - start) * 1000
    samples, hashes, audit = load_samples(args)
    write_json(out / "data_audit.json", audit)
    write_json(out / "input_hashes.json", hashes)
    write_json(
        out / "sample_manifest.json",
        [{"id": s[0], "group_id_proxy": s[1]["text"], "metadata": s[1]} for s in samples],
    )
    source_files = [
        "scripts/evaluate_stage2_v1.py",
        "wemeet/ai/geometry/__init__.py",
        "wemeet/ai/geometry/model.py",
        "wemeet/sw/pipeline/__init__.py",
        "wemeet/sw/rectify/__init__.py",
        "wemeet/sw/rectify/tps.py",
        "wemeet/sw/decoding/__init__.py",
        "scripts/label_recipes.py",
        "wemeet/data/synthesis.py",
    ]
    config = {
        "protocol": PROTOCOL,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {p: digest(p) for p in source_files},
        "dataset_revision": DATA_REVISION,
        "dataset": args.dataset,
        "n": len(samples),
        "partial_smoke": args.limit is not None,
        "weights_sha256": digest(args.weights),
        "checkpoint_epoch": 48,
        "checkpoint_selection": "historical validation loss minimum",
        "split_sha256": digest(args.split_manifest),
        "sample_manifest_sha256": digest(out / "sample_manifest.json"),
        "machine": platform.node(),
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_threads": 1,
        "dtype": "float32",
        "tf32": False,
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "torchvision",
                "numpy",
                "opencv-python",
                "zxing-cpp",
                "pyzbar",
                "Pillow",
            )
        },
        "decoder": (
            "native pyzbar first result, else first valid nonempty zxing read_barcodes; "
            "default options; no GT selection"
        ),
        "pyzbar_available": decoding._pyzbar_decode is not None,
        "symbology_scoring": "Code128 and exact unchanged string",
        "crop_retry_scales": list(pipeline._RETRY_OUT_SCALES),
        "flow_step_px": _FLOW_STEP_PX,
        "interpolation": "INTER_CUBIC",
        "border": "BORDER_REPLICATE",
        "oracle": "GT canvas; label_recipes full-grid TPS; same native decoder",
        "oracle_enabled": not args.skip_oracle,
        "geometry_points": 48,
        "quantile": "numpy linear",
        "model_load_ms": load_ms,
        "parameters": sum(p.numel() for p in model.parameters()),
        "deployment_weights_bytes": args.weights.stat().st_size,
        "limitations": [
            "historical synthetic development benchmark, not fresh test",
            "no Stage1 or real-photo E2E",
            "single training seed42",
            "group independence unproven; payload used as conservative proxy",
        ],
    }
    write_json(out / "run_config.json", config)
    config_sha = digest(out / "run_config.json")
    print(
        json.dumps(
            {"phase": "locked", "config_sha256": config_sha, "n": len(samples), "audit": audit}
        ),
        flush=True,
    )
    records = []
    dst = destination_grid().numpy().astype(np.float64)
    with (out / "per_sample.jsonl").open("w") as stream:
        for i, (sid, row, gray, gt, stored_dst) in enumerate(samples):
            target = DetectedBarcode(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0, 1)
            sync(args.device)
            estimate_start = time.perf_counter()
            with torch.inference_mode():
                pred = (
                    model(geometry._prepare_input(target, torch.device(args.device)))[0]
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            estimate_ms = (time.perf_counter() - estimate_start) * 1000
            valid = pred.shape == (48, 2) and bool(np.isfinite(pred).all())
            plain = RectifiedBarcode(gray, target, None)
            d0_start = time.perf_counter()
            _, d0 = observe_decode(plain)
            d0_ms = (time.perf_counter() - d0_start) * 1000
            r = {
                "id": sid,
                "group_id_proxy": row["text"],
                "bucket": row["bucket"],
                "band": row["band"],
                "preset": row["preset"],
                "gt_text": row["text"],
                "crop_hw": list(gray.shape),
                "d0": d0,
                "d0_status": verdict(d0, row["text"]),
                "stage_ms_diagnostic": {"estimate_raw": estimate_ms, "decode_first": d0_ms},
                "geometry_valid": valid,
                "prediction_src_norm": pred.tolist() if valid else None,
                "gt_src_norm": gt.tolist(),
                "error_xy_px": (
                    (pred - gt) * np.array([gray.shape[1] - 1, gray.shape[0] - 1])
                ).tolist()
                if valid
                else None,
                "monotonicity_violations": int(
                    (np.diff(pred[:, 0].reshape(3, 16), axis=1) <= 0).sum()
                )
                if valid
                else None,
                "paths": {"d0": {"final": d0, "attempts": [d0]}},
            }
            for name, src in (("resnet", pred), ("identity", dst), ("gt48", gt)):
                if name == "resnet" and not valid:
                    item = {
                        "final": {"text": None, "format": None, "reason": "invalid_geometry"},
                        "attempts": [],
                    }
                else:
                    item = crop_attempts(target, src, dst)
                r["paths"]["crop_forced_" + name] = item
                r["paths"]["crop_policy_" + name] = (
                    {"final": d0, "attempts": [], "early_exit": True}
                    if d0["text"] is not None
                    else {**item, "early_exit": False}
                )
                if not args.skip_oracle:
                    r["paths"]["oracle_" + name] = (
                        oracle_attempt(target, src, dst, (row["h_flat"], row["w_flat"]))
                        if valid or name != "resnet"
                        else item
                    )
            if not args.skip_oracle:
                r["paths"]["oracle_gt4"] = oracle_attempt(
                    target, gt, dst, (row["h_flat"], row["w_flat"]), homography=True
                )
            for p in r["paths"].values():
                p["status"] = verdict(p["final"], row["text"])
                for attempt in p["attempts"]:
                    attempt["status"] = verdict(attempt, row["text"])
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(r)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"evaluated {i + 1}/{len(samples)}", flush=True)
    summary = {
        "protocol": PROTOCOL,
        "config_sha256": config_sha,
        "n": len(records),
        "data_audit": audit,
        "geometry": geometry_stats(records),
        "paths": {p: outcome_stats(records, p) for p in records[0]["paths"]},
    }
    for key in ("bucket", "band", "preset"):
        summary["by_" + key] = {}
        for value in sorted({r[key] for r in records}):
            subset = [r for r in records if r[key] == value]
            summary["by_" + key][value] = {
                "geometry": geometry_stats(subset),
                "paths": {p: outcome_stats(subset, p) for p in records[0]["paths"]},
            }
    summary["bootstrap"] = {}
    for path in ("crop_forced", "crop_policy", "oracle"):
        if path + "_resnet" in records[0]["paths"]:
            summary["bootstrap"][path] = paired_bootstrap(
                records, path + "_resnet", path + "_identity"
            )
    write_json(
        out / "hard_ids.json",
        {
            "H": [r["id"] for r in records if r["d0_status"] != "C"],
            "H_U": [r["id"] for r in records if r["d0_status"] == "U"],
        },
    )
    write_json(out / "summary.json", summary)
    if not args.skip_timing:
        summary["timing"] = timing(samples, args, out)
        write_json(out / "summary.json", summary)
    write_json(
        out / "artifact_hashes.json", {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    )
    print(
        json.dumps(
            {
                "complete": str(out),
                "geometry": summary["geometry"],
                "resnet": summary["paths"]["crop_forced_resnet"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
