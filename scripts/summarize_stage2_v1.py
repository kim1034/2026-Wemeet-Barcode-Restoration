"""Postprocess completed v1 evaluations without rerunning/changing predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.evaluate_stage2_v1 import digest, rate, write_json


def grouped_intervals(records, paths, draws=10000):
    """Paired resampling of payload proxy groups; keep sample denominators."""
    groups = defaultdict(lambda: np.zeros((len(paths), 8), dtype=np.int64))
    for row in records:
        h, u = row["d0_status"] != "C", row["d0_status"] == "U"
        for i, path in enumerate(paths):
            c = row["paths"][path]["status"] == "C"
            w = row["paths"][path]["status"] == "W"
            groups[row["group_id_proxy"]][i] += [1, c, w, h, h * c, h * w, u, u * c]
    array = np.stack(list(groups.values()))
    rng = np.random.default_rng(42)
    sums = []
    for start in range(0, draws, 100):
        indices = rng.integers(0, len(array), size=(min(100, draws - start), len(array)))
        sums.append(array[indices].sum(axis=1))
    sums = np.concatenate(sums)
    results = {}
    for i, path in enumerate(paths):
        results[path] = {}
        for metric, num, den in [
            ("EDR", 1, 0),
            ("FDR", 2, 0),
            ("RR_H", 4, 3),
            ("FDR_H", 5, 3),
            ("RR_U", 7, 6),
        ]:
            if metric == "RR_U" and path != "crop_policy_resnet":
                continue
            valid = sums[:, i, den] > 0
            values = 100 * sums[valid, i, num] / sums[valid, i, den]
            results[path][metric] = {
                "valid_draws": int(valid.sum()),
                "ci95_pct": np.quantile(values, [0.025, 0.975]).tolist() if valid.all() else None,
            }
    return {
        "method": "payload proxy group bootstrap, 10000 draws, seed42",
        "groups": len(groups),
        "warning": (
            "Proxy does not establish true source independence. "
            "All-zero observed errors yield degenerate percentile intervals; "
            "also report positive Wilson upper limits."
        ),
        "paths": results,
    }


def retries(records, path):
    stages = []
    baseline = {
        s: sum(
            r["paths"][path].get("early_exit", False) and r["paths"][path]["status"] == s
            for r in records
        )
        for s in "CWU"
    }
    for index in range(3):
        counts = {"C": 0, "W": 0, "U": 0}
        called = 0
        warp_ms = decode_ms = 0.0
        for row in records:
            item = row["paths"][path]
            attempts = item["attempts"]
            if item.get("early_exit") or not attempts:
                status = item["status"]
            else:
                status = attempts[min(index, len(attempts) - 1)]["status"]
                if len(attempts) > index:
                    attempt = attempts[index]
                    called += 1
                    warp_ms += attempt.get("warp_ms", 0)
                    decode_ms += attempt.get("decode_ms_diagnostic", 0)
            counts[status] += 1
        stages.append(
            {
                "attempt": index + 1,
                "scale": [1, 1.5, 3][index],
                "called_n": called,
                "cumulative": counts,
                "EDR": rate(counts["C"], len(records)),
                "added_C": counts["C"]
                - (stages[-1]["cumulative"]["C"] if stages else baseline["C"]),
                "added_W": counts["W"]
                - (stages[-1]["cumulative"]["W"] if stages else baseline["W"]),
                "warp_ms_sum": warp_ms,
                "decode_ms_sum_diagnostic": decode_ms,
                "added_time_ms_mean_all_samples": (warp_ms + decode_ms) / len(records),
            }
        )
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "run_config.json").read_text())
    records = [json.loads(line) for line in (root / "per_sample.jsonl").read_text().splitlines()]
    if len(records) != config["n"] or len(records) != summary["n"]:
        raise ValueError("Incomplete evaluation")
    if digest(root / "run_config.json") != summary["config_sha256"]:
        raise ValueError("Run config changed")
    for path, stats in summary["paths"].items():
        assert stats["C"] + stats["W"] + stats["U"] == len(records)
    paths = list(summary["paths"])
    hashes = json.loads((root / "input_hashes.json").read_text())
    array_hashes = sum(k.endswith(("/rendered_gray", "/gt_array")) for k in hashes)
    write_json(
        root / "integrity_counts.json",
        {
            "published_files_verified": len(hashes) - array_hashes,
            "rendered_arrays_fingerprinted": array_hashes,
            "note": "hashes_checked includes both categories; arrays are fingerprints only",
        },
    )
    write_json(root / "grouped_intervals.json", grouped_intervals(records, paths))
    write_json(
        root / "retry_summary.json",
        {p: retries(records, p) for p in paths if p.startswith("crop_")},
    )
    columns = ["id", "bucket", "band", "preset", "gt_text", "d0_text", "d0_format", "d0_status"]
    columns += [p + "_status" for p in paths] + [p + "_text" for p in paths]
    with (root / "outcomes.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            values = {k: row[k] for k in columns[:5]}
            values.update(
                d0_text=row["d0"]["text"], d0_format=row["d0"]["format"], d0_status=row["d0_status"]
            )
            for path in paths:
                values[path + "_status"] = row["paths"][path]["status"]
                values[path + "_text"] = row["paths"][path]["final"]["text"]
            writer.writerow(values)
    write_json(
        root / "postprocess_config.json",
        {"script_sha256": digest(__file__), "draws": 10000, "seed": 42},
    )
    write_json(
        root / "artifact_hashes.json",
        {
            p.name: digest(p)
            for p in root.iterdir()
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                "n": len(records),
                "config_sha256": summary["config_sha256"],
                "postprocessed": str(root),
            }
        )
    )


if __name__ == "__main__":
    main()
