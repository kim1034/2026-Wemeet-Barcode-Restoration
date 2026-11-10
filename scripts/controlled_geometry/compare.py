"""Paired sample comparison of two completed evaluations with matching conditions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.controlled_geometry.common import digest, read_json, write_json


def paired_comparison(left, right, draws=10000):
    a, b = ({r["id"]: r for r in rows} for rows in (left, right))
    if not a or set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Missing, duplicate or mismatched paired IDs")
    results = {}
    for path in ("crop_forced_model", "crop_policy_model"):
        groups = defaultdict(lambda: np.zeros(5, dtype=np.float64))
        for sid, row in a.items():
            other = b[sid]
            if any(
                row[k] != other[k]
                for k in ("gt_text", "group_id_proxy", "d0_status", "gt_src_norm", "crop_hw")
            ):
                raise ValueError("Paired sample content/D0 mismatch")
            x, y = row["paths"][path]["status"], other["paths"][path]["status"]
            dc, dw, h = (
                int(y == "C") - int(x == "C"),
                int(y == "W") - int(x == "W"),
                int(row["d0_status"] != "C"),
            )
            groups[row["group_id_proxy"]] += [1, dc, dw, h, h * dc]
        array = np.asarray(list(groups.values()))
        rng = np.random.default_rng(42)
        totals = []
        for start in range(0, draws, 100):
            ix = rng.integers(0, len(array), size=(min(100, draws - start), len(array)))
            totals.append(array[ix].sum(axis=1))
        bootstrap = np.concatenate(totals)
        observed = array.sum(axis=0)
        results[path] = {}
        for name, numerator, denominator in (("EDR", 1, 0), ("FDR", 2, 0), ("RR_H", 4, 3)):
            valid = bootstrap[:, denominator] > 0
            values = 100 * bootstrap[valid, numerator] / bootstrap[valid, denominator]
            results[path][name] = {
                "delta_right_minus_left_pp": float(
                    100 * observed[numerator] / observed[denominator]
                )
                if observed[denominator]
                else None,
                "ci95_pp": np.quantile(values, [0.025, 0.975]).tolist() if valid.all() else None,
                "valid_draws": int(valid.sum()),
            }
    return {
        "n": len(a),
        "draws": draws,
        "grouping": "payload proxy; independence unproven",
        "paths": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = [read_json(p / "run_config.json") for p in (args.left, args.right)]
    for key in ("protocol_lock_sha256", "dataset", "preset", "seed", "n"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Not a controlled pair: {key}")
    rows = []
    for directory in (args.left, args.right):
        completion = read_json(directory / "completion.json")
        if not completion.get("files"):
            raise ValueError("Evaluation lacks output hashes")
        for relative, expected in completion["files"].items():
            if digest(directory / relative) != expected:
                raise ValueError(f"Evaluation artifact changed: {relative}")
        row = [
            json.loads(line)
            for line in (directory / "results/per_sample.jsonl").read_text().splitlines()
        ]
        if completion["status"] != "evaluated" or completion["n"] != len(row):
            raise ValueError("Incomplete evaluation")
        rows.append(row)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, {"left": configs[0], "right": configs[1], **paired_comparison(*rows)})


if __name__ == "__main__":
    main()
