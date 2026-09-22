"""Evaluate ResNet18-C3 on the held-out exact-decode benchmark.

The benchmark contains 500 target samples in each of ``low``, ``mid`` and
``high``.  Each sample stores an observed crop, the exact text, and the
ground-truth ``dst_norm``/``src_norm`` fields.  The model predicts only
``src_norm``; the fixed destination grid and the manifest's ``h_flat`` /
``w_flat`` are used for rectification.

The primary counts intentionally compare the decoded text with the manifest
text.  A decoder result that is valid but has the wrong text is a misread, not
success.  This prevents checksum-passing false decodes from being counted as
rescues.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.label_recipes import decode, rectify
from scripts.train_geometry import IMAGENET_MEAN, IMAGENET_STD
from wemeet.ai.geometry.model import ResNet18C3, destination_grid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help=".../synthetic/v1")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/geometry-eval.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="limit per bucket for a smoke run")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="log summary and result JSON to W&B")
    parser.add_argument("--run-name", default="restoration-resnet18-c3-eval-v1")
    return parser


def _load_rows(bucket_dir: Path, limit: int | None) -> list[dict]:
    manifest = bucket_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"평가 manifest가 없습니다: {manifest}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"평가 샘플이 비어 있습니다: {manifest}")
    for row in rows:
        for key in ("file", "npz", "text", "h_flat", "w_flat"):
            if key not in row:
                raise KeyError(f"manifest 필드가 없습니다: {key}")
        if not (bucket_dir / row["file"]).is_file():
            raise FileNotFoundError(bucket_dir / row["file"])
        if not (bucket_dir / row["npz"]).is_file():
            raise FileNotFoundError(bucket_dir / row["npz"])
    return rows


def _input(obs: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(obs, (384, 160), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def _model(checkpoint: Path, device: torch.device) -> tuple[ResNet18C3, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ResNet18C3(pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


def _decode_counts(
    bucket_dir: Path,
    rows: list[dict],
    predictions: list[np.ndarray],
    dst: np.ndarray,
) -> dict:
    ok = misread = failed = first_correct = first_misread = 0
    rectify_ms: list[float] = []
    examples: list[dict] = []
    for row, src in zip(rows, predictions, strict=True):
        obs = cv2.imread(str(bucket_dir / row["file"]), cv2.IMREAD_GRAYSCALE)
        if obs is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {bucket_dir / row['file']}")
        text = row["text"]
        first = decode(obs)
        first_correct += first == text
        first_misread += first is not None and first != text
        started = time.perf_counter()
        try:
            fixed = rectify(
                obs,
                dst,
                src,
                (int(row["h_flat"]), int(row["w_flat"])),
            )
            got = decode(fixed)
        except (np.linalg.LinAlgError, ValueError, cv2.error):
            got = None
        rectify_ms.append((time.perf_counter() - started) * 1000.0)
        exact = got == text
        wrong = got is not None and got != text
        ok += exact
        misread += wrong
        failed += got is None
        if len(examples) < 10 and not exact:
            examples.append({"file": row["file"], "expected": text, "got": got})
    n = len(rows)
    return {
        "n": n,
        "ok": ok,
        "misread": misread,
        "decode_failed": failed,
        "first_correct": first_correct,
        "first_misread": first_misread,
        # The held-out set is target-only, so this follows the agreed metric:
        # rescue_rate = exact restored reads / n.
        "rescue_rate": ok / n,
        "misread_rate": misread / n,
        "decode_failed_rate": failed / n,
        "strict_rescue_rate": (ok - first_correct) / n,
        "rectify_decode_ms_mean": float(np.mean(rectify_ms)),
        "rectify_decode_ms_p95": float(np.percentile(rectify_ms, 95)),
        "non_exact_examples": examples,
    }


def _predict(
    model: ResNet18C3,
    device: torch.device,
    bucket_dir: Path,
    rows: list[dict],
    batch_size: int,
) -> list[np.ndarray]:
    images: list[torch.Tensor] = []
    for row in rows:
        obs = cv2.imread(str(bucket_dir / row["file"]), cv2.IMREAD_GRAYSCALE)
        if obs is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {bucket_dir / row['file']}")
        images.append(_input(obs))
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack(images[start : start + batch_size]).to(device)
            predictions.extend(model(batch).cpu().numpy())
    return predictions


def main() -> None:
    args = _parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model, checkpoint = _model(args.checkpoint.resolve(), device)
    dst = destination_grid().cpu().numpy()
    buckets = ("low", "mid", "high")
    results: dict[str, dict] = {}
    for bucket in buckets:
        bucket_dir = args.data_root.resolve() / "eval" / bucket
        rows = _load_rows(bucket_dir, args.limit)
        for row in rows:
            stored_dst = np.load(bucket_dir / row["npz"])["dst_norm"]
            if not np.allclose(stored_dst, dst, atol=1e-6):
                raise ValueError(f"destination grid 불일치: {bucket}/{row['file']}")
        predictions = _predict(model, device, bucket_dir, rows, args.batch_size)
        results[bucket] = _decode_counts(bucket_dir, rows, predictions, dst)
        print(json.dumps({"bucket": bucket, **results[bucket]}, ensure_ascii=False))

    total_keys = ("n", "ok", "misread", "decode_failed", "first_correct", "first_misread")
    total = {key: sum(result[key] for result in results.values()) for key in total_keys}
    total.update(
        rescue_rate=total["ok"] / total["n"],
        misread_rate=total["misread"] / total["n"],
        decode_failed_rate=total["decode_failed"] / total["n"],
        strict_rescue_rate=(total["ok"] - total["first_correct"]) / total["n"],
    )
    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device),
        "data_root": str(args.data_root.resolve()),
        "buckets": results,
        "total": total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"total": total, "output": str(args.output)}, ensure_ascii=False, indent=2))

    if args.wandb:
        import wandb

        with wandb.init(
            project="wemeet-barcode",
            entity="kim-1034",
            name=args.run_name,
            job_type="eval",
            tags=["candidate", "stage2", "resnet18-c3", "heldout", "exact-decode"],
            config={
                "checkpoint_epoch": checkpoint["epoch"],
                "n": total["n"],
                "device": str(device),
            },
        ) as run:
            metrics = {
                "eval/rescue_rate": total["rescue_rate"],
                "eval/misread_rate": total["misread_rate"],
                "eval/decode_failed_rate": total["decode_failed_rate"],
                "eval/strict_rescue_rate": total["strict_rescue_rate"],
            }
            for bucket, result in results.items():
                metrics.update(
                    {
                        f"eval/{bucket}/rescue_rate": result["rescue_rate"],
                        f"eval/{bucket}/misread_rate": result["misread_rate"],
                        f"eval/{bucket}/decode_failed_rate": result["decode_failed_rate"],
                    }
                )
            run.log(metrics)
            artifact = wandb.Artifact("resnet18-c3-eval-v1", type="evaluation")
            artifact.add_file(str(args.output), name="geometry-eval.json")
            run.log_artifact(artifact, aliases=["latest", "candidate"])
            print(f"W&B run: {run.url}")


if __name__ == "__main__":
    main()
