"""Train and evaluate the Stage 2 ResNet18-C3 geometry regressor.

Training data are recipe JSONL files. Images are regenerated deterministically
with ``wemeet.data.synthesis.build`` and are never committed to Git. The
published ``eval`` directory is intentionally not used by this trainer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from wemeet.ai.geometry.model import ResNet18C3
from wemeet.data.synthesis import build, recipe_from_dict

BANDS = ("target", "hard", "first_ok")
RECIPE_FIELDS = {
    "seed",
    "bucket",
    "text",
    "d_m0",
    "d_t",
    "aspect",
    "preset",
    "cyl_share",
    "psi",
    "w_c_f",
    "lam_f",
    "phase",
    "offset",
    "light",
    "ks",
    "p",
    "sigma",
    "noise",
    "jpeg",
    "rot_deg",
    "margin",
    "symbology",
    "n_x",
    "n_y",
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _recipe(row: dict) -> dict:
    return {key: row[key] for key in RECIPE_FIELDS if key in row}


def read_records(
    recipe_root: Path,
    per_band: dict[str, int],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Select per-bucket quotas and split each quota deterministically."""
    val_quota = {band: max(1, round(quota * val_fraction)) for band, quota in per_band.items()}
    train, val = [], []
    for bucket in ("L", "M", "H"):
        path = recipe_root / f"train.{bucket}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(f"학습 레시피가 없습니다: {path}")
        selected_count = {band: 0 for band in BANDS}
        bucket_train_count = {band: 0 for band in BANDS}
        bucket_val_count = {band: 0 for band in BANDS}
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream):
                row = json.loads(line)
                band = row.get("band")
                if band not in BANDS:
                    continue
                if selected_count[band] >= per_band[band]:
                    continue
                key = f"{bucket}:{line_no}"
                record = {"id": key, "bucket": bucket, "band": band, "recipe": _recipe(row)}
                in_val = _stable_fraction(key, seed) < val_fraction
                if in_val and bucket_val_count[band] < val_quota[band]:
                    val.append(record)
                    bucket_val_count[band] += 1
                else:
                    train.append(record)
                    bucket_train_count[band] += 1
                selected_count[band] += 1
                if all(selected_count[b] >= per_band[b] for b in BANDS):
                    break
        missing = [band for band in BANDS if selected_count[band] < per_band[band]]
        if missing:
            raise RuntimeError(
                f"{bucket} 버킷 selection quota 부족: {missing}, counts={selected_count}"
            )
    if not train or not val:
        raise RuntimeError("선택된 train/validation 레코드가 비어 있습니다")
    return train, val


class RecipeDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        sample = build(recipe_from_dict(record["recipe"]))
        image = cv2.resize(sample.obs, (384, 160), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        target = torch.from_numpy(sample.src_norm.astype(np.float32))
        height, width = sample.obs.shape
        size = torch.tensor([width, height], dtype=torch.float32)
        return image, target, size, record["id"]


def _metrics(pred: torch.Tensor, target: torch.Tensor, size: torch.Tensor) -> dict[str, float]:
    error = (pred - target).abs()
    pixel = error * (size - 1.0).unsqueeze(1)
    flat = pixel.reshape(-1)
    return {
        "control_point_mean_px": float(flat.mean().item()),
        "control_point_rmse_px": float(torch.sqrt(pixel.square().mean()).item()),
        "control_point_p95_px": float(torch.quantile(flat, 0.95).item()),
    }


def _run_epoch(model, loader, optimizer, device, training: bool, beta: float) -> dict[str, float]:
    model.train(training)
    total_loss, count = 0.0, 0
    values: list[dict[str, float]] = []
    for images, targets, sizes, _ in loader:
        images, targets, sizes = images.to(device), targets.to(device), sizes.to(device)
        with torch.set_grad_enabled(training):
            pred = model(images)
            loss = F.smooth_l1_loss(pred, targets, beta=beta)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch = images.shape[0]
        total_loss += loss.item() * batch
        count += batch
        values.append(_metrics(pred.detach(), targets, sizes))
    return {
        "loss": total_loss / max(count, 1),
        **{key: float(np.mean([item[key] for item in values])) for key in values[0]},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help=".../synthetic/v1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.02)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--hard", type=int, default=70)
    parser.add_argument("--first-ok", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resnet18-c3"))
    parser.add_argument("--run-name", default="restoration-resnet18-c3-v1")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--offline-wandb", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    set_seed(args.seed)
    args.data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_band = {"target": args.target, "hard": args.hard, "first_ok": args.first_ok}
    train_records, val_records = read_records(
        args.data_root / "recipes", per_band, args.val_fraction, args.seed
    )
    split = {"train": train_records, "validation": val_records}
    (args.output_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18C3(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        RecipeDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        RecipeDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    import wandb

    mode = "offline" if args.offline_wandb else "online"
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        model="ResNet18-C3",
        input_hw=[160, 384],
        control_grid=[16, 3],
        device=str(device),
        train_size=len(train_records),
        validation_size=len(val_records),
        dataset_root=str(args.data_root),
    )
    with wandb.init(
        project="wemeet-barcode",
        entity="kim-1034",
        name=args.run_name,
        job_type="train",
        tags=["candidate", "stage2", "resnet18-c3"],
        mode=mode,
        config=config,
    ) as run:
        best = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_stats = _run_epoch(
                model, train_loader, optimizer, device, True, args.smooth_l1_beta
            )
            with torch.no_grad():
                val_stats = _run_epoch(
                    model, val_loader, optimizer, device, False, args.smooth_l1_beta
                )
            metrics = {f"train/{key}": value for key, value in train_stats.items()}
            metrics.update({f"val/{key}": value for key, value in val_stats.items()})
            metrics["epoch"] = epoch
            run.log(metrics, step=epoch)
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "val": val_stats,
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if val_stats["loss"] < best:
                best = val_stats["loss"]
                torch.save(checkpoint, args.output_dir / "best.pt")
        artifact = wandb.Artifact(
            "resnet18-c3-model", type="model", metadata={"best_val_loss": best}
        )
        artifact.add_file(str(args.output_dir / "best.pt"), name="best.pt")
        artifact.add_file(str(args.output_dir / "last.pt"), name="last.pt")
        artifact.add_file(str(args.output_dir / "split.json"), name="split.json")
        logged = run.log_artifact(artifact, aliases=["latest", "candidate"])
        if mode == "online":
            logged.wait()
        run.summary.update({"best_val_loss": best, "device": str(device)})
        print(f"W&B run: {run.url}")
        print(f"best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
