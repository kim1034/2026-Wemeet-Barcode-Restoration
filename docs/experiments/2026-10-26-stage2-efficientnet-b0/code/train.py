"""EfficientNet-B0-C3 학습. `scripts/train_geometry.py` 의 `main()` 복사본.

원본과 다른 점은 아래뿐이다. 데이터 선택·분할·전처리·손실·학습 루프는 원본 함수를
import 해서 그대로 쓴다.

1. 모델 생성 줄: `ResNet18C3(...)` → `EfficientNetB0C3(...)`
2. 기본 출력 폴더·run 이름: `restoration-efficientnet-b0-c3-v1`
3. W&B: model 이름, tags, config 추가(데이터 revision, params 수, split sha256)
4. W&B 모델 artifact 업로드 삭제 (가중치는 채택 후 Hugging Face 에만 둔다)
5. `--smoke`: 시험 실행 표시(tag `smoke`)

실행 (저장소 루트에서):
    PYTHONPATH=. uv run python docs/experiments/2026-10-26-stage2-efficientnet-b0/code/train.py \
        --data-root ../data/barcode-datasets-a833a3a59441/synthetic/v1 --epochs 50 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from model import EfficientNetB0C3, count_params
from torch.utils.data import DataLoader

from scripts.train_geometry import RecipeDataset, _run_epoch, read_records, set_seed

DATA_REVISION = "a833a3a5944148cb9815ccbd1f05aced50f6c464"
MODEL_NAME = "EfficientNet-B0-C3"
MODEL_TAG = "efficientnet-b0-c3"


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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/restoration-efficientnet-b0-c3-v1")
    )
    parser.add_argument("--run-name", default="restoration-efficientnet-b0-c3-v1")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--offline-wandb", action="store_true")
    parser.add_argument("--data-revision", default=DATA_REVISION)
    parser.add_argument("--smoke", action="store_true", help="시험 실행. W&B tag 에 smoke 를 붙인다")
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
    # 원본 `_recipe()` 는 set 순서로 키를 넣어서 실행마다 split.json 의 바이트가 달라진다
    # (PYTHONHASHSEED). 내용 비교가 되도록 키를 정렬한 정규형으로 해시를 낸다.
    split_sha256 = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB0C3(pretrained=not args.no_pretrained).to(device)  # 원래: ResNet18C3(...)
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
        model=MODEL_NAME,
        backbone="efficientnet_b0().features[:6]",
        backbone_weights="EfficientNet_B0_Weights.IMAGENET1K_V1",
        params=count_params(model),
        input_hw=[160, 384],
        control_grid=[16, 3],
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        train_size=len(train_records),
        validation_size=len(val_records),
        dataset_root=str(args.data_root),
        dataset_revision=args.data_revision,
        split_sha256=split_sha256,
    )
    tags = ["stage2", "backbone-compare", MODEL_TAG]
    if args.smoke:
        tags.append("smoke")
    with wandb.init(
        project="wemeet-barcode",
        entity="kim-1034",
        name=args.run_name,
        job_type="train",
        tags=tags,
        mode=mode,
        config=config,
    ) as run:
        best, best_epoch = float("inf"), None
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
                best, best_epoch = val_stats["loss"], epoch
                torch.save(checkpoint, args.output_dir / "best.pt")
                run.summary.update(
                    {f"best/val_{key}": value for key, value in val_stats.items()}
                )
        # 원본의 wandb.Artifact("resnet18-c3-model", ...) 업로드는 지웠다.
        # 가중치(best.pt / last.pt)는 W&B 에 올리지 않는다 (가이드 5장).
        run.summary.update(
            {
                "best_val_loss": best,
                "best_epoch": best_epoch,
                "device": str(device),
                "params": config["params"],
                "split_sha256": split_sha256,
            }
        )
        print(f"W&B run: {run.url}")
        print(f"best checkpoint: {args.output_dir / 'best.pt'} (epoch {best_epoch})")
        print(f"split.json sha256: {split_sha256}")


if __name__ == "__main__":
    main()
