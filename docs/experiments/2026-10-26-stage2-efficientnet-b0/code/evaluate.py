"""held-out exact-decode 평가. `scripts/evaluate_geometry.py` 의 `main()` 복사본.

원본과 다른 점은 아래뿐이다. 이미지 읽기·예측·TPS 보정·디코딩·집계는 원본 함수
(`_load_rows`, `_predict`, `_decode_counts`)를 import 해서 그대로 쓴다.

1. `_model()`: `--arch` 로 EfficientNet-B0-C3 / ResNet18-C3 를 고른다.
   두 모델을 **같은 평가 코드**로 돌리기 위해서다 (가이드 4장 실행 순서 4).
2. `_model()`: 학습 체크포인트(`best.pt`, ``{"model": ..., "epoch": ...}``)와
   Hugging Face 배포 파일(`model_state_dict.pt`, state_dict 만 있음)을 둘 다 받는다.
   원본은 앞의 것만 받는다.
3. W&B: run 이름·tags 를 가이드 5장 표대로, 정답/오독/실패 개수도 기록,
   artifact 이름을 run 이름으로 바꾸고 `latest`/`candidate` 별칭을 붙이지 않는다.

실행 (저장소 루트에서):
    PYTHONPATH=. uv run python docs/experiments/2026-10-26-stage2-efficientnet-b0/code/evaluate.py \
        --arch efficientnet-b0-c3 --data-root .../synthetic/v1 \
        --checkpoint runs/restoration-efficientnet-b0-c3-v1/best.pt --wandb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from model import EfficientNetB0C3, count_params

from scripts.evaluate_geometry import _decode_counts, _load_rows, _predict
from wemeet.ai.geometry.model import ResNet18C3, destination_grid

ARCHS = {
    "efficientnet-b0-c3": {
        "cls": EfficientNetB0C3,
        "model": "EfficientNet-B0-C3",
        "run_name": "restoration-efficientnet-b0-c3-eval-v1",
    },
    "resnet18-c3": {
        "cls": ResNet18C3,
        "model": "ResNet18-C3",
        "run_name": "restoration-resnet18-c3-reeval-efb0",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCHS), required=True)
    parser.add_argument("--data-root", type=Path, required=True, help=".../synthetic/v1")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="best.pt 또는 model_state_dict.pt"
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="state_dict 만 있는 파일은 epoch 정보가 없다. 기록용으로 직접 적는다 (ResNet: 48)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="limit per bucket for a smoke run")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="log summary and result JSON to W&B")
    parser.add_argument("--offline-wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true", help="시험 실행. W&B tag 에 smoke 를 붙인다")
    return parser


def _model(arch: str, checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]  # 학습 체크포인트 (best.pt)
    elif isinstance(payload, dict):
        state_dict, payload = payload, {}  # HF 배포 파일 (model_state_dict.pt)
    else:
        raise ValueError(f"체크포인트 형식을 알 수 없습니다: {checkpoint}")
    model = ARCHS[arch]["cls"](pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, payload


def main() -> None:
    args = _parser().parse_args()
    spec = ARCHS[args.arch]
    run_name = args.run_name or spec["run_name"]
    output = args.output or Path(f"runs/{run_name}.json")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model, checkpoint = _model(args.arch, args.checkpoint.resolve(), device)
    checkpoint_epoch = checkpoint.get("epoch", args.checkpoint_epoch)
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
    result = {
        "arch": args.arch,
        "model": spec["model"],
        "params": count_params(model),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "data_root": str(args.data_root.resolve()),
        "buckets": results,
        "total": total,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"total": total, "output": str(output)}, ensure_ascii=False, indent=2))

    if args.wandb:
        import wandb

        tags = ["stage2", "backbone-compare", args.arch, "heldout", "exact-decode"]
        if args.smoke:
            tags.append("smoke")
        with wandb.init(
            project="wemeet-barcode",
            entity="kim-1034",
            name=run_name,
            job_type="eval",
            tags=tags,
            mode="offline" if args.offline_wandb else "online",
            config={
                "model": spec["model"],
                "params": result["params"],
                "checkpoint_file": args.checkpoint.name,
                "checkpoint_epoch": checkpoint_epoch,
                "n": total["n"],
                "limit": args.limit,
                "device": str(device),
            },
        ) as run:
            metrics = {
                "eval/rescue_rate": total["rescue_rate"],
                "eval/misread_rate": total["misread_rate"],
                "eval/decode_failed_rate": total["decode_failed_rate"],
                "eval/strict_rescue_rate": total["strict_rescue_rate"],
                "eval/ok": total["ok"],
                "eval/misread": total["misread"],
                "eval/decode_failed": total["decode_failed"],
            }
            for bucket, bucket_result in results.items():
                metrics.update(
                    {
                        f"eval/{bucket}/rescue_rate": bucket_result["rescue_rate"],
                        f"eval/{bucket}/misread_rate": bucket_result["misread_rate"],
                        f"eval/{bucket}/decode_failed_rate": bucket_result["decode_failed_rate"],
                        f"eval/{bucket}/ok": bucket_result["ok"],
                        f"eval/{bucket}/misread": bucket_result["misread"],
                        f"eval/{bucket}/decode_failed": bucket_result["decode_failed"],
                    }
                )
            run.log(metrics)
            run.summary.update(metrics)
            # 평가 JSON 만 올린다 (가중치 아님). 원본의 latest/candidate 별칭은 붙이지 않는다.
            artifact = wandb.Artifact(run_name, type="evaluation")
            artifact.add_file(str(output), name="geometry-eval.json")
            run.log_artifact(artifact)
            print(f"W&B run: {run.url}")


if __name__ == "__main__":
    main()
