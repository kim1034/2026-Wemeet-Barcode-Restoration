#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/../data/barcode-datasets-a833a3a59441/synthetic/v1}"

cd "$ROOT"
export BARCODE_EXECUTION_CONTEXT="${BARCODE_EXECUTION_CONTEXT:-gpu06}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"

python -m scripts.train_geometry \
  --data-root "$DATA_ROOT" \
  --epochs "${EPOCHS:-50}" \
  --target 5000 \
  --hard 3500 \
  --first-ok 1500 \
  --val-fraction 0.1 \
  --batch-size 8 \
  --num-workers 1 \
  --output-dir runs/restoration-resnet18-c3-v1 \
  --run-name restoration-resnet18-c3-v1
