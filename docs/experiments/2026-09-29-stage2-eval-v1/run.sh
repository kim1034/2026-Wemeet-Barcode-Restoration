#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$repo_root"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
python_bin="${BARCODE_PYTHON:-$repo_root/../.venvs/barcode/bin/python}"
data_root="${BARCODE_DATA_ROOT:-$repo_root/../data/barcode-datasets-a833a3a59441/synthetic/v1}"
output_root="${BARCODE_EVAL_OUTPUT:-$repo_root/runs/stage2-eval-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
"$python_bin" -m scripts.evaluate_stage2_v1 --data-root "$data_root" --output "$output_root/benchmark"
"$python_bin" -m scripts.summarize_stage2_v1 "$output_root/benchmark"
# Validation uses the exact historical split manifest and regenerates its crops.
# Oracle and timing are confined to the fixed 1500-image benchmark above.
"$python_bin" -m scripts.evaluate_stage2_v1 --data-root "$data_root" --dataset validation --skip-oracle --skip-timing --output "$output_root/validation"
"$python_bin" -m scripts.summarize_stage2_v1 "$output_root/validation"
