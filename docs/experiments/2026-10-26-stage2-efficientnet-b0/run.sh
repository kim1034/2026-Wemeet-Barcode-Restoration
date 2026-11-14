#!/usr/bin/env bash
# Stage 2 EfficientNet-B0-C3 비교 실험 — 처음부터 끝까지 재현하는 명령.
#
# 사용법 (어디서 실행해도 저장소 루트로 이동한다):
#   bash docs/experiments/2026-10-26-stage2-efficientnet-b0/run.sh <단계>
#
# 단계 (가이드 4장 실행 순서):
#   data     데이터(지정 revision)와 ResNet 기준 가중치 내려받기
#   smoke    짧은 시험 실행: 1 epoch, 표본 100/70/30, 평가 버킷당 20장
#   train    본 학습: 50 epoch (ResNet 기준선은 A100 1장에서 약 17시간 40분 걸렸다)
#   split    split.json 이 ResNet 실행과 같은지 확인
#   eval     EfficientNet best.pt 평가 + ResNet 가중치 재평가 (같은 평가 코드)
#   latency  CPU 1스레드 지연 측정 (두 모델). GPU 서버가 아니라 비교할 컴퓨터에서, 다른 프로그램을 끄고
#   all      data → smoke → train → split → eval → latency
#
# 환경변수로 바꿀 수 있는 것 (3장 고정값은 바꾸지 않는다):
#   DATA_ROOT            기본 ../data/barcode-datasets-a833a3a59441/synthetic/v1
#   PYTHON               기본 "uv run python" (gpu06 venv 를 쓰면 PYTHON=python)
#   CUDA_VISIBLE_DEVICES 기본은 설정하지 않음(보이는 GPU 를 그대로 씀).
#                        치타 gpu06 처럼 GPU 가 여러 장이면 `nvidia-smi` 로 빈 번호를 보고 지정한다
#                        (ResNet 기준선은 4번을 썼다. 번호는 결과와 무관하다).
#                        GPU 가 1장인 치타 컨테이너에서는 지정하지 않거나 0 으로 둔다.
#   WANDB_OFFLINE=1      W&B 에 연결이 안 될 때. 끝나고 `uv run wandb sync wandb/offline-run-*` 필수
#   ALLOW_CPU=1          GPU 없이 학습해도 된다고 확인한 경우만. 기본은 GPU 가 안 잡히면 멈춘다
#   RESNET_CKPT          ResNet 가중치. 기본: runs/restoration-resnet18-c3-v1/best.pt 가 있으면 그것,
#                        없으면 HF 에서 받은 downloads/geometry/resnet18-c3/model_state_dict.pt
set -euo pipefail

EXP="docs/experiments/2026-10-26-stage2-efficientnet-b0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

REVISION="a833a3a5944148cb9815ccbd1f05aced50f6c464"
DATA_DIR="${DATA_DIR:-$ROOT/../data/barcode-datasets-a833a3a59441}"
DATA_ROOT="${DATA_ROOT:-$DATA_DIR/synthetic/v1}"
read -r -a PY <<< "${PYTHON:-uv run python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BARCODE_EXECUTION_CONTEXT="${BARCODE_EXECUTION_CONTEXT:-gpu06}"

# 원본 학습 코드는 GPU 가 안 잡히면 오류 없이 CPU 로 학습한다. 50 epoch 를 CPU 로 돌리면
# 며칠이 걸려도 알아채기 어려우므로, 학습 전에 GPU 를 확인하고 없으면 멈춘다.
require_gpu() {
  if [[ "${ALLOW_CPU:-0}" == "1" ]]; then return; fi
  if ! "${PY[@]}" -c "import torch, sys; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else '없음', '| CUDA_VISIBLE_DEVICES =', '${CUDA_VISIBLE_DEVICES:-(미설정)}'); sys.exit(0 if ok else 1)"; then
    echo "GPU 가 잡히지 않았습니다. nvidia-smi 와 CUDA_VISIBLE_DEVICES 를 확인하세요." >&2
    echo "(GPU 가 1장인 컨테이너라면 CUDA_VISIBLE_DEVICES 를 지우거나 0 으로 두세요.)" >&2
    exit 1
  fi
}

RUN="restoration-efficientnet-b0-c3-v1"
OFFLINE=()
if [[ "${WANDB_OFFLINE:-0}" == "1" ]]; then OFFLINE=(--offline-wandb); fi

resnet_ckpt() {
  if [[ -n "${RESNET_CKPT:-}" ]]; then echo "$RESNET_CKPT"
  elif [[ -f runs/restoration-resnet18-c3-v1/best.pt ]]; then echo runs/restoration-resnet18-c3-v1/best.pt
  else echo downloads/geometry/resnet18-c3/model_state_dict.pt
  fi
}

step_data() {
  uv run hf download 123metro/barcode-datasets --repo-type dataset \
    --revision "$REVISION" \
    --include "synthetic/v1/*" \
    --local-dir "$DATA_DIR"
  uv run hf download 123metro/barcode-weights \
    geometry/resnet18-c3/model_state_dict.pt \
    --local-dir downloads
  for b in low mid high; do
    printf '%s: %s장\n' "$b" "$(wc -l < "$DATA_ROOT/eval/$b/manifest.jsonl")"
  done
}

step_smoke() {
  require_gpu
  "${PY[@]}" "$EXP/code/train.py" \
    --data-root "$DATA_ROOT" \
    --epochs 1 --target 100 --hard 70 --first-ok 30 \
    --val-fraction 0.1 --batch-size 8 --num-workers 1 \
    --output-dir runs/smoke-efficientnet-b0-c3 \
    --run-name smoke-efficientnet-b0-c3 --smoke "${OFFLINE[@]}"
  "${PY[@]}" "$EXP/code/evaluate.py" --arch efficientnet-b0-c3 \
    --data-root "$DATA_ROOT" \
    --checkpoint runs/smoke-efficientnet-b0-c3/best.pt \
    --output runs/smoke-efficientnet-b0-c3/eval.json \
    --limit 20 --wandb --smoke --run-name smoke-efficientnet-b0-c3-eval "${OFFLINE[@]}"
}

step_train() {
  # 3장 고정값. ResNet 기준선 run.sh 와 같은 인자다. 바꾸지 않는다.
  require_gpu
  "${PY[@]}" "$EXP/code/train.py" \
    --data-root "$DATA_ROOT" \
    --epochs 50 \
    --target 5000 \
    --hard 3500 \
    --first-ok 1500 \
    --val-fraction 0.1 \
    --batch-size 8 \
    --num-workers 1 \
    --lr 1e-4 --weight-decay 1e-4 --smooth-l1-beta 0.02 --seed 42 \
    --output-dir "runs/$RUN" \
    --run-name "$RUN" "${OFFLINE[@]}"
}

step_split() {
  if [[ -n "${RESNET_SPLIT:-}" ]]; then
    "${PY[@]}" "$EXP/code/check_split.py" --candidate "runs/$RUN/split.json" --reference "$RESNET_SPLIT"
  elif [[ -f runs/restoration-resnet18-c3-v1/split.json ]]; then
    "${PY[@]}" "$EXP/code/check_split.py" --candidate "runs/$RUN/split.json" \
      --reference runs/restoration-resnet18-c3-v1/split.json
  else
    "${PY[@]}" "$EXP/code/check_split.py" --candidate "runs/$RUN/split.json" --from-wandb
  fi
}

step_eval() {
  "${PY[@]}" "$EXP/code/evaluate.py" --arch efficientnet-b0-c3 \
    --data-root "$DATA_ROOT" \
    --checkpoint "runs/$RUN/best.pt" \
    --output "runs/$RUN/eval.json" \
    --batch-size 32 --wandb "${OFFLINE[@]}"
  "${PY[@]}" "$EXP/code/evaluate.py" --arch resnet18-c3 \
    --data-root "$DATA_ROOT" \
    --checkpoint "$(resnet_ckpt)" --checkpoint-epoch 48 \
    --output "runs/$RUN/resnet18-c3-reeval.json" \
    --batch-size 32 --wandb "${OFFLINE[@]}"
}

step_latency() {
  "${PY[@]}" "$EXP/code/latency.py" \
    --threads 1 --warmup 20 --iters 200 \
    --output "runs/$RUN/latency.json" --wandb "${OFFLINE[@]}"
}

case "${1:-}" in
  data) step_data ;;
  smoke) step_smoke ;;
  train) step_train ;;
  split) step_split ;;
  eval) step_eval ;;
  latency) step_latency ;;
  all) step_data; step_smoke; step_train; step_split; step_eval; step_latency ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 1 ;;
esac
