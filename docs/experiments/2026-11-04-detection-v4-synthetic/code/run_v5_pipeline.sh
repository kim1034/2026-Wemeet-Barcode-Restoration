#!/bin/bash
# v5: 고친 생성기로 1920x1080 데이터 → 검사 → 스모크 → n → s (CHEETAH stage1-v4-train, 11/05~)
#   cd ~/barcode_data/code && setsid nohup bash run_v5_pipeline.sh > ~/barcode_data/pipeline_v5.log 2>&1 &
# 단계 하나라도 실패하면 이유를 남기고 멈춘다 (다음 단계로 넘어가지 않음).
# 이미 끝난 단계는 건너뛴다 — 고친 뒤 같은 명령으로 다시 띄우면 이어서 간다.
set -u
export TZ=Asia/Seoul
PY=/home/jovyan_venv/.venv/torch2.3.0-py3.11-cuda12.1/bin/python
D=/home/jovyan/barcode_data
DATA="$D/dataset_v5"
N_HOURS=${N_HOURS:-36}
S_HOURS=${S_HOURS:-36}
# 배치는 probe_batch.py 로 잰 값 (1920, 최대 메모리 약 66%). MIG 에선 AutoBatch 가 NVML 오류로 죽는다
N_BATCH=${N_BATCH:-4}
S_BATCH=${S_BATCH:-2}
cd "$D/code"
log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "❌ $1 — $2 확인. 파이프라인 정지"; exit 1; }

# 중복 실행 방지 (11/02 파이프라인이 두 번 떠 있었다)
exec 9> "$D/.pipeline_v5.lock"
flock -n 9 || { log "❌ 이미 v5 파이프라인이 돌고 있음"; exit 1; }
if pgrep -f train_cheetah.py > /dev/null; then log "❌ 다른 학습이 돌고 있음"; exit 1; fi
log "파이프라인 시작 (n ${N_HOURS}h → s ${S_HOURS}h)"

# 1) 생성 (중간에 끊겨도 같은 명령이면 없는 것만 만든다)
n_img=$(find "$DATA/images" -name '*.jpg' 2> /dev/null | wc -l)
if [ "$n_img" -lt 40000 ]; then
  free_gb=$(df -BG --output=avail "$D" | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge 20 ] || fail "디스크 ${free_gb}GB (20GB 필요)" "df -h $D"
  log "생성 시작 (현재 ${n_img}장)"
  $PY -u make_damaged_barcode_dataset_v4.py --height 1080 --out "$DATA" --num 40000 --workers 3 \
      >> "$D/gen_v5.log" 2>&1 || fail "생성 실패" "gen_v5.log"
  n_img=$(find "$DATA/images" -name '*.jpg' | wc -l)
  [ "$n_img" -eq 40000 ] || fail "생성 ${n_img}장 (40000 필요)" "gen_v5.log"
fi
log "생성 끝: $(ls "$DATA/images/train" | wc -l) / $(ls "$DATA/images/val" | wc -l)"

# 2) 검사: 형식 · 정답 위치(label_cover) · v2 독립 검사
#    v2 검사는 train 4,000장 — 생성기가 val 에는 손상 없는 장면을 안 넣어서(복합 손상·twist 만) val 로는 판정 불가
if [ ! -f "$DATA/qa/PASS" ]; then
  log "검사 시작"
  $PY -u qa_dataset.py "$DATA" --size 1920 1080 --zero-shot best_v2.pt --zs-split train --zs-max 4000 --imgsz 1920 --device 0 \
      > "$D/qa_v5.log" 2>&1 || fail "데이터 검사 불합격" "qa_v5.log, dataset_v5/qa/qa_report.json"
  touch "$DATA/qa/PASS"
fi
log "검사 통과: $(grep -a 'v2 독립' "$D/qa_v5.log" | cut -c1-200)"

# 3) 스모크 (2%, 1 에폭): s 먼저 — 1920 에서 메모리가 모자라면 n 36시간 전에 알아야 한다
for m in s n; do
  [ "$m" = s ] && { W=yolo11s-obb.pt; B=$S_BATCH; } || { W=best_v2.pt; B=$N_BATCH; }
  if ! grep -qa "스모크 통과" "$D/smoke_v5_$m.log" 2> /dev/null; then
    log "$m 스모크 시작"
    $PY -u train_cheetah.py --data "$DATA" --device 0 --model $W --imgsz 1920 --batch $B --smoke \
        --name smoke_v5_$m --project "$D/runs" > "$D/smoke_v5_$m.log" 2>&1 || fail "$m 스모크 실패" "smoke_v5_$m.log"
  fi
  log "$(grep -a '스모크:' "$D/smoke_v5_$m.log")"
done

# 4) W&B 로그인 (키는 ~/.netrc. 사용자가 JupyterLab 터미널에서 wandb login)
until (cd /tmp && $PY -c "import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)") 2> /dev/null; do
  log "W&B 로그인 대기 중 (JupyterLab 터미널에서 wandb login)"; sleep 300
done

# 5) 본 학습 n → s
train() {  # $1=이름 $2=시작 가중치 $3=시간 $4=배치
  if [ -f "$D/runs/$1/weights/best.pt" ] && grep -qa "✅ 완료" "$D/train_$1.log" 2> /dev/null; then
    log "$1 이미 끝남 — 건너뜀"; return
  fi
  log "$1 학습 시작 (${3}h, 배치 $4)"
  $PY -u train_cheetah.py --data "$DATA" --device 0 --model "$2" --imgsz 1920 --batch "$4" --hours "$3" \
      --name "$1" --project "$D/runs" --wandb-entity naseesunny- > "$D/train_$1.log" 2>&1 \
      || fail "$1 학습 실패" "train_$1.log"
  [ -f "$D/runs/$1/weights/best.pt" ] || fail "$1 best.pt 없음" "train_$1.log"
  log "$1 끝: $(grep -a 'mAP50 =' "$D/train_$1.log" | tail -1)"
}
train v5_n best_v2.pt "$N_HOURS" "$N_BATCH"
train v5_s yolo11s-obb.pt "$S_HOURS" "$S_BATCH"
log "✅ 전부 끝"
