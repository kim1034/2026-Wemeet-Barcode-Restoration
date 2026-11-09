#!/bin/bash
# 2026-11-04 detection v4/v5 synthetic — 재현 순서
# 리포 환경(pyproject.toml)과 별개로 ultralytics·opencv·python-barcode 가 깔린 환경에서 돌린다
# (CHEETAH: ultralytics 8.4.198, 노트북 평가: 8.4.137).
set -eu
C="$(cd "$(dirname "$0")" && pwd)/code"

# ── 1. 데이터 v5 (1920×1080, 4만 장 = train 36,000 / val 4,000). CPU 3개로 약 5시간
python "$C/make_damaged_barcode_dataset_v4.py" --height 1080 --out dataset_v5 --num 40000 --workers 3

# ── 2. 학습 전 검사. 실패하면 exit 1
#    v4 생성기 val 에는 손상 없는 장면이 없어서 v2 독립 검사는 train 에서 뽑는다
python "$C/qa_dataset.py" dataset_v5 --size 1920 1080 \
    --zero-shot barcode_v2_best.pt --zs-split train --zs-max 4000 --imgsz 1920 --device 0

# ── 3. 학습 (CHEETAH MIG 1g.10gb). 배치는 AutoBatch 대신 직접 잰 값
python "$C/probe_batch.py" barcode_v2_best.pt --imgsz 1920   # → 4
python "$C/probe_batch.py" yolo11s-obb.pt --imgsz 1920       # → 2
#    서버에서는 2~3 단계와 스모크·n·s 학습을 한 번에: code/run_v5_pipeline.sh
python "$C/train_cheetah.py" --data dataset_v5 --device 0 --model barcode_v2_best.pt \
    --imgsz 1920 --batch 4 --hours 36 --name v5_n --project runs
python "$C/train_cheetah.py" --data dataset_v5 --device 0 --model yolo11s-obb.pt \
    --imgsz 1920 --batch 2 --hours 36 --name v5_s --project runs

# ── 4. v2 val 정답 다시 계산 (옛 생성기 make_barcode_dataset.py 를 import 하므로 같은 폴더에 둘 것)
python "$C/relabel720.py" relabeled          # → relabeled/labels/val/*.txt (10000번대 1,000장)
#    dataset_720 val 을 복사하고 labels/val 에 위 파일을 덮어써서 dataset_720_fixedval 을 만든다

# ── 5. 비교
python "$C/compare_models.py" \
    --model v2=barcode_v2_best.pt --model v4_n=v4_n/best.pt --model v5_n=runs/v5_n/weights/best.pt \
    --val-data dataset_720_fixedval/data.yaml --bench downloads/hf/benchmark/v2 \
    --imgsz 1920 --out results/compare_models_fixedval.json
python "$C/compare_models.py" \
    --model v2=barcode_v2_best.pt --model v5_n=runs/v5_n/weights/best.pt --model v5_s=runs/v5_s/weights/best.pt \
    --val-data dataset_720_fixedval/data.yaml --bench downloads/hf/benchmark/v2 \
    --imgsz 1920 --out results/compare_models_s.json
