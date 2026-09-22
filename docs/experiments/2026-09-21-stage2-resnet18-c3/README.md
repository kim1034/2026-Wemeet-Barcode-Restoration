# Stage 2 ResNet18-C3 v1 학습 결과 — 2026-09-21

- **상태**: 후보 기준선 학습 완료, 최종 배포 모델 미결정
- **관련 결정**: [0003. 기하 추정 모델 선정](../../decisions/0003-기하-추정-모델-선정.md)
- **W&B 프로젝트**: [kim-1034/wemeet-barcode](https://wandb.ai/kim-1034/wemeet-barcode)
- **W&B 학습 실행**: [restoration-resnet18-c3-v1](https://wandb.ai/kim-1034/wemeet-barcode/runs/p9eghdt2)
- **데이터 Artifact 실행**: [dataset-synthetic-v1-kangmin](https://wandb.ai/kim-1034/wemeet-barcode/runs/8h9int00)

## 결론

ImageNet 사전학습 ResNet18의 `layer3`까지 사용하는 `ResNet18-C3` 후보를
기하 추정 모델의 기준선으로 학습했다. 입력은 합성 바코드 관측 이미지이고, 출력은
복원 이미지가 아니라 현재 왜곡된 crop의 **48개 control point 좌표**(`16×3×2`)다.

최고 검증 결과는 48 epoch에서 나왔다. 마지막 50 epoch에서는 검증 오차가 조금
증가했으므로 이후 추론·평가에는 `best.pt`를 사용한다.

이번 결과는 합성 검증셋의 **좌표 회귀 성능**이다. 아직 control point로 이미지를
펴는 Stage 3와 실제 바코드를 읽는 Stage 4를 연결하지 않았으므로, 이 결과만으로
최종 바코드 복원 성공률을 주장하지 않는다.

## 데이터와 분할

원본 데이터는 Hugging Face `123metro/barcode-datasets`의 다음 revision을 사용했다.

```text
a833a3a5944148cb9815ccbd1f05aced50f6c464
```

원본 합성 데이터 Artifact는 W&B에 `barcode-synthetic-v1:v0`로 보관했다. 학습
레시피는 L/M/H 각 bucket에서 아래 quota를 적용했다.

| band | bucket당 선택 수 | 전체 3 bucket |
|---|---:|---:|
| target | 5,000 | 15,000 |
| hard | 3,500 | 10,500 |
| first_ok | 1,500 | 4,500 |
| **합계** | **10,000** | **30,000** |

각 bucket에서 결정론적으로 90/10 분할하여 최종 크기는 다음과 같다.

```text
train:      27,024
validation:  2,976
```

held-out `eval` 데이터는 학습과 검증에 사용하지 않았다.

## 모델과 학습 조건

```text
model:          ResNet18-C3 (ImageNet pretrained)
input:          grayscale → 3-channel repeat, resize 160×384, ImageNet normalization
feature:        ResNet18 layer3, 256×10×24
output:         48×2 normalized source control points (16×3 grid)
loss:           Smooth L1, beta=0.02
optimizer:      AdamW
learning rate:  1e-4
weight decay:   1e-4
batch size:     8
epochs:         50
seed:           42
GPU:            gpu-06, A100-PCIE-40GB, CUDA 12.6
```

학습 구현은 [`scripts/train_geometry.py`](../../../scripts/train_geometry.py), 모델은
[`wemeet/ai/geometry/model.py`](../../../wemeet/ai/geometry/model.py)에 있다.

## 결과

| 체크포인트 | epoch | validation loss | 평균 오차(px) | RMSE(px) | P95(px) |
|---|---:|---:|---:|---:|---:|
| **best.pt** | **48** | **0.0010065** | **1.3723** | **2.7068** | **3.7989** |
| last.pt | 50 | 0.0010773 | 1.4306 | 2.8184 | 3.9388 |

오차는 합성 정답 control point와 모델 예측 좌표 사이의 pixel 오차다. W&B에는
epoch별 `train/*`, `val/*`, `epoch`, GPU 시스템 지표와 모델 Artifact가 기록되어 있다.

## 재현 명령

저장소 루트에서 GPU 환경과 W&B 인증을 준비한 뒤 다음처럼 실행한다. API 키는
명령줄에 직접 쓰지 말고 `WANDB_API_KEY` 환경변수 또는 `wandb login`으로 설정한다.

```bash
source ../barcode_setup/env.sh
source ../.venvs/barcode/bin/activate
export BARCODE_EXECUTION_CONTEXT=gpu06
export CUDA_VISIBLE_DEVICES=4
export WANDB_MODE=online

python -m scripts.train_geometry \
  --data-root ../data/barcode-datasets-a833a3a59441/synthetic/v1 \
  --epochs 50 \
  --target 5000 \
  --hard 3500 \
  --first-ok 1500 \
  --val-fraction 0.1 \
  --batch-size 8 \
  --num-workers 1 \
  --output-dir runs/restoration-resnet18-c3-v1 \
  --run-name restoration-resnet18-c3-v1
```

체크포인트와 split 파일은 `runs/` 아래에 생성되며 Git에는 올리지 않는다. `.gitignore`
가 `runs/`, `*.pt`를 제외하므로 대용량 파일은 W&B Artifact에 보관한다.

## 다음 단계

1. `best.pt`로 held-out `eval/low`, `eval/mid`, `eval/high`를 추론한다.
2. 예측 control point와 고정 destination grid를 Stage 3 TPS/OpenCV 보정에 연결한다.
3. 보정 전후 exact barcode decode를 측정한다.
4. rescue rate, false decode rate, bucket별 성공률, end-to-end latency를 같은 W&B 프로젝트에 기록한다.
5. 그 결과를 기준으로 ResNet18-C3를 MobileNetV3-Small·EfficientNet-B0와 비교하고 최종 백본을 결정한다.

현재 저장소에는 이 held-out rectification·decode 평가 코드는 아직 없다.
