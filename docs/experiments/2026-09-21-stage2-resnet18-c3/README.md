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

## 모델 크기와 학습 자원

| 항목 | 측정값 |
|---|---:|
| 전체 파라미터 | 3,217,538개 (3.22M) |
| 학습 가능 파라미터 | 3,217,538개 |
| 순수 모델 tensor payload | 12,888,192 bytes (12.29 MiB) |
| `best.pt` 전체 크기 | 38,695,208 bytes (36.90 MiB, 38.70 MB) |
| `last.pt` 전체 크기 | 38,695,208 bytes (36.90 MiB, 38.70 MB) |
| 학습 시간 | 63,620.38초 = 17시간 40분 20초 |
| GPU 사용량 | A100-PCIE-40GB 1개 × 17.672 GPU-hours |

`best.pt`와 `last.pt`의 전체 파일에는 모델 가중치뿐 아니라 AdamW optimizer 상태,
학습 설정과 검증 결과가 함께 들어 있다. 따라서 배포용 가중치 크기와 재시작 가능한
학습 체크포인트 크기가 다르다. W&B 실행의 `_runtime`과 체크포인트 파일을 기준으로
계산했으며, W&B 메타데이터에 서버의 GPU 8개가 보이는 것은 시스템 모니터링 정보이고
이번 학습 프로세스가 사용한 GPU는 1개다.

Cheetah 내부 GPU 시간당 요금은 실행 기록에 포함되어 있지 않아 원화 금액은 확정할 수
없다. 실제 비용은 `17.672 GPU-hours × A100-PCIE-40GB 내부 단가(원/GPU-hour)`로
계산하면 된다. 예를 들어 단가가 1,000원/GPU-hour라면 17,672원이다.

## Held-out exact-decode 평가

학습에 사용하지 않은 `eval/low`, `eval/mid`, `eval/high`를 각 500장씩 사용해
총 1,500장을 평가했다. 모델이 예측한 `src_norm`과 고정 `16×3` destination grid를
사용하고, 출력 크기는 각 manifest의 `(h_flat, w_flat)`로 맞췄다. 보정은 TPS,
디코더는 `zxing-cpp`이며, 결과 문자열이 manifest의 `text`와 **정확히 같은 경우만**
성공으로 세었다.

```python
fixed = rectify(obs, dst_grid(), predicted_src_norm, (h_flat, w_flat))
got = decode(fixed)
correct = got == text
misread = got is not None and got != text
```

평가 세트는 원래 보정 전 exact decode가 실패하고 정답 control point로 보정하면
읽히는 `target` 샘플만 포함하므로, 아래 `rescue_rate`는 `ok / 1500`으로 계산했다.

| bucket | n | exact success (`ok`) | misread | decode failed | rescue rate | misread rate |
|---|---:|---:|---:|---:|---:|---:|
| low | 500 | 338 | 2 | 160 | 67.60% | 0.40% |
| mid | 500 | 367 | 1 | 132 | 73.40% | 0.20% |
| high | 500 | 362 | 4 | 134 | 72.40% | 0.80% |
| **total** | **1,500** | **1,067** | **7** | **426** | **71.13%** | **0.47%** |

보정과 디코딩만 포함한 평균 시간은 low 165ms, mid 339ms, high 914ms였다. 이 값은
검출·모델 추론·파이프라인 재시도를 포함하지 않으며, full-resolution TPS를 순차
계산한 측정이다.

평가 구현은 [`scripts/evaluate_geometry.py`](../../../scripts/evaluate_geometry.py)다.
실행 결과의 상세 JSON은 로컬 `runs/geometry-eval-resnet18-c3-v1.json`에 생성되며,
대용량 산출물과 체크포인트는 Git에 넣지 않는다.

재현 명령은 저장소 루트에서 다음과 같다. `--wandb`를 붙이면 현재 계정의 W&B
인증으로 별도 평가 실행과 JSON Artifact를 기록한다.

```bash
python scripts/evaluate_geometry.py \
  --data-root ../data/barcode-datasets-a833a3a59441/synthetic/v1 \
  --checkpoint runs/restoration-resnet18-c3-v1/best.pt \
  --output runs/geometry-eval-resnet18-c3-v1.json \
  --batch-size 32 \
  --wandb
```

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

1. 이번 exact-decode 결과를 W&B의 별도 `eval` 실행으로 업로드한다.
2. 실제 검출기 crop과 실촬영 데이터에서도 같은 exact-decode 지표를 측정한다.
3. 재시도 배율·TPS 격자 축소를 적용한 end-to-end latency와 성공률을 측정한다.
4. 그 결과를 기준으로 ResNet18-C3를 MobileNetV3-Small·EfficientNet-B0와 비교하고 최종 백본을 결정한다.

현재 저장소에는 실제 검출기 crop·실촬영 이미지·전체 재시도 경로를 포함한 end-to-end 평가는 아직 없다.
