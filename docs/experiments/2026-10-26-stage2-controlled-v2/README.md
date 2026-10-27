> 이 문서는 2026-10-26 당시의 준비·실행 기록을 보존한 것입니다. 최종 완료 결과는 [REPORT_ko.md](REPORT_ko.md), 후속 실험까지 합친 결과는 [통합 보고서](../2026-10-27-mobilenet-followup/REPORT_ko.md)를 참조하세요. 아래의 준비 중·실행 전 상태는 현재 상태가 아닙니다.

# MobileNet·ResNet 동일 조건 비교 — CPU 준비

2026-10-26 (KST). 이번 요청 범위는 GPU 사용 전 준비다. GPU 메모리·epoch 시간
측정과 본 학습은 시작하지 않는다. 실행 서비스나 자동 학습 큐도 만들지 않는다.
실제 완료 상태와 W&B 링크는 `runs/stage2-controlled-v2/CURRENT_STATE.json`에 기록한다.

CPU 준비 완료: 12개 설정 검증, 전체 테스트 233개 통과(기존 별도 fixture 미비로
10개 skip), 실제 ImageNet 가중치와 low/mid/high 이미지 6장의 두 모델 CPU 검사 통과.
기존 ResNet 체크포인트의 출력 차이는 0이며 원본 220개 파일의 hash가 유지됐다.
[최종 W&B 준비 기록](https://wandb.ai/kim-1034/wemeet-barcode/runs/liicjcwy)은
학습 성적이 아닌 입력 고정과 기술 검증 기록이다. GPU는 4번의 메모리·시간 측정부터
필요하며, 그 GPU 측정과 5번 본 학습은 아직 실행하지 않았다.

## 비교 조건

| preset | 각 조건에 적용할 모델 | lr | batch | epoch | Smooth L1 beta |
|---|---|---:|---:|---:|---:|
| R | ResNet18-C3, MobileNetV3-Small | 1e-4 | 8 | 50 | 0.02 |
| M | ResNet18-C3, MobileNetV3-Small | 3e-4 | 64 | 100 | 1/154 |

seed 42의 4개 실행과 seed 43/44 반복을 합쳐 12개 설정을 준비한다. 설정 생성은
학습 실행이 아니다. 같은 preset과 seed에서 두 모델을 비교한다. preset 사이에는
네 변수가 함께 바뀌므로 개별 학습률·beta의 인과 효과로 해석하지 않는다.

- train 27,024 / validation 2,976: 기존 실제 split을 바이트 단위로 복사하고 hash 고정.
- 기존 benchmark 1,500장: 개발 비교용이며 새 독립 test라고 부르지 않는다.
- ImageNet1K V1 백본, 입력 160×384, grayscale 3채널 반복 및 ImageNet 정규화.
- OS16 feature에서 C→256 adapter, 동일 spatial head, 16×3 제어점 출력.
- 기존 ResNet과 동일한 `identity + 0.5*tanh` 출력과 zero-output 초기화.
- AdamW, weight decay 1e-4, 일정 학습률, FP32, TF32 off, 학습 조기 종료 없음.
- seed는 초기화/셔플에만 사용한다. split은 seed와 독립적으로 유지한다.

[기존 MobileNet spatial run](https://wandb.ai/kim-1034/wemeet-barcode/runs/rjujpri3)의
설정을 인증된 W&B에서 확인했다. lr 3e-4, batch 64, 100 epoch, FP32, seed 42,
입력 160×384, 16×3 제어점이다. loss에는 beta 0.006494가 반올림되어 기록되어 있다.
원본 코드·체크포인트는 로컬·검토한 upstream main·해당 run의 파일과 Artifact에서
확보하지 못했다. 이번 MobileNet은 공통 head를 적용한 새 후보이며, 과거 보고서
모델의 정확한 재현을 주장하지 않는다. 모델 비용에는 adapter와 head를 포함한다.
조회 근거는 `runs/stage2-controlled-v2/references/mobilenet_spatial_wandb.json`에 있다.

CPU는 결정적 연산을 엄격히 검사한다. CUDA는 가능한 결정적 연산을 사용하되
미지원 연산은 경고를 기록한다. 기존 head의 AdaptiveAvgPool2d 역전파는
[PyTorch 2.8의 엄격한 CUDA 결정성 모드에서 지원되지 않는다](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html).
따라서 CUDA 재개의 비트 단위 동일성은 보장하지 않으며, seed 반복으로 변동을 보고한다.

## 기존 결과 보존과 버전

기준 source는 `8306e25`다. 최신 upstream `f449b9c34df3a42c5c324a0629b0dbee3308f1b3`도
확인했으며, 해당 버전에는 디코더 변경이 있다. 이번 비교는 기존 v1 평가와 연결하기
위해 현재 디코더·TPS·재시도 구현을 유지한다. 새 디코더 효과는 별도 비교 대상이다.

기존 모델, trainer, v1 evaluator, 결과 파일은 수정하지 않는다. 새 구현은
`wemeet/ai/geometry/controlled.py`와 `scripts/controlled_geometry/`에만 추가한다.
원본 220개 파일의 전후 hash 검사는 준비 산출물에 보관한다.

최종 입력 잠금은 `runs/stage2-controlled-v2/prepared/protocol.json`, 12개 실행 설정은
같은 폴더의 `configs/`에 있다. `preparation/`은 개발 중 검사 및 초기 준비 기록으로
보존하며 최종 학습 설정으로 사용하지 않는다. 학습기는 최종 source/package/data/
split/pretrained/D0 hash와 설정을 검사하고 달라지면 실행을 거절한다.

## 평가와 체크포인트

매 epoch loss와 **전체 점 기준 2D EPE**를 기록한다. 축별 MAE도 별도 이름으로
기록한다. 배치별 P95를 평균하지 않는다. beta가 다른 loss 값으로 순위를 정하지 않는다.

5 epoch마다와 마지막 epoch에 validation 전체를 같은 crop 경로로 평가한다.
checkpoint 선택 순서는 `crop_policy_model`의 RR_H 최대 → FDR_H 최소 →
EPE mean 최소 → 더 이른 epoch다. checkpoint 선택에 benchmark를 사용하지 않는다.

각 이미지에서 보정 전 D0, 모든 crop 강제 보정, D0 무응답 시 보정하는 운영 정책을
구분한다. 첫 오독 반환도 종료하며 GT로 재시도 결과를 선택하지 않는다. D0의 C/W/U와
원본 픽셀·좌표 hash를 준비 단계에 고정해 모델별 분모 재선별을 막는다.

정답/오독/무응답 장수와 비율, RR_H/FDR_H/RR_U, EPE mean/P95/RMSE, x/y MAE,
단조성 위반을 저장한다. Stage 1을 우회한 결과를 전체 E2E 성적으로 부르지 않는다.
독립적으로 확보한 새 holdout이나 실촬영 검증은 이번 준비에 포함되지 않는다.

`compare`는 같은 preset/seed/데이터 잠금의 두 완결 평가를 sample ID로 짝짓고,
payload 대리 그룹 bootstrap으로 ΔEDR/FDR/RR_H와 95% 구간을 계산한다.
대리 그룹은 원본 독립성의 증명이 아니다. 3개 학습 seed 변동은 별도 보고한다.

## 캐시와 재개

같은 recipe와 렌더링 코드·환경 fingerprint에 대해 원본 gray 이미지와 좌표를
캐시한다. 샘플별 파일 잠금으로 두 GPU의 별도 프로세스가 안전하게 공유할 수 있다.
캐시는 원본 값과 파일 hash를 검사하며 손상된 파일을 조용히 재생성하지 않는다.
준비 시 validation 캐시를 만들고 train 캐시는 최초 접근 시 생성한다.

학습은 한 프로세스에서 노출된 GPU 1개만 사용한다. GPU 2개 사용 시 각각 다른
설정의 프로세스를 실행한다. 자동 장치 선택이나 CUDA 실패 시 CPU 전환은 없다.

`last.pt`는 모델/optimizer/학습·셔플 RNG/선택된 best 상태/마지막 지표를 함께
원자적으로 저장한다. 재개 시 설정 일치를 검사한다. 불완전 validation 기록은
남기고 새 attempt 경로를 사용한다. 최종 `best.pt`의 summary는 같은 checkpoint의
지표만 사용한다.

## W&B

- entity/project: `kim-1034/wemeet-barcode`
- group: `stage2-backbone-controlled-v2`
- train/evaluation/preparation run을 job type으로 구분.
- 실제 학습·평가는 online 모드이며 인증 실패를 offline 성공으로 바꾸지 않는다.
- 입력 Artifact: protocol, split, D0 manifest, 실행 설정, source archive.
- 모델 Artifact: best/last checkpoint, 설정, 학습 지표.
- 평가 Artifact: sample별 원시 예측/GT 좌표와 판독 결과, 요약, CSV, 선택적 시간 측정.
- 준비와 CPU 기술 검증은 학습 성적과 구분한다. 키는 기록이나 Artifact에 넣지 않는다.

## 실행 안내

바코드 전용 터미널에서 환경을 사용한다. 기존 공용 환경을 다시 설치하지 않는다.

```bash
cd /home/jnuadmin/endtoend/kangmin/etc/2026-Wemeet-Barcode-Restoration
source ../barcode_setup/env.sh
```

아래 명령은 **검증만** 수행하며 GPU를 할당하거나 학습하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' "$UV_PROJECT_ENVIRONMENT/bin/python" \
  -m scripts.controlled_geometry.train \
  --config runs/stage2-controlled-v2/prepared/configs/restoration-resnet18-c3-presetR-s42-v2.json
```

본 학습은 GPU 실측 및 후속 시작 요청 이후 위 명령에 `--execute --device cuda:0`를
추가하고, `CUDA_VISIBLE_DEVICES`에는 실제 배정 GPU 하나를 지정해 실행한다.
다른 실험의 GPU를 자동으로 선택하지 않는다. 같은 실행 재개는 `--resume`을 추가한다.
완료한 준비나 학습을 무조건 다시 실행하지 않는다.

선택 checkpoint가 생성된 이후 평가 명령 형태는 다음과 같다. 대문자 값은 실제
파일 경로로 바꾼다. `--execute`가 없으면 입력 검증만 수행한다.

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.controlled_geometry.evaluate \
  --checkpoint SELECTED_BEST_PT --output NEW_EVAL_DIRECTORY \
  --dataset benchmark --device cpu
```

`--timing`은 별도, 경쟁 작업 없는 측정에만 사용한다. 50회 warm-up과 전체 평가셋
3회 반복의 p50/p95를 저장한다. GPU 시간 측정은 전후 동기화를 포함한다.

## 확인 범위

CPU 테스트는 두 백본의 forward/backward, 기존 ResNet state/output 일치,
고정 split, 전체 2D EPE 집계, 셔플/dropout/optimizer를 포함한 재개 일치,
오독 조기 종료, 설정·파일 변조 감지, 캐시 일치 및 W&B offline 저장을 다룬다.
실제 사전학습 가중치와 실제 이미지의 CPU 체크는 `check_cpu`가 별도 기록한다.
CPU 검증은 GPU 메모리, GPU 수치·처리량, 전체 epoch 시간이나 모델 정확도의
검증을 대신하지 않는다.

## 시간 추정의 근거

기존 ResNet 기록은 50 epoch에 약 17시간 40분, 기존 MobileNet spatial run은
RTX 2080 Ti 한 장에서 100 epoch에 6,631초(1시간 50분 31초)였다. 과거 두 구현의
head·데이터 로딩·평가 주기가 같다는 증거가 없으므로 모델별 속도 차이로 단정하지 않는다.
이전 일정 추정은 ResNet 시간을 전체에 적용한 보수적 값이다. 이번 코드의 캐시 최초
생성 시간, warm epoch 시간, 5 epoch마다 수행하는 디코딩 평가 시간을 GPU 단계에서
각각 측정해 4-run/12-run 일정과 1-GPU/2-GPU 일정을 다시 계산한다.
