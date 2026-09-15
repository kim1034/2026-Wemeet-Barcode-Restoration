# Stage 2 백본 설계 및 비교 실험 계획 — 2026-09-15

- **상태**: 설계 문서 및 비교 실험 계획
- **범위**: 코드 구현 없음
- **관련 결정**: [0003. 기하 추정 모델 선정](../../decisions/0003-기하-추정-모델-선정.md)
- **관련 인계 규격**: [Stage 2 → Stage 3](../../stage2-handoff.md)

> 이 문서는 최종 배포 모델을 확정하는 결과 보고서가 아니다. `ResNet18-C3`를
> Stage 2 v1의 구현 기준으로 삼고, 같은 조건에서 두 경량 백본을 비교하기 위한
> 설계와 측정 방법을 기록한다.

## 1. 결론 요약

이번 비교의 기준 백본은 **ImageNet pretrained ResNet18-C3**로 둔다.
`ResNet18-C3`는 torchvision의 공식 모델 이름이 아니라, `resnet18`의
`layer3`까지 사용하는 구성을 이 문서에서 부르는 이름이다.

비교 모델은 다음 세 가지다.

| 모델 | 비교에서의 역할 |
|---|---|
| **ResNet18-C3** | Stage 2 v1 구현 기준 및 정확도 기준선 |
| **MobileNetV3-Small** | 경량·지연시간 기준선 |
| **EfficientNet-B0** | 정확도와 효율의 절충 후보 |

세 모델은 같은 입력 전처리, 같은 Geometry Head, 같은 `16×3` 제어점,
같은 TPS·디코더 경로로 비교한다. 비교 전에는 어느 모델이 최종 배포에 적합한지
알 수 없으므로, `ResNet18-C3`를 **최종 배포 모델로 확정했다고 쓰지 않는다**.

## 2. 문제 정의

Stage 2의 질문은 이미지의 클래스를 맞히는 것이 아니다.

```text
왼쪽은 얼마나 휘었는가?
중앙은 어떻게 압축됐는가?
오른쪽 제어점은 어디에 있는가?
```

현재 파이프라인에서 Stage 2는 다음 흐름의 숫자 예측 단계다.

```text
DetectedBarcode crop
        ↓
Stage 2: Geometry Estimation
        ↓
GeometryField (좌표와 메타데이터)
        ↓
Stage 3: TPS / OpenCV remap
        ↓
Stage 4: barcode decode
```

Stage 2는 이미지를 생성하지 않는다. AI는 `src` 제어점 좌표를 예측하고,
실제 픽셀 이동은 Stage 3의 OpenCV가 수행한다. 생성된 픽셀을 디코더에 넣지
않으므로, 기하 추정 모델이 그럴듯한 바코드 픽셀을 새로 만들어 오독을 유도하는
경로를 설계하지 않는다.

## 3. 이미 확정된 Stage 2 조건

### GeometryField와 제어점

2026-09-14 제어점 실험의 확정값은 **`16×3 = 48점`**이다.

- `dst`: 펼친 뒤의 regular `16×3` 격자. 구조적으로 정해지며 모델이 예측하지 않는다.
- `src`: 현재 휘어진 crop에서 각 `dst` 내용이 있는 위치. 모델이 예측한다.
- 각 점은 `(x, y)`이므로 모델이 예측하는 `src` 좌표는 `48×2 = 96개`다.
- 배열은 행 우선이다. 위쪽 행 16점, 가운데 행 16점, 아래쪽 행 16점 순서다.
- 좌표는 `DetectedBarcode.crop_bgr_uint8` 원본 crop 기준의 normalized coordinate다.
- 좌표 순서는 `(x, y)`이고 y는 이미지처럼 아래 방향으로 증가한다. Stage 3는
  normalized 좌표에 각각 `crop_width-1`, `crop_height-1`을 곱해 pixel 좌표로 바꾼다.
- `dst` 범위는 `0.0~1.0`, `src`는 현재 계약에 따라 `-0.5~1.5`를 허용한다.

계약은 [wemeet/schemas.py](../../../wemeet/schemas.py)에 있고 이번 문서 PR에서
변경하지 않는다. 데이터 생성 코드의 기준값은
[`N_X, N_Y = 16, 3`](../../../wemeet/data/synthesis.py)이다.

### Stage 2 v1 입력 제안

현재 비교 실험에서 고정할 입력 전처리의 제안은 다음과 같다.

```text
crop_bgr_uint8
        ↓  BGR 기준 grayscale 변환
grayscale
        ↓  강제 resize
H=160, W=384
        ↓  1채널을 3채널로 repeat
3-channel image
        ↓  ImageNet normalization
backbone
```

- 1D 바코드에서는 색상보다 검정·흰색 구조, 막대 폭, 경계가 중요하므로 grayscale을 제안한다.
- 3채널 repeat은 ImageNet pretrained 백본의 첫 convolution을 그대로 사용할 수 있게 한다.
- `160×384 = 61,440`픽셀로, 기존 `256×256 = 65,536`픽셀과 비용이 비슷하면서
  바코드의 가로 정보를 더 많이 보존한다.
- `160×384`는 v1 **제안값**이다. `256×256` 대 `160×384`는 백본 비교와 섞지
  않고 별도의 입력 해상도 ablation으로 검증한다.

강제 resize는 종횡비를 바꾸지만 학습과 추론에서 같은 전처리를 사용한다.
letterbox를 도입하려면 패딩 좌표를 원본 crop 기준으로 환산하는 별도 검증이 필요하다.

## 4. ResNet18-C3 선정 이유

입력이 `H=160, W=384`일 때 torchvision ResNet18의 주요 spatial feature는
다음 크기가 된다. 구조 확인 기준은 torchvision 0.29.0의
[resnet.py](https://github.com/pytorch/vision/blob/v0.29.0/torchvision/models/resnet.py)다.

| 위치 | 출력 크기 | 이 문서에서의 사용 |
|---|---:|---|
| 입력 | `160×384` | 입력 |
| stem + maxpool / Layer1 | `40×96` | 사용하지 않음 |
| Layer2 | `20×48` | 사용하지 않음 |
| **Layer3** | **`10×24`, 256채널** | **사용 (`ResNet18-C3`)** |
| Layer4 | `5×12`, 512채널 | 사용하지 않음 |
| GAP / FC | `1×1` / vector | 사용하지 않음 |

가로 제어점은 16개다. Layer4까지 내려가면 feature width가 약 12가 되어
`12 spatial cells → 16 control points`가 된다. Layer3에서는 width가 약 24라서
`24 spatial cells → 16 control points`로 더 자연스럽게 대응시킬 수 있다.

이는 spatial geometry regression에 대한 설계 가설이다. Layer3가 실제로 더
좋은지는 아래 비교 실험으로 확인해야 하며, ImageNet 분류 성능만으로 증명되는
결론이 아니다.

## 5. Spatial feature를 사용하는 이유

일반적인 ImageNet 분류 경로는 대략 다음과 같다.

```text
spatial feature
        ↓
GAP (Global Average Pooling)
        ↓
1×1 vector
        ↓
classifier
```

GAP는 왼쪽·중앙·오른쪽의 위치 정보를 한 벡터로 섞는다. 그러나 Stage 2는
위치별 변형을 예측해야 한다. 따라서 기본 제안은 GAP+FC direct regression이
아니라 spatial feature를 유지하는 Geometry Head다.

다만 GAP+FC direct regression은 **단순 baseline**으로 비교할 수 있다. 이
baseline의 목적은 spatial head의 이득을 확인하는 것이며, 3-way 백본 비교의
주요 축과 섞지 않는다.

## 6. Geometry Head 제안

세 백본의 feature 채널 수가 다르므로, 공통 Head에 넣기 전에 백본별 `1×1`
channel adapter로 256채널에 맞춘다. 이 adapter를 포함한 뒤의 Geometry Head는
모든 모델에서 동일하게 유지한다.

```text
backbone OS=16 feature: B × C × 10 × 24
        ↓
1×1 channel adapter: C → 256
        ↓
3×3 Conv: 256 → 128
        ↓
Adaptive Pool: 3 × 16
        ↓
B × 128 × 3 × 16
        ↓
3×3 Conv: 128 → 64
        ↓
1×1 Conv: geometry parameters
        ↓
structured coordinate decoder
        ↓
48 × (x, y) = 96개 src 좌표
```

ResNet18-C3의 `C=256`, MobileNetV3-Small의 `C=48`, EfficientNet-B0의
`C=112`를 각각 같은 Head 입력으로 변환한다. Adapter와 Head의 파라미터는
모델 비용에 포함한다. 이 문서의 기본 protocol은 ResNet18-C3에도 `256→256`
adapter를 두어 세 모델 모두 `C→256` adapter를 거치게 하는 것이다. Adapter와
downstream Head의 구조·초기화·학습 정책은 같게 하고, 각 모델의 가중치는 같은
학습 조건에서 별도로 학습한다. ResNet adapter를 identity로 바꾸는 것은 별도
ablation으로만 다룬다.

### 좌표 표현

좌표를 독립적으로 회귀하면 한 행에서 `x1 < x2 < ... < x16`이 깨질 수 있다.
현재 제안은 다음과 같은 positive interval 표현이다.

```text
raw interval parameters
        ↓
softplus
        ↓
positive intervals
        ↓
cumulative sum
        ↓
monotonic x coordinates
```

이 inductive bias는 각 행의 `src_x` 단조성을 구조적으로 돕는다. 다만 현재
`GeometryField`는 crop 밖 `[-0.5, 1.5]`까지 `src`를 허용하므로, offset·범위
처리와 endpoint parameterization은 구현 전에 검증해야 한다. 이 PR은 위 개념을
기록할 뿐, 검증되지 않은 수식을 확정 구현으로 정하지 않는다.

초기 head가 `src ≈ dst`가 되도록 identity geometry 근처에서 시작하는 것도
제안한다. 실제 초기화 방식은 모델 구현 PR에서 확인한다.

### Loss 제안

기본 loss 후보는 Smooth L1이다. 좌표를 normalized 값 그대로만 비교하지 않고,
원본 crop의 pixel 단위 오차로 환산해 기록한다.

```text
error_x_px = (pred_x - gt_x) × (crop_width  - 1)
error_y_px = (pred_y - gt_y) × (crop_height - 1)
```

기존 제어점 실험의 sigma가 pixel 단위였으므로, 크기가 다른 crop에서도 같은
기준으로 비교할 수 있다. 약 2px 수준의 effective error 영역은 중요한 관찰
기준이지만, 학습 결과 전에 보장되는 성능 목표로 쓰지 않는다.

## 7. MobileNetV3-Small 비교 이유

MobileNetV3-Small은 **경량·latency 비교 후보**다.

- ResNet18-C3와 비슷한 구제율·좌표 정확도를 더 적은 계산량으로 얻는지 확인한다.
- 파라미터, 메모리, CPU 지연시간 측면에서 배포에 유리할 가능성이 있다.
- 지나친 경량화로 local spatial geometry 표현력이 부족하면 좌표 오차가 커질 수 있다.

### 사용할 feature

현재 저장소에는 `torchvision`이 아직 설치되어 있지 않다. 따라서 구현 PR에서
torch/torchvision 버전을 먼저 고정하고 shape test를 추가해야 한다. 아래 구조는
torchvision **0.29.0 소스의 `mobilenet_v3_small`**를 확인하고, 입력
`(1, 3, 160, 384)`를 실제 feature sequence에 통과시켜 확인한 값이다. 확인한
소스는 [mobilenetv3.py](https://github.com/pytorch/vision/blob/v0.29.0/torchvision/models/mobilenetv3.py)다.

| torchvision 경로 | stride | 출력 |
|---|---:|---:|
| `model.features[8]`까지 (`model.features[:9]`) | 16 | `B×48×10×24` |
| `model.features[9]`부터 | 32 | `B×96×5×12` |

따라서 classifier와 average pool을 사용하지 않고 `model.features[:9]`의
출력을 공통 Geometry Head에 넣는다. `features[8]`은 output stride 16의 마지막
intermediate feature이며, `features[9]` 이후는 Layer4에 해당하는 stride 32
구간으로 보아 사용하지 않는다.

## 8. EfficientNet-B0 비교 이유

EfficientNet-B0는 **정확도와 효율의 절충 후보**다.

- MobileNetV3-Small의 표현력이 부족하고 ResNet18-C3가 필요 이상으로 크다면
  중간 선택지가 될 수 있다.
- ImageNet classification 정확도가 높다고 geometry regression도 반드시 우수한
  것은 아니므로, 판단은 우리 Stage 2 지표로 한다.

### 사용할 feature

같은 torchvision 0.29.0 소스와 입력 shape 검증에서 `efficientnet_b0`의
feature stage는 다음처럼 확인됐다. 확인한 소스는
[efficientnet.py](https://github.com/pytorch/vision/blob/v0.29.0/torchvision/models/efficientnet.py)다.

| torchvision 경로 | stride | 출력 |
|---|---:|---:|
| `model.features[5]`까지 (`model.features[:6]`) | 16 | `B×112×10×24` |
| `model.features[6]`부터 | 32 | `B×192×5×12` |

따라서 `model.features[:6]`의 마지막 stride-16 feature를 사용하고,
classifier와 average pool은 사용하지 않는다. 출력 112채널은 `1×1`
adapter로 256채널에 맞춘다.

## 9. 3-way fair comparison protocol

한 실험에서 백본 외 변수를 바꾸지 않는다.

| 고정할 항목 | 고정 내용 |
|---|---|
| 데이터셋·split | 같은 dataset revision, 같은 train/validation split |
| 합성 분포 | 현재 v1 구성인 목표 50% / 기하 관측 가능 불가 35% / 1차 성공 15%를 공통 적용. 포화로 정보가 사라진 불가는 0% |
| 입력 | grayscale → `160×384` 강제 resize → 3-channel repeat → 같은 ImageNet normalization |
| 백본 feature | 각 모델의 stride-16 intermediate feature만 사용 |
| channel adapter | `C→256` 방식과 학습 여부를 고정하고 모델 비용에 포함 |
| Geometry Head | 위의 spatial Head 구조와 weight initialization 정책을 동일 적용 |
| 출력 | `16×3=48`점, 모델은 `src` 96개를 예측하고 `dst`는 고정 격자로 생성 |
| loss | 같은 Smooth L1 정의와 pixel 환산 방법 |
| 학습 | optimizer, learning rate, epoch, batch size, augmentation, seed 목록을 고정 |
| pretrained | 세 모델 모두 동일한 ImageNet pretrained 사용 정책 |
| 보정·디코딩 | 같은 TPS, 같은 OpenCV remap, 같은 decoder와 retry 순서 |
| 평가 | 같은 synthetic validation recipe와 같은 real `dev` set |
| 실행 환경 | 같은 기계, batch size 1, warm-up·반복 횟수·동시 프로세스 조건을 고정 |

비교 흐름은 다음과 같다.

```text
Same Input
    ↓
ResNet18-C3 / MobileNetV3-Small / EfficientNet-B0
    ↓
Same channel interface
    ↓
Same Geometry Head
    ↓
Same 16×3 control-point contract
    ↓
Same TPS / OpenCV remap
    ↓
Same decoder
```

평가 세트는 두 종류로 나눈다.

1. **백본 차이를 보기 위한 target-focused set**: 1차 디코딩은 실패했지만
   밀집 대응장 `G`로 remap하면 읽히는 샘플에 집중한다. 이 구간이 모델이 실제로
   구제할 수 있는 범위이므로, 샘플 생성 후 이 비율이 100%에 가깝도록 선별한다.
2. **전체 파이프라인 set**: 1차 성공, 구제 가능, 원리적 불가를 포함해 실제 운영
   분포에서의 종합 성능과 조기 종료율을 기록한다.

항등 필드와 4점 호모그래피는 백본 3-way와 별도의 참조 baseline으로 측정한다.
아직 이 문서에는 학습·비교 결과를 채우지 않는다.

## 10. 평가 지표

최종 선택 순서는 다음과 같다.

```text
1. 난인식 구제율
2. control-point pixel error
3. p95 latency
4. model size / memory
```

| 우선순위 | 지표 | 정의 및 측정 대상 |
|---:|---|---|
| 1 | **난인식 구제율** | 분모는 1차 decode 실패 샘플, 분자는 Stage 2 + Stage 3 뒤 정답으로 decode된 샘플 |
| 2 | **Control-point pixel error** | 합성 validation에서 mean pixel error, RMSE, P95를 원본 crop pixel 기준으로 측정 |
| 3 | **Stage 2 latency** | 같은 실행 환경에서 batch size 1의 p50, p95. warm-up과 측정 반복을 사전에 고정 |
| 4 | **Model cost** | trainable parameter 수, weight 파일 크기, inference memory |

조기 종료율(1차 decode 성공률)과 전체 decode 성공률도 함께 기록한다. 1차에서
이미 읽힌 샘플은 Stage 2 모델의 차이를 보여주지 않으므로, 전체 성공률만으로
백본을 판단하지 않는다.

결과를 채울 표는 다음과 같다.

| 모델 | feature | 좌표 오차(px) | 구제율 | Stage 2 p50/p95 (ms) | params / memory | W&B run |
|---|---|---:|---:|---:|---|---|
| ResNet18-C3 | `layer3`, `B×256×10×24` | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| MobileNetV3-Small | `features[:9]`, `B×48×10×24` | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| EfficientNet-B0 | `features[:6]`, `B×112×10×24` | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

## 11. 아직 미확정인 항목

- 세 백본 중 최종 배포 모델. 비교 실험 전에는 결정하지 않는다.
- `torch`·`torchvision`의 정확한 버전, ImageNet weight revision, CPU/GPU 실행 정책.
  현재 저장소 환경에는 `torchvision`이 없다.
- 기본 protocol의 학습형 channel adapter와 별도로 ResNet adapter를 identity로
  두는 ablation을 실행할지 여부.
- softplus + cumulative sum의 정확한 offset·endpoint parameterization 및
  독립 좌표 회귀 baseline과의 ablation.
- identity initialization의 구체적인 구현.
- Smooth L1의 pixel scaling·가중치와 선택적 reprojection loss.
- `256×256`과 `160×384` 중 최종 입력 해상도, letterbox 도입 여부.
- v1에서 `confidence`를 상수로 둘 때의 pipeline 처리 정책.
- 기준 기계와 실제 Stage 2 latency 예산. 결과는 동일한 기준 기계에서 측정해야 한다.
- 실촬영 `dev` set에서의 최종 구제율과 decoder 결정에 따른 평가 경계.

## 12. 다음 실험 계획

1. 구현 PR에서 torch/torchvision 버전과 ImageNet weight를 고정하고, 세 extraction
   path가 `(B, C, 10, 24)`를 내는 shape test를 추가한다.
2. 공통 channel adapter와 Geometry Head를 구현한다. `dst` 고정 격자, `src` 96개
   출력, 좌표 방향, 단조성 검사를 계약에 맞춰 검증한다.
3. 같은 dataset revision·split·seed로 항등 필드와 4점 호모그래피 baseline을
   먼저 기록한다.
4. 같은 조건에서 세 백본을 학습하고 synthetic target-focused set과 전체 set을
   각각 평가한다.
5. 기준 기계에서 warm-up 후 p50/p95 latency와 parameter·memory를 단독 실행으로
   측정하고 W&B에 dataset revision과 설정을 남긴다.
6. 결과를 이 표와 `0003`에 채운다. 구제율·pixel error·p95 latency·cost 순으로
   검토해 최종 배포 backbone을 재확인한다.

이번 PR에서는 `wemeet/ai/geometry.py`와 training code를 만들지 않으며,
`torch` dependency·`schemas.py`·합성 코드·control-point 수를 변경하지 않는다.
