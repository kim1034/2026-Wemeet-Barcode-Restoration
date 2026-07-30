# AI파트가 할 일

**담당: 김시선, 김대원**
**내 폴더: `wemeet/ai/`**

---

## 한 문단 요약

사진에서 **바코드를 찾아 잘라내고**(1단계), **구겨진 걸 펴고 반사를 지웁니다**(2단계). 잘라낸 결과와 복원한 결과를 SW파트에게 넘기면, SW파트가 그걸 번호로 바꿉니다.

우리는 **악기를 만드는 쪽**입니다. SW파트가 지휘자로서 우리 함수를 불러다 씁니다.

---

## 담당 파일

| 파일 | 무엇 |
|---|---|
| `wemeet/ai/detection.py` | 1단계 — 바코드 찾아서 자르기 |
| `wemeet/ai/restoration.py` | 2단계 — 펴고 복원하기 |
| `wemeet/ai/train_detection.py` | 탐지 모델 학습 |
| `wemeet/ai/train_restoration.py` | 복원 모델 학습 |
| `wemeet/ai/configs/` | 학습 설정 YAML |

---

## 절대 하면 안 되는 것

```python
from wemeet.sw.decoding import decode      # ← 안 됩니다
from wemeet.data.synthesis import warp     # ← 안 됩니다
```

**우리 파일에 `wemeet.sw`나 `wemeet.data`라는 글자를 쓰지 않습니다.** 필요한 게 있으면 `wemeet.schemas`에서 가져옵니다.

```python
from wemeet.schemas import DetectedBarcode, RestoredBarcode      # ← 이건 됩니다
```

**왜 그런가**: SW파트가 우리를 부릅니다. 우리가 SW파트를 부르면 서로 부르는 원이 생기고, 그러면 파이썬이 이런 에러를 냅니다.

```
ImportError: cannot import name 'DetectedBarcode' from partially initialized
module 'wemeet.sw.schemas' (most likely due to a circular import)
```

이 에러는 어느 파일을 먼저 실행하느냐에 따라 나타났다 사라집니다. 테스트는 통과하는데 서버만 안 뜨는 식이라 원인 찾기가 아주 어렵습니다.

걱정하지 않아도 됩니다. **실수로 쓰면 PR이 빨간 X로 막힙니다.** CI가 검사합니다.

---

## `wemeet/ai/detection.py` — 1단계 탐지

### 무엇을 하는 파일인가

사진 한 장을 받아서 바코드가 있는 부분만 잘라냅니다. 비스듬히 놓인 바코드는 **똑바로 세워서** 잘라냅니다.

### 구현할 함수

```python
def detect(image_bgr: np.ndarray) -> list[DetectedBarcode]:
```

| | 내용 |
|---|---|
| 입력 | BGR 이미지, 0~255, shape `(H, W, 3)`, dtype `uint8` |
| 출력 | `DetectedBarcode` 리스트 |
| 못 찾으면 | 빈 리스트 `[]` — **예외를 던지지 마세요** |
| 여러 개 찾으면 | `confidence` 높은 순으로 정렬해서 전부 반환 |

여러 개 중에 무엇을 쓸지는 **우리가 정하지 않습니다.** SW파트의 파이프라인이 정합니다. 우리는 찾은 걸 다 주고 정렬만 해줍니다.

### 반환할 것

```python
from wemeet.schemas import DetectedBarcode

DetectedBarcode(
    crop_bgr_uint8 = crop,      # 잘라낸 이미지. BGR 순서, 0~255
    angle_deg_ccw  = 12.5,      # 몇 도 돌려서 세웠는지. 도(degree), 반시계
    confidence     = 0.93,      # 0.0 ~ 1.0
)
```

**이름에 붙은 단위를 반드시 지키세요.**

- `crop_bgr_uint8` — **BGR** 순서입니다. OpenCV는 BGR, PyTorch는 RGB입니다. 이거 헷갈리면 복원 결과가 이상하게 나오고, 원인 찾는 데 며칠 걸립니다
- `angle_deg_ccw` — **도(degree)**, **반시계** 방향입니다. 라디안 아닙니다

틀린 값을 넣으면 그 자리에서 바로 에러가 납니다. `schemas.py`가 검사합니다.

### 지켜야 할 것

- 잘라낸 이미지는 회전을 보정해서 **바코드가 수평**이 되게 합니다
- **100ms 안에** 끝냅니다 (전체 500ms 중 우리 몫)
- `wemeet.sw` / `wemeet.data` 를 import하지 않습니다

### 아직 안 정해진 것

**어떤 모델을 쓸지 미정입니다.** 후보 3개를 비교해서 정합니다 → [`docs/decisions/0002-탐지-모델-선정.md`](../decisions/0002-탐지-모델-선정.md)

| 후보 | 장점 | 단점 |
|---|---|---|
| **전통 영상처리** (MSER + 최소외접사각형) | 학습이 필요 없어서 며칠이면 됨 | 배경이 복잡하면 약함 |
| **Ultralytics YOLO-OBB** | 자료가 많고 회전 박스를 기본 지원 | 라이선스가 AGPL-3.0 |
| **YOLOX / MMRotate** | 라이선스 자유 | 설치·학습이 어려움 |

**9월 1주차에 전통 영상처리부터 만듭니다.** 학습이 필요 없어서 빠르고, 이게 기준선이 됩니다. 그다음 YOLO로 학습해서 비교합니다.

그리고 이 규칙을 미리 정해뒀습니다 — **전통 영상처리가 아래를 만족하면 YOLO 학습을 아예 건너뜁니다.**

- 탐지 누락률 5% 이하
- 크롭 각도 오차 5° 이하
- 추론 100ms 이하

건너뛰게 되면 학습·튜닝·라벨링이 전부 사라집니다. 11월 일정이 빡빡하니 이게 최선일 수도 있습니다. **"딥러닝이 필요 없었다"는 결과도 보고서에 쓸 가치가 있습니다.**

### 막히면 이렇게 시작하세요

`TODO` 자리에 이걸 넣으면 파이프라인이 일단 돌아갑니다. 이미지 중앙 40%를 그냥 자르는 가짜 구현입니다.

```python
def detect(image_bgr: np.ndarray) -> list[DetectedBarcode]:
    h, w = image_bgr.shape[:2]
    crop = image_bgr[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
    return [DetectedBarcode(
        crop_bgr_uint8=crop,
        angle_deg_ccw=0.0,
        confidence=0.5,
    )]
```

가짜라도 넣어두면 전체가 돌아가고, 자기가 만든 게 어디에 쓰이는지 눈으로 볼 수 있습니다.

---

## `wemeet/ai/restoration.py` — 2단계 복원

### 무엇을 하는 파일인가

잘라낸 바코드를 받아서 **구겨진 걸 펴고**, **반사로 날아간 부분을 되살립니다.**

### 구현할 함수

```python
def restore(target: DetectedBarcode) -> RestoredBarcode:
```

| | 내용 |
|---|---|
| 입력 | `DetectedBarcode` **하나** (리스트 아님) |
| 출력 | `RestoredBarcode` **하나** |

리스트가 아니라 하나만 받습니다. 여러 개 중에 무엇을 복원할지는 파이프라인이 정합니다.

### 반환할 것

```python
from wemeet.schemas import RestoredBarcode

RestoredBarcode(
    image_gray_uint8 = restored,    # 흑백 1채널, 0~255, shape (H, W)
    source           = target,      # 받은 걸 그대로 넣어줍니다
)
```

**출력은 흑백입니다.** 입력은 BGR 3채널이었지만 출력은 회색 1채널입니다. 디코더가 흑백 패턴을 읽기 때문입니다.

### 지켜야 할 것

- **250ms 안에** 끝냅니다 (전체 500ms 중 우리 몫, 가장 큰 몫)
- 예쁘게 만드는 게 목표가 아닙니다. **기계가 읽을 수 있게** 만드는 게 목표입니다
- 특히 **바(bar)의 경계가 뚜렷해야** 합니다. 흐릿하게 뭉개지면 SSIM은 높아도 디코더가 못 읽습니다

이게 이 프로젝트의 학술적 차별점입니다. 기존 복원 연구는 사람 눈에 자연스럽게 만드는 데 집중했는데, 우리는 **기계 판독용 복원**을 합니다.

### 아직 안 정해진 것

모델 미정입니다 → [`docs/decisions/0003-복원-모델-선정.md`](../decisions/0003-복원-모델-선정.md)

| 후보 | 장점 | 단점 |
|---|---|---|
| **U-Net** | 단순하고 학습이 빠름. 스킵 커넥션이 바 경계를 잘 보존 | 아주 심한 훼손엔 한계 |
| Pix2Pix (GAN) | 결과가 선명함 | 학습이 불안정하고 오래 걸림 |
| Restormer / NAFNet | 성능이 좋음 | 무거워서 250ms를 넘길 위험 |

**U-Net부터 합니다.** SSIM이 목표에 못 미치면 Pix2Pix를 시도합니다.

### 손실 함수

이게 이 파트에서 가장 중요한 설계입니다. 픽셀 오차만 줄이면 흐릿한 결과가 나오는데, 흐릿하면 디코더가 못 읽습니다. 세 가지를 섞습니다.

| 항목 | 역할 |
|---|---|
| **L1 손실** | 기본 픽셀 오차 |
| **엣지 손실** (Edge / Gradient) | 바 경계를 뚜렷하게 유지 |
| **SSIM 손실** | 구조가 정답과 닮게 |

가중치는 실험으로 정하고 W&B에 기록합니다.

### 막히면 이렇게 시작하세요

CLAHE(명암 대비 개선)만 걸어주는 가짜 구현입니다. AI가 아니지만 파이프라인이 돌아갑니다.

```python
import cv2

def restore(target: DetectedBarcode) -> RestoredBarcode:
    gray = cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return RestoredBarcode(image_gray_uint8=clahe.apply(gray), source=target)
```

참고로 이 가짜 구현이 **기준선**이 됩니다. 우리가 만든 AI 모델이 이것보다 못하면 의미가 없습니다.

---

## 학습할 때 (`train_detection.py` / `train_restoration.py`)

### 파일을 두 개로 나눈 이유

탐지와 복원은 학습 방식이 완전히 다릅니다. 탐지는 Ultralytics API를 부르고, 복원은 커스텀 PyTorch 루프를 돌립니다. 한 파일에 넣으면 금방 수백 줄이 되고, 둘이서 같은 파일을 고쳐 충돌합니다.

### 학습 기록은 W&B에

상세는 [W&B 사용법](../external/wandb.md)에 있고, 핵심만 적으면 이렇습니다.

```python
import wandb

wandb.init(
    project="wemeet-barcode",
    entity="metro-wemeet",
    name="detection-yolo-obb-v2",        # <단계>-<모델>-v<번호>
    tags=["candidate"],
    config={
        "dataset_revision": "synth-v1",   # ← 이게 제일 중요합니다
        "seed": 42,                       # ← 이것도
        "model": "yolov8n-obb",
        "epochs": 100,
        "lr": 0.001,
    },
)
```

**`dataset_revision`을 반드시 남기세요.** 데이터가 계속 늘어나는 프로젝트입니다. 이걸 안 남기면 두 실험의 성능 차이가 데이터 때문인지 설정 때문인지 알 수 없습니다.

기록할 지표는 정확도만이 아닙니다.

```python
wandb.summary.update({
    "mAP50": 0.912,
    "mAP50_95": 0.624,
    "angle_error_deg": 2.4,          # OBB는 각도가 크롭 품질을 좌우합니다
    "inference_ms_p50": 38.4,        # 100ms 예산 확인
    "inference_ms_p95": 51.2,
})
```

**추론 시간을 꼭 재세요.** 정확해도 예산을 넘으면 탈락입니다.

### 실패한 실험도 남깁니다

`failed` 태그를 붙여서 남겨두세요. 지우면 다른 사람이 같은 실수를 반복합니다. "학습률 0.1은 발산한다"도 정보입니다.

### 모델 파일은 Hugging Face에

가중치 파일(`.pt`)은 git에 올라가지 않습니다(`.gitignore`에 있음). [Hugging Face](../external/huggingface.md)에 올립니다.

**학습 설정을 가중치와 항상 같이 올리세요.** 설정 없는 가중치는 반년 뒤에 재현이 안 돼서 쓸 수 없습니다.

---

## 우리 파트 일정

| 시기 | 할 일 |
|---|---|
| **9월 1주차** | 전통 영상처리 탐지 만들기 (학습 없는 기준선) |
| **9월 2주차** | 탐지 모델 학습 시작 |
| **9월 3주차** | 전통 vs 학습 비교 → 조기 확정 규칙 판정 |
| **9월 4주차** | 탐지 모델 확정 → `decisions/0002` 기록 |
| **10월 1~2주차** | 복원 모델(U-Net) 학습 |
| **10월 4주차** | 추론 시간 측정. 500ms 넘으면 경량화 |
| **11월** | 필요 시 경량화, 최종 평가 지원 |

---

## 자주 헷갈리는 것

**Q. `schemas.py`를 고쳐야 할 것 같은데요?**
혼자 고치지 마세요. SW파트 승인이 있어야 머지됩니다. Issue를 만들고 `decision` 라벨을 붙여서 상의하세요. 이 파일이 우리와 SW파트를 잇는 유일한 약속입니다.

**Q. 탐지가 여러 개 찾았는데 어떻게 하죠?**
전부 반환하고 `confidence` 순으로 정렬만 하세요. 무엇을 쓸지는 파이프라인이 정합니다.

**Q. 복원이 실패하면 예외를 던져야 하나요?**
아니요. 파이프라인이 알아서 처리합니다. 예외를 던지면 SW파트가 잡아서 원본 크롭으로 대체합니다.

**Q. BGR인지 RGB인지 자꾸 헷갈립니다.**
변수 이름에 적혀 있습니다. `crop_bgr_uint8`은 BGR, `image_gray_uint8`은 흑백입니다. 이름을 믿으세요.
