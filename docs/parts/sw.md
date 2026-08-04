# SW파트가 할 일

**담당: 김종연, 이도훈**
**내 폴더: `wemeet/sw/`, `web/`**

---

## 한 문단 요약

AI파트가 만든 함수들을 **불러다 순서대로 이어 붙이고**(파이프라인), AI가 예측한 휘어짐 정보로 **이미지를 실제로 펴고**(기하 보정), 펴진 이미지를 **바코드 번호로 바꾸고**(디코딩), 결과를 **화면에 띄웁니다**(서버 + React).

**이미지를 실제로 변형하는 것은 우리입니다.** AI는 "얼마나 휘었는지"만 숫자로 주고, OpenCV로 펴는 것은 우리 몫입니다.

우리는 **지휘자**입니다. AI파트가 악기를 만들고 우리가 연주 순서를 정합니다. 그래서 이 프로젝트가 실제로 돌아가느냐는 우리 손에 달려 있습니다.

---

## 담당 파일

| 파일 | 무엇 |
|---|---|
| `wemeet/sw/pipeline.py` | **4단계를 이어 붙이는 곳** (우리 파트의 핵심) |
| `wemeet/sw/rectify.py` | 3단계 — OpenCV로 실제로 펴기 |
| `wemeet/sw/decoding.py` | 4단계 — 반사 대응 + 번호로 바꾸기 |
| `wemeet/sw/server.py` | FastAPI 서버 |
| `web/` | React 화면 |

---

## 우리가 쓸 수 있는 것

우리는 AI파트를 부를 수 있습니다.

```python
from wemeet.ai.detection import detect            # ← 됩니다
from wemeet.ai.geometry import estimate_geometry  # ← 됩니다
from wemeet.schemas import PipelineResult         # ← 됩니다
from wemeet.data.synthesis import warp            # ← 안 됩니다
```

`wemeet.data`는 못 씁니다. 데이터 생성 스크립트가 서버 코드에 끌려 들어오면 안 되기 때문입니다.

---

## `wemeet/sw/pipeline.py` — 4단계 이어 붙이기

### 이 파일이 가장 중요합니다

여기가 프로젝트의 심장입니다. 그리고 **9월·10월에 이 파일을 잘 만들어두면 "통합"이라는 힘든 작업이 사라집니다.**

### 구현할 함수

```python
def run(image_bgr: np.ndarray) -> PipelineResult:
```

기본 흐름은 이렇습니다.

```python
from wemeet.ai.detection import detect
from wemeet.ai.geometry import estimate_geometry
from wemeet.schemas import PipelineResult, RectifiedBarcode
from wemeet.sw.decoding import decode
from wemeet.sw.rectify import apply_field

def run(image_bgr):
    boxes = detect(image_bgr)                 # 1단계 (AI)
    if not boxes:
        return ...                            # 못 찾았을 때 (아래 표 참고)
    best = boxes[0]                           # confidence 최상위 하나만

    # 1차 시도 — 보정 없이 먼저 읽어본다. 여기서 끝나면 약 150ms
    plain = RectifiedBarcode(
        image_gray_uint8=cv2.cvtColor(best.crop_bgr_uint8, cv2.COLOR_BGR2GRAY),
        source=best,
        field=None,                           # None = 보정하지 않았다
    )
    result = decode(plain)
    if result.text is not None:
        return PipelineResult(image_bgr, best, plain, result)

    field  = estimate_geometry(best)          # 2단계 (AI) — 숫자만 받는다
    fixed  = apply_field(best, field)         # 3단계 (우리) — 실제로 편다
    result = decode(fixed)                    # 4단계 (우리)
    return PipelineResult(
        original_bgr_uint8=image_bgr,
        detected=best,
        rectified=fixed,
        decode=result,
    )
```

**1차 디코딩을 AI에게 맡기지 않는 이유**: AI가 디코더를 부르면 `wemeet.sw` import가 되어 규칙 위반입니다. 그리고 "먼저 읽어보고 안 되면 보정한다"는 조율 판단이므로 지휘자인 우리 일입니다.

### `PipelineResult`를 반환하는 이유

`DecodeResult`(번호만 들어 있는 것)를 반환하면 **펴진 이미지가 화면까지 도달하지 못합니다.**

그런데 이 프로젝트가 상용 솔루션(Scandit 등)과 차별화되는 지점이 바로 그겁니다. 상용 제품은 내부가 블랙박스라서 실패했을 때 작업자가 아무것도 볼 수 없습니다. 우리는 **"여기까지 펴봤는데 안 읽혔다"를 보여줘서** 작업자가 즉시 판단할 수 있게 합니다.

그래서 원본·검출결과·펴진이미지·번호를 다 담아서 반환합니다. 그리고 `rectified.field` 에 "무엇을 근거로 어떻게 폈는지"가 함께 들어 있어서 화면에 제어점을 그려 보여줄 수도 있습니다.

### 실패했을 때 무엇을 반환하나

**절대 예외를 던지지 않습니다.** 어떤 상황에서도 `PipelineResult`를 반환합니다.

현장 컨베이어에서 예외가 올라오면 서비스가 멈춥니다. 그게 곧 노리드 존 정체이고, 우리가 없애려는 바로 그 문제입니다. **실패는 예외가 아니라 값으로 표현합니다.**

| 상황 | 어떻게 하나 |
|---|---|
| 이미지 로드 실패 | `failure_reason="invalid_input"`, 나머지 `None` |
| **검출 0개** | 이후 단계 생략. `failure_reason="not_detected"` |
| **검출 2개 이상** | `confidence` 1등만 처리. `candidate_count`에 총 개수 기록 |
| **1차 디코딩 성공** | 기하 추정·보정을 **건너뛰고 즉시 종료**. `rectified.field=None` |
| 기하 추정 신뢰도 낮음 | 보정을 건너뛰고 실패 처리. `degraded=True` |
| 기하 추정이 예외를 던짐 | 예외를 잡고 **보정 없이** 진행. `degraded=True` |
| 디코딩 실패 | **보정 파라미터를 바꿔** 최대 3회 재시도 |
| 재시도 후에도 실패 | `failure_reason="decode_failed"` |
| **500ms 초과** | **중단하지 않고 끝까지 진행.** 시간만 기록 |

두 개만 부연합니다.

**탐지 2개 이상은 흔합니다.** 박스에 송장 바코드와 상품 바코드가 같이 붙어 있습니다. 1등만 처리하고 총 개수를 기록해두면, 나중에 "여러 개 처리가 필요한가"를 데이터로 판단할 수 있습니다.

**500ms를 넘어도 중단하지 않습니다.** 중단하면 읽을 수 있었던 화물을 스스로 버리는 셈이고, 무엇보다 초과 원인을 알 수 없게 됩니다. 시간만 기록하고 예산 준수는 평가 단계에서 통계로 판정합니다.

### 시간을 단계별로 재세요

```python
result.decode.stage_ms = {
    "detect": 95.0,
    "decode_first": 22.0,     # 1차 시도. 여기서 성공하면 나머지가 없다
    "estimate": 140.0,
    "warp": 18.0,
    "decode": 26.0,
}
```

**예산이 정해져 있습니다.**

| 단계 | 키 | 예산 |
|---|---|---|
| 검출 + 기울기 보정 | `detect` | 110 ms |
| **1차 디코딩** (조기 종료 지점) | `decode_first` | 40 ms |
| 기하 추정 (AI) | `estimate` | 150 ms |
| 기하 보정 (우리) | `warp` | 20 ms |
| 디코딩 + 재시도 최대 3회 | `decode` / `retry` | 150 ms |
| 여유 | — | 30 ms |
| **합계** | | **500 ms** |

**1차에서 읽히면 약 150ms에 끝납니다.** 사전 실험에서 수직으로만 휜 바코드는 보정 없이도 읽히는 것을 확인했으니, 이 경로를 타는 비율이 상당할 것으로 봅니다.

단계별로 재두면 나중에 누가 초과했는지 바로 나옵니다. 안 재두면 "느린데 어디가 느린지 모르겠다"가 됩니다.

### 먼저 할 일 — 가짜로라도 끝까지 이어놓기 (9월 3주차)

**AI 모델을 기다리지 마세요.** 각 단계를 가짜 구현으로 채워서 파이프라인이 실제로 돌아가게 만듭니다.

```python
def run(image_bgr):
    boxes = detect(image_bgr)                 # 가짜: 중앙 40% 자르기
    field = estimate_geometry(boxes[0])       # 가짜: 항등 필드 (안 휘었다고 답함)
    fixed = apply_field(boxes[0], field)      # 이건 처음부터 진짜 (OpenCV)
    return decode(fixed)                      # 이것도 처음부터 진짜
```

**3·4단계는 학습이 없으니 처음부터 진짜로 만들 수 있습니다.** 가짜가 필요한 건 AI 두 단계뿐입니다.

이렇게 해두면 10월에 AI파트가 하는 일은 `detection.py` 안의 가짜를 진짜로 바꿔 넣는 것뿐입니다. **`pipeline.py`는 건드리지 않습니다.** "통합"이라는 별도 작업이 없어지고, 이게 11월 일정을 지킬 수 있는 이유입니다.

그리고 더 중요한 게 있습니다. BGR/RGB 같은 문제가 **9월에 드러납니다.** 가짜끼리라도 데이터가 실제로 흘러가니까요. 10월에 처음 붙여보는 것과 9월부터 계속 붙어 있는 것의 차이가 프로젝트 성패를 가릅니다.

일찍 시작할 수 있는 이유가 하나 더 있습니다 — **3·4단계는 학습이 없어서 우리 쪽 두 파일은 처음부터 진짜입니다.** 가짜가 필요한 건 AI 두 단계뿐입니다.

---

## `wemeet/sw/rectify.py` — 3단계 기하 보정

### 무엇을 하는 파일인가

AI가 준 제어점으로 **이미지를 실제로 펴는 곳**입니다. 학습이 없는 결정론적 코드라 **AI 모델을 기다리지 않고 지금 바로 진짜로 만들 수 있습니다.**

### 구현할 함수

```python
def apply_field(
    target: DetectedBarcode,
    field: GeometryField,
    interpolation: int = cv2.INTER_CUBIC,
) -> RectifiedBarcode:
```

`interpolation`을 인자로 노출하세요. 재시도에서 이 값을 바꿉니다.

### 좌표 변환을 먼저 하세요

`GeometryField`의 제어점은 **0.0~1.0 정규화 좌표**입니다. TPS를 풀려면 픽셀로 바꿔야 합니다.

```python
h, w = target.crop_bgr_uint8.shape[:2]
scale = np.array([w - 1, h - 1])
src_px = field.control_points_src_norm * scale
dst_px = field.control_points_dst_norm * scale
```

**방향을 지키세요.** `dst`가 펴진 격자, `src`가 지금 있는 위치입니다. 뒤집으면 왜곡이 두 번 적용되는데, **약한 왜곡에서는 우연히 읽혀서 버그를 놓칩니다.** 결과가 들쭉날쭉하면 이걸 먼저 의심하세요.

### TPS는 직접 계산합니다

`cv2.createThinPlateSplineShapeTransformer`는 **`opencv-python`에 없습니다.** contrib(78MB)에만 있어서 쓰지 않기로 했습니다.

numpy로 30줄입니다. **아래 코드는 사전 실험에서 검증된 것이라 그대로 쓰면 됩니다.** 전문은 [기하 파이프라인 설계 §9](../superpowers/specs/2026-08-04-geometry-pipeline-design.md)에 있습니다.

```python
def tps_flow(src, dst, shape, reg=0.0):
    """제어점 대응으로 dense flow map 을 만든다. 반환값을 remap 에 넣는다."""
    n = len(dst)
    d = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(d > 0, d**2 * np.log(d**2), 0.0)
    k += reg * np.eye(n)
    p = np.hstack([np.ones((n, 1)), dst])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
    ...   # 격자에서 평가 → map_x, map_y
```

적용은 이렇습니다.

```python
map_x, map_y = tps_flow(src_px, dst_px, (h, w))
gray = cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY)
fixed = cv2.remap(gray, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)
```

`field.method == "perspective"` 면 `cv2.getPerspectiveTransform` + `cv2.warpPerspective`를 씁니다. 제어점 4개일 때만 유효합니다.

### 지켜야 할 것

- **20ms 안에** 끝냅니다. 재시도에서 3번 불릴 수 있습니다
- 출력은 **흑백 1채널**입니다 (`RectifiedBarcode`가 검사합니다)
- `field`를 반환값에 그대로 담아주세요. 화면에서 제어점을 그릴 때 씁니다
- **새 픽셀을 만들지 마세요.** `remap`은 기존 픽셀을 이동·보간할 뿐입니다. 인페인팅이나 생성을 붙이면 이 설계의 근거가 무너집니다

### 성능이 안 나오면

격자 계산이 지배적입니다. `tps_flow`를 1/4 크기로 계산하고 `cv2.resize`로 늘리면 됩니다. 왜곡이 매끄러워서 정확도 손실이 거의 없습니다.

### 가장 싼 검증 방법

**항등 대응을 넣어보세요.** `src == dst`면 `map_x[y, x] ≈ x` 여야 합니다.

```python
map_x, map_y = tps_flow(grid, grid, (h, w))
assert abs(map_x[10, 20] - 20) < 0.01
```

이게 틀리면 TPS 구현이 잘못된 것입니다. 이미지로 확인하기 전에 이것부터 하세요.

---

## `wemeet/sw/decoding.py` — 4단계 디코딩 + 반사 대응

### 구현할 함수

```python
def decode(image: RectifiedBarcode) -> DecodeResult:
```

흐름은 이렇습니다.

```
펴진 흑백 이미지 (또는 미보정 크롭)
      ↓
반사 대응 — 바 높이 방향으로 여러 행을 읽어
           포화된 행은 버리고 나머지로 합친다
      ↓
CLAHE (명암 대비 개선)
      ↓
적응형 이진화 (흑백 두 값으로)
      ↓
디코더 호출
```

재시도 루프는 `decode()` 안이 아니라 **`pipeline.py`가 돕니다** — 보정을 다시 하기 때문입니다.

### 반사 대응 — 이건 우리가 새로 맡은 일입니다

기하 보정은 픽셀을 **옮기는** 것이라 반사로 하얗게 날아간 곳은 옮겨도 하얗습니다. 그런데 **1D 바코드는 바 높이 방향으로 정보가 중복됩니다.**

```
펴진 이미지에서 바가 수직으로 정렬돼 있다
    ↓
행마다 품질을 본다 (평균 밝기가 너무 높거나, 분산이 너무 낮은 행 = 반사)
    ↓
살아 있는 행만 골라 열마다 값을 정한다 (중앙값, 또는 대비 최대 행)
    ↓
1D 신호 하나로 디코딩
```

**있는 픽셀 중에서 고르는 것**이므로 없는 정보를 만들지 않습니다. 이게 우리가 생성 모델을 쓰지 않고도 반사를 다루는 방법입니다.

**기하 보정이 선행 조건입니다.** 바가 휘어 있으면 "같은 열"이 정의되지 않아 행을 합칠 수 없습니다. 즉 3단계가 이 기법을 가능하게 만듭니다.

한계도 알아두세요 — 반사가 **전체 높이를 덮은 열**은 정보가 진짜로 없습니다. 실패로 보고하는 게 맞습니다. 그리고 QR 같은 2D 코드는 높이 중복이 없어서 이 방법을 쓸 수 없습니다.

### 디코더를 아직 안 정했습니다

기획서는 pyzbar를 쓰라고 했지만, **9월 1주차에 3개를 비교해서 정합니다** → [`docs/decisions/0006-디코더-선정.md`](../decisions/0006-디코더-선정.md)

| 후보 | 비고 |
|---|---|
| **pyzbar** (ZBar) | 기획서 안. 오래된 라이브러리이고 회전·왜곡에 약하다고 알려져 있음 |
| **zxing-cpp** | ZXing의 C++ 재작성판 |
| **OpenCV `barcode`** | `cv2.barcode.BarcodeDetector`. OpenCV 4.8부터 기본 포함 |

**이걸 왜 비교하나**: 디코더만 바꿔서 성공률이 크게 오르면, "AI 기하 보정이 기여한 향상폭"이 실제보다 부풀려집니다. 심하면 AI 없이 디코더 교체만으로 목표를 달성할 수도 있습니다. 그럼 프로젝트 전제를 다시 봐야 합니다. 11월 심사에서 "왜 pyzbar만 썼나"라는 질문에 답할 근거도 필요합니다.

비용은 하루입니다. 코드는 같고 호출부만 바꿉니다. **그래서 `decoding.py`는 디코더를 갈아끼울 수 있게 만드세요.**

### 재시도 순서가 정해져 있습니다

기획서는 "사전 정의된 탐색 순서"라고만 하고 순서를 안 정했습니다. 초기값을 이렇게 정했습니다.

| 시도 | 무엇을 바꾸나 | 어떤 실패를 노리나 |
|---|---|---|
| 0차 | `INTER_CUBIC` + 기본 이진화 | — |
| 1차 재시도 | **`INTER_LANCZOS4`** | 보간이 뭉개서 얇은 바가 붙은 경우 |
| 2차 재시도 | **이진화 임계값을 낮춤** | 반사로 밝아져서 바가 사라진 경우 |
| 3차 재시도 | **제어점 변위 0.9배 + `INTER_LINEAR`** | 기하 추정이 과하게 폈거나 노이즈가 심한 경우 |

보간법을 먼저 바꾸는 이유: 보정 비용이 20ms라 가장 싸고, 얇은 바가 붙는 문제가 판독 실패의 흔한 원인입니다. 이 순서는 초기값이고 실측 후 `docs/decisions/0006`에 확정합니다.

### 재시도할 때 보정은 다시 해도 됩니다

**기하 추정은 다시 돌리지 않지만, 기하 보정은 다시 합니다.** 두 비용이 완전히 다릅니다.

| | 비용 | 재시도에서 |
|---|---|---|
| 기하 추정 (AI) | 150 ms | **다시 돌리지 않습니다.** 4번이면 예산이 무너집니다 |
| 기하 보정 (OpenCV) | 20 ms | **다시 합니다.** 3번 해도 60ms입니다 |

그래서 재시도가 실질적인 의미를 갖습니다 — 같은 `GeometryField`를 **다르게 적용해서** 다시 읽습니다.

이것이 기존 설계와 달라진 점입니다. 예전에는 복원이 250ms라 다시 돌릴 수 없었고, 재시도에서 바꿀 것이 이진화 설정뿐이었습니다.

### 오독은 실패보다 나쁩니다

디코더가 **틀린 번호를 자신 있게** 내놓는 경우가 있습니다. 물류에서는 잘못된 화물 정보가 그대로 흘러가므로, 못 읽는 것보다 나쁩니다.

그래서 평가에서 **오독률 목표는 0%**입니다. 판독 성공 판정은 "뭔가 읽혔다"가 아니라 **"정답 번호와 완전히 일치한다"**로 합니다.

### 막히면 이렇게 시작하세요

```python
import cv2
from pyzbar import pyzbar

def decode(image: RectifiedBarcode) -> DecodeResult:
    binary = cv2.adaptiveThreshold(
        image.image_gray_uint8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5,
    )
    found = pyzbar.decode(binary)
    if found:
        return DecodeResult(
            text=found[0].data.decode(),
            symbology=found[0].type,
            retry_count=0,
        )
    return DecodeResult(
        text=None, symbology=None, retry_count=0,
        failure_reason="decode_failed",
    )
```

**이 코드가 곧 9월 1주차의 베이스라인 측정 도구입니다.** 그리고 그게 그대로 11월 최종 평가 코드가 됩니다. 평가 도구를 11월에 만들기 시작하면 일정을 감당할 수 없으니, 처음부터 재사용을 생각하고 만드세요.

---

## `wemeet/sw/server.py` — FastAPI

### 엔드포인트 2개

| 메서드 | 경로 | 용도 |
|---|---|---|
| `POST` | `/api/decode` | 이미지 한 장 처리 |
| `GET` | `/api/health` | 서버 상태, 모델 버전 |

### 응답 형식

```json
{
  "ok": true,
  "text": "8801234567890",
  "symbology": "CODE128",
  "retry_count": 1,
  "failure_reason": null,
  "candidate_count": 1,
  "degraded": false,
  "stage_ms": { "detect": 95.0, "decode_first": 22.0, "estimate": 140.0, "warp": 18.0, "decode": 26.0 },
  "total_ms": 302.1,
  "images": {
    "original": "data:image/png;base64,...",
    "detected_box": [[120, 340], [520, 350], [518, 430], [118, 420]],
    "rectified": "data:image/png;base64,..."
  }
}
```

### 판독 실패에도 HTTP 200을 씁니다

**판독 실패는 서버 오류가 아닙니다.** 이 시스템의 정상적인 결과 중 하나입니다.

4xx/5xx를 쓰면 React가 `catch`에서 처리하게 되고, 그러면 **실패한 경우에 복원 이미지를 화면에 못 띄웁니다.** 그런데 이 프로젝트에서 가장 중요한 화면이 바로 그 경우입니다. 작업자는 "실패했다"만 보는 게 아니라 "여기까지 펴봤는데 안 읽혔다"를 보고 수동 대처를 판단합니다.

4xx/5xx는 진짜 오류에만 씁니다.

| 상태 | 상황 |
|---|---|
| `400` | 파일 없음, 이미지로 해석 불가 |
| `413` | 파일 크기 초과 |
| `500` | 서버 내부 오류 (모델 로드 실패 등) |

### 이미지는 base64로 JSON에 담습니다

파일 서버나 임시 URL을 만들지 않습니다.

- 요청 하나에 응답 하나로 끝나서 React 코드가 단순해집니다
- **서버에 이미지를 저장하지 않습니다** — 실제 송장에는 수취인 정보가 있으니 디스크에 남기면 안 됩니다

원본은 클 수 있으니 **긴 변 1280px로 줄여서** 보냅니다. 화면 대조용이라 원해상도가 필요 없습니다.

`detected_box`는 원본 좌표계의 네 점입니다. React가 원본 위에 그대로 그립니다. 회전된 사각형이므로 네 점입니다.

### 로그에 남기지 말아야 하는 것

**남깁니다**: 처리 시각, `stage_ms`, `retry_count`, `failure_reason`, 성공/실패 여부

**남기지 않습니다**: 입력 이미지, 크롭 이미지, **바코드 번호**

바코드 번호는 화물을 특정하는 식별자입니다. 로그 파일이 공유 폴더나 저장소에 올라가는 순간 개인정보 정책이 무의미해집니다. 디버깅에 이미지가 필요하면 개발 환경 전용 플래그로 켜고, 출력은 `downloads/` 아래로 (git 제외 대상) 보냅니다.

---

## `web/` — React 화면

### 환경

| 항목 | 선택 |
|---|---|
| 빌드 도구 | **Vite** |
| 언어 | **TypeScript** |
| 패키지 매니저 | **npm** |
| Node | **22 LTS** (`web/.nvmrc`에 고정) |
| UI 라이브러리 | **없음** — 화면이 하나라 CSS로 충분 |
| 상태 관리 | **없음** — `useState`로 충분 |

명령어 2개입니다.

```bash
cd web && npm install      # 처음 한 번
cd web && npm run dev      # 개발 서버
```

### 화면에 넣을 것 (이 4개만)

기획서가 요구한 것입니다. 그 이상은 만들지 않습니다.

1. 이미지 로드
2. 탐지 영역 표시 (원본 위에 사각형)
3. 원본과 펴진 결과 **나란히 대조**
4. 바코드 번호와 처리 시간(ms) 표시

**넣지 않는 것**: 실시간 컨베이어 모니터링, 처리 이력 통계, 사용자 관리, 설정 화면. 11월 일정이 압축돼 있어서 범위를 줄였습니다.

### API 타입을 적어두세요

`web/src/api.ts`에 응답 형식을 타입으로 적습니다. 이게 프론트엔드 쪽 계약입니다.

```typescript
export type DecodeResponse = {
  ok: boolean;
  text: string | null;
  symbology: string | null;
  retry_count: number;
  failure_reason: "invalid_input" | "not_detected" | "decode_failed" | null;
  candidate_count: number;
  degraded: boolean;
  stage_ms: Record<string, number>;
  total_ms: number;
  images: {
    original: string;
    detected_box: [number, number][] | null;
    rectified: string | null;
  };
};
```

### 응답 형식을 바꿀 때

**`server.py`와 `web/src/api.ts`를 같은 PR에서 함께 고치세요.** 한쪽만 바꾸면 화면이 조용히 깨집니다. 에러도 안 나고 그냥 빈 화면이 됩니다.

---

## 우리 파트 일정

| 시기 | 할 일 |
|---|---|
| **9월 1주차** | **디코더 3종 베이스라인 측정** (이게 최우선) |
| **9월 1~2주차** | `rectify.py` 구현 — 학습이 필요 없어 지금 바로 진짜로 만들 수 있다 |
| **9월 2주차** | 평가 지표 체계 확정 지원 |
| **9월 3~4주차** | `pipeline.py` 연결 + 조기 종료. AI는 가짜 구현으로 채워둔다 |
| **10월 2~3주차** | 가짜를 진짜 모델로 교체 (우리는 `pipeline.py` 를 안 건드린다) |
| **10월 3주차** | 재시도 루프 + 반사 대응 구현 |
| **10월 4주차** | E2E 시간 측정, 500ms 검증 |
| **11월 1주차** | React 화면 |
| **11월 2주차** | 최종 평가 실행 |

8월 항목이 원래 "가짜 파이프라인 완성"이었는데 범위에서 뺐습니다. 8월에는 계약·테스트·CI만 세웠고, 파이프라인 연결은 9월 3주차로 옮겼습니다.

### 9월 1주차가 왜 최우선인가

지금 우리는 **현재 성공률이 몇 %인지 모릅니다.** 기준선 없이 "90% 달성"을 주장할 수 없습니다.

그리고 이 작업은 AI 모델을 기다릴 필요가 없습니다. 디코더와 다운로드한 데이터만 있으면 됩니다. 9월에 AI파트가 학습하는 동안 우리가 놀지 않는 방법이기도 합니다.

---

## 자주 헷갈리는 것

**Q. AI 모델이 없는데 뭘 하죠?**
가짜 구현으로 채워서 파이프라인을 완성하세요. 그게 8월 목표입니다. AI 모델은 나중에 끼워 넣습니다.

**Q. 탐지가 2개 나왔는데 어떻게 하죠?**
`confidence` 1등만 처리하고, 총 개수를 `candidate_count`에 기록하세요.

**Q. 기하 추정이 예외를 던졌습니다.**
잡아서 보정 없이 진행하고 `degraded=True`로 표시하세요. 파이프라인이 멈추면 안 됩니다.

**Q. 기하 추정 `confidence`가 낮게 나왔습니다.**
0.3 미만이면 보정을 건너뛰고 실패로 처리하세요. 신뢰할 수 없는 필드로 펴면 오히려 더 안 읽힙니다. 임계값은 초기값이라 AI파트가 실측 후 조정합니다.

**Q. `schemas.py`를 고쳐야 할 것 같은데요?**
AI파트 승인이 있어야 머지됩니다. Issue를 만들고 `decision` 라벨을 붙이세요.

**Q. 500ms를 넘었습니다.**
중단하지 말고 끝까지 진행하고 시간만 기록하세요. `stage_ms`를 보면 어느 단계가 문제인지 나옵니다.
