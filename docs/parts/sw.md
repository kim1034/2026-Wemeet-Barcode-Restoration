# SW파트가 할 일

**담당: 김종연, 이도훈**
**내 폴더: `wemeet/sw/`, `web/`**

---

## 한 문단 요약

AI파트가 만든 함수들을 **불러다 순서대로 이어 붙이고**(파이프라인), 복원된 이미지를 **바코드 번호로 바꾸고**(디코딩), 결과를 **화면에 띄웁니다**(서버 + React).

우리는 **지휘자**입니다. AI파트가 악기를 만들고 우리가 연주 순서를 정합니다. 그래서 이 프로젝트가 실제로 돌아가느냐는 우리 손에 달려 있습니다.

---

## 담당 파일

| 파일 | 무엇 |
|---|---|
| `wemeet/sw/pipeline.py` | **3단계를 이어 붙이는 곳** (우리 파트의 핵심) |
| `wemeet/sw/decoding.py` | 3단계 — 번호로 바꾸기 + 재시도 루프 |
| `wemeet/sw/server.py` | FastAPI 서버 |
| `web/` | React 화면 |

---

## 우리가 쓸 수 있는 것

우리는 AI파트를 부를 수 있습니다.

```python
from wemeet.ai.detection import detect            # ← 됩니다
from wemeet.ai.restoration import restore         # ← 됩니다
from wemeet.schemas import PipelineResult         # ← 됩니다
from wemeet.data.synthesis import warp            # ← 안 됩니다
```

`wemeet.data`는 못 씁니다. 데이터 생성 스크립트가 서버 코드에 끌려 들어오면 안 되기 때문입니다.

---

## `wemeet/sw/pipeline.py` — 3단계 이어 붙이기

### 이 파일이 가장 중요합니다

여기가 프로젝트의 심장입니다. 그리고 **9월·10월에 이 파일을 잘 만들어두면 "통합"이라는 힘든 작업이 사라집니다.**

### 구현할 함수

```python
def run(image_bgr: np.ndarray) -> PipelineResult:
```

기본 흐름은 이렇습니다.

```python
from wemeet.ai.detection import detect
from wemeet.ai.restoration import restore
from wemeet.sw.decoding import decode
from wemeet.schemas import PipelineResult

def run(image_bgr):
    boxes = detect(image_bgr)                 # 1단계
    if not boxes:
        return ...                            # 못 찾았을 때 (아래 표 참고)

    best = boxes[0]                           # confidence 최상위 하나만
    restored = restore(best)                  # 2단계
    result = decode(restored)                 # 3단계
    return PipelineResult(
        original_bgr_uint8=image_bgr,
        detected=best,
        restored=restored,
        decode=result,
    )
```

### `PipelineResult`를 반환하는 이유

`DecodeResult`(번호만 들어 있는 것)를 반환하면 **복원된 이미지가 화면까지 도달하지 못합니다.**

그런데 이 프로젝트가 상용 솔루션(Scandit 등)과 차별화되는 지점이 바로 그겁니다. 상용 제품은 내부가 블랙박스라서 실패했을 때 작업자가 아무것도 볼 수 없습니다. 우리는 **"AI가 여기까지 복원했는데 안 읽혔다"를 보여줘서** 작업자가 즉시 판단할 수 있게 합니다.

그래서 원본·탐지결과·복원이미지·번호를 다 담아서 반환합니다.

### 실패했을 때 무엇을 반환하나

**절대 예외를 던지지 않습니다.** 어떤 상황에서도 `PipelineResult`를 반환합니다.

현장 컨베이어에서 예외가 올라오면 서비스가 멈춥니다. 그게 곧 노리드 존 정체이고, 우리가 없애려는 바로 그 문제입니다. **실패는 예외가 아니라 값으로 표현합니다.**

| 상황 | 어떻게 하나 |
|---|---|
| 이미지 로드 실패 | `failure_reason="invalid_input"`, 나머지 `None` |
| **탐지 0개** | 복원·디코딩 생략. `failure_reason="not_detected"` |
| **탐지 2개 이상** | `confidence` 1등만 처리. `candidate_count`에 총 개수 기록 |
| 복원이 예외를 던짐 | 예외를 잡고 **원본 크롭으로 대체** 진행. `degraded=True` |
| 디코딩 1차 실패 | 재시도 루프 최대 3회 |
| 재시도 후에도 실패 | `failure_reason="decode_failed"` |
| **500ms 초과** | **중단하지 않고 끝까지 진행.** 시간만 기록 |

두 개만 부연합니다.

**탐지 2개 이상은 흔합니다.** 박스에 송장 바코드와 상품 바코드가 같이 붙어 있습니다. 1등만 처리하고 총 개수를 기록해두면, 나중에 "여러 개 처리가 필요한가"를 데이터로 판단할 수 있습니다.

**500ms를 넘어도 중단하지 않습니다.** 중단하면 읽을 수 있었던 화물을 스스로 버리는 셈이고, 무엇보다 초과 원인을 알 수 없게 됩니다. 시간만 기록하고 예산 준수는 평가 단계에서 통계로 판정합니다.

### 시간을 단계별로 재세요

```python
result.decode.stage_ms = {
    "detect": 42.1,
    "restore": 210.3,
    "decode": 18.7,
    "retry": 31.0,
}
```

**예산이 정해져 있습니다.**

| 단계 | 예산 |
|---|---|
| 탐지 | 100 ms |
| 복원 | 250 ms |
| 디코딩 1차 | 50 ms |
| 재시도 (최대 3회) | 100 ms |
| **합계** | **500 ms** |

단계별로 재두면 나중에 누가 초과했는지 바로 나옵니다. 안 재두면 "느린데 어디가 느린지 모르겠다"가 됩니다.

### 8월에 할 일 — 가짜로라도 끝까지 이어놓기

**AI 모델을 기다리지 마세요.** 8월에 각 단계를 가짜 구현으로 채워서 파이프라인이 실제로 돌아가게 만듭니다.

```python
def run(image_bgr):
    boxes    = detect(image_bgr)      # 가짜: 중앙 40% 자르기
    restored = restore(boxes[0])      # 가짜: CLAHE만
    return decode(restored)           # 이건 처음부터 진짜
```

이렇게 해두면 10월에 AI파트가 하는 일은 `detection.py` 안의 가짜를 진짜로 바꿔 넣는 것뿐입니다. **`pipeline.py`는 건드리지 않습니다.** "통합"이라는 별도 작업이 없어지고, 이게 11월 일정을 지킬 수 있는 이유입니다.

그리고 더 중요한 게 있습니다. BGR/RGB 같은 문제가 **8월에 드러납니다.** 가짜끼리라도 데이터가 실제로 흘러가니까요. 10월에 처음 붙여보는 것과 8월부터 계속 붙어 있는 것의 차이가 프로젝트 성패를 가릅니다.

---

## `wemeet/sw/decoding.py` — 3단계 디코딩 + 재시도

### 구현할 함수

```python
def decode(image: RestoredBarcode) -> DecodeResult:
```

흐름은 이렇습니다.

```
복원된 흑백 이미지
      ↓
CLAHE (명암 대비 개선)
      ↓
적응형 이진화 (흑백 두 값으로)
      ↓
디코더 호출
      ↓
실패? → 설정 바꿔서 최대 3회 재시도
```

### 디코더를 아직 안 정했습니다

기획서는 pyzbar를 쓰라고 했지만, **9월 1주차에 3개를 비교해서 정합니다** → [`docs/decisions/0006-디코더-선정.md`](../decisions/0006-디코더-선정.md)

| 후보 | 비고 |
|---|---|
| **pyzbar** (ZBar) | 기획서 안. 오래된 라이브러리이고 회전·왜곡에 약하다고 알려져 있음 |
| **zxing-cpp** | ZXing의 C++ 재작성판 |
| **OpenCV `barcode`** | `cv2.barcode.BarcodeDetector`. OpenCV 4.8부터 기본 포함 |

**이걸 왜 비교하나**: 디코더만 바꿔서 성공률이 크게 오르면, "AI 복원이 기여한 향상폭"이 실제보다 부풀려집니다. 심하면 AI 없이 디코더 교체만으로 목표를 달성할 수도 있습니다. 그럼 프로젝트 전제를 다시 봐야 합니다. 11월 심사에서 "왜 pyzbar만 썼나"라는 질문에 답할 근거도 필요합니다.

비용은 하루입니다. 코드는 같고 호출부만 바꿉니다. **그래서 `decoding.py`는 디코더를 갈아끼울 수 있게 만드세요.**

### 재시도 순서가 정해져 있습니다

기획서는 "사전 정의된 탐색 순서"라고만 하고 순서를 안 정했습니다. 초기값을 이렇게 정했습니다.

| 시도 | 무엇을 바꾸나 | 어떤 실패를 노리나 |
|---|---|---|
| 0차 | 기본 설정 | — |
| 1차 재시도 | **임계값을 낮춤** | 반사로 밝아져서 바가 사라진 경우 |
| 2차 재시도 | **임계값을 높임** | 그림자로 어두워져서 흰 여백이 사라진 경우 |
| 3차 재시도 | **커널 크기 확대 + 대비 강화** | 주름으로 바가 끊어진 경우 |

임계값을 먼저 **낮추는** 이유: 이 프로젝트의 1순위 훼손이 **비닐 반사**이고, 반사는 밝아지는 방향의 손실입니다. 흔한 원인부터 시도하면 평균 재시도 횟수가 줄고 지연 예산에 유리합니다.

### 재시도할 때 복원을 다시 돌리지 마세요

**이진화와 디코딩만 다시 합니다.** 복원까지 다시 돌리면 250ms가 4번 들어가서 500ms 예산이 즉시 무너집니다.

기획서도 재시도 루프를 "학습된 모델의 출력 이후 단계에서 작동하는 규칙 기반 보완 장치"로 규정합니다.

### 오독은 실패보다 나쁩니다

디코더가 **틀린 번호를 자신 있게** 내놓는 경우가 있습니다. 물류에서는 잘못된 화물 정보가 그대로 흘러가므로, 못 읽는 것보다 나쁩니다.

그래서 평가에서 **오독률 목표는 0%**입니다. 판독 성공 판정은 "뭔가 읽혔다"가 아니라 **"정답 번호와 완전히 일치한다"**로 합니다.

### 막히면 이렇게 시작하세요

```python
import cv2
from pyzbar import pyzbar

def decode(image: RestoredBarcode) -> DecodeResult:
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
  "stage_ms": { "detect": 42.1, "restore": 210.3, "decode": 18.7, "retry": 31.0 },
  "total_ms": 302.1,
  "images": {
    "original": "data:image/png;base64,...",
    "detected_box": [[120, 340], [520, 350], [518, 430], [118, 420]],
    "restored": "data:image/png;base64,..."
  }
}
```

### 판독 실패에도 HTTP 200을 씁니다

**판독 실패는 서버 오류가 아닙니다.** 이 시스템의 정상적인 결과 중 하나입니다.

4xx/5xx를 쓰면 React가 `catch`에서 처리하게 되고, 그러면 **실패한 경우에 복원 이미지를 화면에 못 띄웁니다.** 그런데 이 프로젝트에서 가장 중요한 화면이 바로 그 경우입니다. 작업자는 "실패했다"만 보는 게 아니라 "여기까지 복원했는데 안 읽혔다"를 보고 수동 대처를 판단합니다.

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
3. 원본과 복원 결과 **나란히 대조**
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
    restored: string | null;
  };
};
```

### 응답 형식을 바꿀 때

**`server.py`와 `web/src/api.ts`를 같은 PR에서 함께 고치세요.** 한쪽만 바꾸면 화면이 조용히 깨집니다. 에러도 안 나고 그냥 빈 화면이 됩니다.

---

## 우리 파트 일정

| 시기 | 할 일 |
|---|---|
| **8월** | 가짜 파이프라인 완성 — 이미지 넣으면 끝까지 흘러가게 |
| **9월 1주차** | **디코더 3종 베이스라인 측정** (이게 최우선) |
| **9월 2주차** | 평가 지표 체계 확정 지원 |
| **10월 2~3주차** | 가짜를 진짜 모델로 교체 |
| **10월 3주차** | 재시도 루프 구현 |
| **10월 4주차** | E2E 시간 측정, 500ms 검증 |
| **11월 1주차** | React 화면 |
| **11월 2주차** | 최종 평가 실행 |

### 9월 1주차가 왜 최우선인가

지금 우리는 **현재 성공률이 몇 %인지 모릅니다.** 기준선 없이 "90% 달성"을 주장할 수 없습니다.

그리고 이 작업은 AI 모델을 기다릴 필요가 없습니다. 디코더와 다운로드한 데이터만 있으면 됩니다. 9월에 AI파트가 학습하는 동안 우리가 놀지 않는 방법이기도 합니다.

---

## 자주 헷갈리는 것

**Q. AI 모델이 없는데 뭘 하죠?**
가짜 구현으로 채워서 파이프라인을 완성하세요. 그게 8월 목표입니다. AI 모델은 나중에 끼워 넣습니다.

**Q. 탐지가 2개 나왔는데 어떻게 하죠?**
`confidence` 1등만 처리하고, 총 개수를 `candidate_count`에 기록하세요.

**Q. 복원이 예외를 던졌습니다.**
잡아서 원본 크롭으로 대체하고 `degraded=True`로 표시하세요. 파이프라인이 멈추면 안 됩니다.

**Q. `schemas.py`를 고쳐야 할 것 같은데요?**
AI파트 승인이 있어야 머지됩니다. Issue를 만들고 `decision` 라벨을 붙이세요.

**Q. 500ms를 넘었습니다.**
중단하지 말고 끝까지 진행하고 시간만 기록하세요. `stage_ms`를 보면 어느 단계가 문제인지 나옵니다.
