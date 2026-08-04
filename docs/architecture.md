# 구조와 규칙

매일 볼 내용만 모았습니다. 전체 근거는 [설계 문서](superpowers/specs/2026-07-30-main-branch-structure-design.md)에 있습니다.

---

## 폴더 구조

```
wemeet/
├── __init__.py
├── schemas.py             파트 간 계약 (아무것도 import 안 함)
│
├── ai/                    ← AI파트
│   ├── detection.py           Stage 1  탐지·크롭            (2단계)
│   ├── restoration.py         Stage 2  평탄화·반사 제거      (2단계)
│   ├── train_detection.py     탐지 모델 학습               (2단계)
│   ├── train_restoration.py   복원 모델 학습               (2단계)
│   └── configs/               학습 설정 YAML               (2단계)
│
├── sw/                    ← SW파트
│   ├── decoding.py            Stage 3  디코더 + 재시도 루프  (2단계)
│   ├── pipeline.py            3단계 연결                   (2단계)
│   └── server.py              FastAPI                     (2단계)
│
└── data/                  ← 데이터파트
    ├── download.py            HF Hub에서 받아오기          (2단계)
    ├── synthesis.py           역방향 합성                  (2단계)
    └── ground_truth.py        실촬영 정답 번호 확보         (2단계)

web/                       React 화면 (SW파트)              (2단계)
tests/
├── conftest.py            바코드 이미지 생성 픽스처
└── test_schemas.py        계약 검증
downloads/                 받아온 이미지·모델 (git 제외)
```

파일 이름이 파이프라인 단계와 그대로 맞습니다. `detection.py`는 1단계, `restoration.py`는 2단계, `decoding.py`는 3단계입니다.

`(2단계)` 표시는 아직 만들지 않은 파일입니다. 지금 있는 것은 `schemas.py`와 `tests/` 뿐입니다.

---

## 의존 방향

### 먼저: 이 화살표는 "부르는 방향"입니다

**데이터가 흐르는 방향과 반대입니다.** 이걸 먼저 잡아두지 않으면 아래 그림이 거꾸로 보입니다.

| | 방향 |
|---|---|
| **데이터 흐름** (시간 순서) | 사진 → 탐지(AI) → 복원(AI) → 디코딩(SW) → 번호 |
| **의존 방향** (누가 누구를 부르는가) | `pipeline`(SW) → `detect`·`restore`(AI) |

탐지가 먼저 실행되지만, **그 실행을 시작시키는 것은 `sw/pipeline.py`** 입니다. 부르는 쪽이 부르는 대상을 import해야 하므로 `sw → ai` 입니다.

지휘자 비유가 정확히 이것입니다. 지휘자가 연주자를 **부르고**, 소리는 연주자에게서 관객으로 **흐릅니다.** 부르는 방향과 소리가 가는 방향은 반대입니다.

**"위"가 "먼저 실행됨"을 뜻하지 않습니다.** 위는 부르는 쪽, 아래는 불리는 쪽입니다.

### 4개 층

세 파트와 계약 모듈이 4개 층을 이룹니다. 화살표(= 부른다)는 위에서 아래로만 흐릅니다.

```
   data/        데이터파트 — 아무도 부르지 않는다
     │  ╲
     │   ╲
    sw/   │     SW파트 — ai 를 불러다 조립한다 (지휘자)
     │    │
    ai/   │     AI파트 — sw 도 data 도 부르지 않는다
     │    │
     ↓    ↓
   schemas.py   wemeet 안의 어떤 모듈도 부르지 않는다 (잎)
```

허용되는 import는 이 다섯 개뿐입니다.

| 주체 | 부를 수 있는 것 |
|---|---|
| `ai` | `schemas` |
| `sw` | `schemas`, `ai` |
| `data` | `schemas`, **`sw.decoding`** |
| `schemas` | (없음) |

반대 방향 화살표가 하나만 생겨도 순환이 완성되어 `ImportError: cannot import name ... from partially initialized module`이 발생합니다. 이 오류는 import 진입점에 따라 나타났다 사라지므로 원인 추적이 어렵고, 하필 통합 시점에 터집니다.

`data → sw.decoding`만 예외로 허용하는 이유는 실촬영 테스트셋의 정답 번호를 "평탄 상태로 촬영 → 디코더로 정상 판독 → 정답 기록" 순서로 확보하기 때문입니다. 디코더가 `sw/decoding.py`에 있으니 데이터파트가 그것을 불러야 합니다. 다만 **`sw.pipeline`과 `sw.server`는 부를 수 없습니다.**

### 팀 규칙 3줄

> **AI파트는 자기 파일에 `wemeet.sw` 나 `wemeet.data` 라는 글자를 쓰지 않는다. 공용이 필요하면 `wemeet.schemas` 를 쓴다.**
>
> **데이터파트는 `wemeet.sw.decoding` 까지만 쓸 수 있다. `pipeline` 과 `server` 는 쓰지 않는다.**
>
> **함수 안에서 import 하지 않는다.**

세 번째 규칙의 근거: 함수 내부 import는 순환을 런타임으로 미루는 응급처치일 뿐 순환 자체를 제거하지 않습니다. 다른 진입점에서 재발하며, 6개월 뒤에는 왜 그 자리에 있는지 알 수 없게 됩니다.

### 강제 방법

`import-linter`가 CI에서 막습니다. 로컬에서도 확인할 수 있습니다.

```bash
uv run lint-imports
```

계약은 `pyproject.toml`의 `[tool.importlinter]`에 있습니다. 어겼을 때 어떻게 나오는지는 이렇습니다.

```
wemeet.ai is not allowed to import wemeet.sw:
-   wemeet.ai -> wemeet.sw (l.1)
```

---

## 각 파트가 구현할 함수

계약으로 고정된 형태입니다. 여기서 벗어나면 통합이 깨집니다.

```python
# wemeet/ai/detection.py        @AI파트
def detect(image_bgr: np.ndarray) -> list[DetectedBarcode]: ...

# wemeet/ai/restoration.py      @AI파트
def restore(target: DetectedBarcode) -> RestoredBarcode: ...
#   주의: 리스트가 아니라 하나를 받는다. 여러 개 처리는 pipeline이 결정한다.

# wemeet/sw/decoding.py         @SW파트
def decode(image: RestoredBarcode) -> DecodeResult: ...

# wemeet/sw/pipeline.py         @SW파트
def run(image_bgr: np.ndarray) -> PipelineResult: ...
```

`restore()`가 리스트가 아니라 단일 객체를 받는 것이 중요합니다. "탐지 결과가 여러 개일 때 무엇을 복원할지"는 AI파트가 아니라 파이프라인이 정할 문제입니다. 이 경계를 흐리면 두 파트가 같은 결정을 서로 다르게 구현합니다.

**파이프라인은 예외를 던지지 않습니다.** 어떤 상황에서도 `PipelineResult`를 반환합니다. 현장 컨베이어에서 예외가 올라오면 서비스가 멈추고, 그것이 곧 노리드 존 정체입니다. 실패는 예외가 아니라 **값**으로 표현합니다 — 실패한 `DecodeResult`에는 항상 `failure_reason`이 들어갑니다.

---

## 코드가 흐르는 순서

사진 한 장이 들어와서 번호가 나오기까지, **어느 파일의 어느 줄이 언제 실행되는지**를 따라갑니다.

> 아래 코드는 **아직 만들지 않았습니다** (2단계 산출물). 계약(`schemas.py`)에 따라 이런 형태가 될 것이라는 예시입니다. 실제로 구현할 때 이름과 반환형은 이 계약을 지켜야 합니다.

### 호출 스택

```
server.py                POST /api/decode 를 받는다          [SW파트]
   └─ pipeline.run()     전체를 지휘한다                      [SW파트]
        ├─ detect()      ai/detection.py 가 실행된다          [AI파트]
        ├─ restore()     ai/restoration.py 가 실행된다        [AI파트]
        └─ decode()      sw/decoding.py 가 실행된다           [SW파트]
```

들여쓰기가 곧 의존 방향입니다. **안쪽이 불리는 쪽**입니다. AI 코드는 `pipeline`이 부를 때 실행되고, 스스로 아무것도 시작하지 않습니다.

### 한 줄씩

```python
# wemeet/sw/pipeline.py                                   ← SW파트가 소유한다
import time

from wemeet.ai.detection import detect          # ① SW가 AI를 부르므로 여기서 import
from wemeet.ai.restoration import restore       #    AI 쪽에는 이런 줄이 없어야 한다
from wemeet.schemas import DecodeResult, PipelineResult, RestoredBarcode
from wemeet.sw.decoding import decode


def run(image_bgr):
    stage_ms = {}

    # ② 탐지 — 이 줄에서 AI파트 코드가 실행된다
    t = time.perf_counter()
    found = detect(image_bgr)                   # -> list[DetectedBarcode]
    stage_ms["detect"] = (time.perf_counter() - t) * 1000

    # ③ 판단은 SW가 한다. AI는 "찾은 것"만 돌려줬을 뿐이다
    if not found:
        return _실패(image_bgr, "not_detected", stage_ms)

    best = max(found, key=lambda b: b.confidence)   # 2개 이상이면 1등만 처리한다

    # ④ 복원 — 리스트가 아니라 하나만 넘긴다.
    #    "무엇을 복원할지"는 위 ③에서 파이프라인이 이미 정했다
    t = time.perf_counter()
    try:
        restored = restore(best)                # -> RestoredBarcode
        degraded = False
    except Exception:
        # 복원 모델이 터져도 멈추지 않는다. 원본 크롭을 그대로 넘긴다
        restored = RestoredBarcode(
            image_gray_uint8=_흑백으로(best.crop_bgr_uint8),
            source=best,
        )
        degraded = True                         # 복원 없이 갔다는 표시
    stage_ms["restore"] = (time.perf_counter() - t) * 1000

    # ⑤ 디코딩 + 재시도 루프 (최대 3회)
    t = time.perf_counter()
    result = decode(restored)                   # -> DecodeResult
    stage_ms["decode"] = (time.perf_counter() - t) * 1000

    # ⑥ 결과에 파이프라인이 아는 정보를 채운다
    result.stage_ms = stage_ms
    result.candidate_count = len(found)         # 몇 개 찾았는지는 AI가 모른다
    result.degraded = degraded

    # ⑦ 중간 산출물까지 전부 담아서 돌려준다
    return PipelineResult(
        original_bgr_uint8=image_bgr,
        detected=best,
        restored=restored,
        decode=result,
    )
```

### 각 번호가 중요한 이유

| | 무엇을 보여주나 |
|---|---|
| ① | **import가 SW 쪽에만 있습니다.** AI 파일에 `from wemeet.sw...` 가 생기면 서로 부르는 순환이 되고 CI가 막습니다 |
| ② | `detect()` 는 먼저 실행되지만 **부른 쪽은 `run()`** 입니다. 이것이 "데이터는 AI → SW, 호출은 SW → AI"의 실제 모습입니다 |
| ③ | "0개면?", "2개 이상이면?" 같은 판단이 **파이프라인에 모여 있습니다.** AI 안으로 새면 두 파트가 같은 결정을 서로 다르게 구현합니다 |
| ④ | `restore()` 가 하나만 받는 이유가 여기서 드러납니다. 고르는 일은 ③에서 끝났습니다 |
| ⑤ | 재시도는 SW의 몫입니다. AI는 "한 번 복원"만 알고 재시도를 모릅니다 |
| ⑥ | `candidate_count` 를 파이프라인이 채웁니다 — 몇 개 찾았는지는 **`decode()` 가 알 수 없는 정보**입니다 |
| ⑦ | 중간 이미지를 버리지 않습니다. 이게 없으면 화면에 "복원된 모습"을 못 보여줍니다 |

### 예외를 던지지 않으면 실패는 어떻게 전달되나

`try/except` 로 감싼 뒤 **값으로 바꿔서** 돌려줍니다.

| 상황 | `detected` | `restored` | `decode.text` | `failure_reason` | 비고 |
|---|---|---|---|---|---|
| 정상 판독 | 있음 | 있음 | `"123456"` | `None` | |
| 이미지 형식 오류 | `None` | `None` | `None` | `invalid_input` | 즉시 종료 |
| 탐지 0개 | `None` | `None` | `None` | `not_detected` | 복원·디코딩 생략 |
| 탐지 2개 이상 | 1등만 | 있음 | 보통 성공 | `None` | `candidate_count`에 총 개수 |
| 복원 실패 | 있음 | **원본 크롭** | 될 수도 | — | `degraded=True` |
| 재시도 후 실패 | 있음 | 있음 | `None` | `decode_failed` | `retry_count=3` |
| 500ms 초과 | 있음 | 있음 | 보통 성공 | `None` | **중단하지 않는다.** 시간만 기록 |

호출한 쪽(`server.py`)은 예외를 잡지 않고 `result.ok` 만 봅니다.

```python
# wemeet/sw/server.py
result = run(image_bgr)
return {
    "ok": result.ok,                       # text 가 None 이 아니면 True
    "text": result.decode.text,
    "failure_reason": result.decode.failure_reason,
    "total_ms": result.decode.total_ms,
    ...
}
```

판독 실패도 **HTTP 200** 으로 보냅니다. 실패는 서버 오류가 아니라 정상적인 처리 결과이기 때문입니다.

### AI파트가 자기 코드를 단독으로 돌릴 수 있습니다

의존 방향을 이렇게 잡은 부수 효과입니다. `detect()` 는 파이프라인 없이도 바로 부를 수 있습니다.

```python
# 노트북이나 학습 스크립트에서
from wemeet.ai.detection import detect

boxes = detect(image)      # 서버도 디코더도 필요 없다
```

만약 AI가 SW를 알고 있었다면, 탐지 실험을 하려고 FastAPI와 디코더까지 끌어와야 합니다. 그러면 실험이 무거워지고, 서버 코드를 고칠 때 학습 스크립트가 깨집니다.

---

## 시간 예산

기획서의 0.5초를 단계별로 배분한 것입니다. `stage_ms`로 측정하고 목표 초과 시 원인을 특정합니다.

| 단계 | 예산 |
|---|---|
| 탐지 (detect) | 100 ms |
| 복원 (restore) | 250 ms |
| 디코딩 1차 (decode) | 50 ms |
| 재시도 여유 (retry, 최대 3회) | 100 ms |
| 합계 | **500 ms** |

**0.5초는 재시도를 포함한 전체 경로 기준입니다.** 기획서 2.1과 4.3.3의 표현이 충돌하는 부분을 이렇게 확정했습니다.

예산을 초과해도 **중단하지 않습니다.** 중단하면 판독 가능한 화물을 스스로 버리는 것이 되고, 무엇보다 초과 원인을 알 수 없게 됩니다. 시간을 기록만 하고 준수 여부는 평가 단계에서 통계로 판정합니다.

---

## 아직 정하지 않은 것

**추론 대상 하드웨어** — 0.5초 목표의 기준 장비가 기획서에 없습니다. 이것이 정해지지 않으면 "500ms 안에 끝났다"는 말의 의미가 확정되지 않습니다. 멘토 자문이 필요합니다.
담당: 팀장 / 기한: 통합 단계 전 (설계 문서 §20)
