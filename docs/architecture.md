# 구조와 규칙

매일 볼 내용만 모았습니다. 전체 근거는 [설계 문서](superpowers/specs/2026-07-30-main-branch-structure-design.md)와 [기하 파이프라인 설계](superpowers/specs/2026-08-04-geometry-pipeline-design.md)에 있습니다.

---

## 파이프라인 4단계

```
사진 → [1 검출·AI] → 1차 디코딩 시도 → 읽히면 종료
                          │ 안 읽히면
                          ▼
        [2 기하 추정·AI] → [3 기하 보정·SW] → [4 디코딩·SW]
```

| 단계 | 담당 | 하는 일 | 반환형 |
|---|---|---|---|
| 1 검출 | AI | 바코드 영역을 찾아 기울기 보정 후 크롭 | `DetectedBarcode` |
| — 1차 디코딩 | **SW (파이프라인)** | 보정 없이 먼저 읽어본다 | `DecodeResult` |
| 2 기하 추정 | AI | **얼마나 휘었는지만** 예측 | `GeometryField` |
| 3 기하 보정 | SW | OpenCV로 기존 픽셀을 이동 | `RectifiedBarcode` |
| 4 디코딩 | SW | 반사 대응 후 판독, 실패 시 최대 3회 재시도 | `DecodeResult` |

**AI는 픽셀을 만들지 않습니다.** 얼마나 휘었는지만 예측하고, 실제 변형은 OpenCV가 합니다. `GeometryField`에 이미지 필드가 없어서 **구조적으로 불가능**합니다.

이유는 생성 모델이 만든 가짜 바코드가 **체크섬을 우연히 통과할 수 있기 때문**입니다(Code128 약 1/103). 틀린 번호가 조용히 통과하는 것은 못 읽는 것보다 나쁩니다.

---

## 폴더 구조

```
wemeet/
├── __init__.py
├── schemas.py             파트 간 계약 (아무것도 import 안 함)
│
├── ai/                    ← AI파트
│   ├── detection.py           Stage 1  검출·기울기 보정        (2단계)
│   ├── geometry.py            Stage 2  기하 추정              (2단계)
│   ├── train_detection.py     검출 모델 학습                 (2단계)
│   ├── train_geometry.py      기하 추정 모델 학습             (2단계)
│   └── configs/               학습 설정 YAML                 (2단계)
│
├── sw/                    ← SW파트
│   ├── rectify.py             Stage 3  기하 보정 (OpenCV)     (2단계)
│   ├── decoding.py            Stage 4  디코더 + 반사 대응     (2단계)
│   ├── pipeline.py            4단계 연결 + 재시도 루프        (2단계)
│   └── server.py              FastAPI                       (2단계)
│
└── data/                  ← 데이터파트
    ├── download.py            HF Hub에서 받아오기            (2단계)
    ├── synthesis.py           왜곡 + 정답 파라미터 합성       (2단계)
    └── ground_truth.py        실촬영 정답 번호 확보          (2단계)

web/                       React 화면 (SW파트)               (2단계)
tests/
├── conftest.py            바코드 이미지 생성 픽스처
└── test_schemas.py        계약 검증 22개
downloads/                 받아온 이미지·모델 (git 제외)
```

`(2단계)` 표시는 아직 만들지 않은 파일입니다. 지금 있는 것은 `schemas.py`와 `tests/`뿐입니다.

파일 이름이 파이프라인 단계와 맞습니다. `detection.py`는 1단계, `geometry.py`는 2단계, `rectify.py`는 3단계, `decoding.py`는 4단계입니다.

---

## 의존 방향

### 먼저: 이 화살표는 "부르는 방향"입니다

**데이터가 흐르는 방향과 반대입니다.** 이걸 먼저 잡아두지 않으면 아래 그림이 거꾸로 보입니다.

| | 방향 |
|---|---|
| **데이터 흐름** (시간 순서) | 사진 → 검출(AI) → 기하 추정(AI) → 보정(SW) → 디코딩(SW) |
| **의존 방향** (누가 누구를 부르는가) | `pipeline`(SW) → `detect`·`estimate_geometry`(AI) |

검출이 먼저 실행되지만, **그 실행을 시작시키는 것은 `sw/pipeline.py`** 입니다. 부르는 쪽이 부르는 대상을 import해야 하므로 `sw → ai` 입니다.

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

`data → sw.decoding`만 예외로 허용하는 이유는 실촬영 테스트셋의 정답 번호를 "평탄 상태로 촬영 → 디코더로 정상 판독 → 정답 기록" 순서로 확보하기 때문입니다. 디코더가 `sw/decoding.py`에 있으니 데이터파트가 그것을 불러야 합니다. 다만 **`sw.pipeline`·`sw.server`·`sw.rectify`는 부를 수 없습니다.**

### 팀 규칙 3줄

> **AI파트는 자기 파일에 `wemeet.sw` 나 `wemeet.data` 라는 글자를 쓰지 않는다. 공용이 필요하면 `wemeet.schemas` 를 쓴다.**
>
> **데이터파트는 `wemeet.sw.decoding` 까지만 쓸 수 있다. `pipeline` · `server` · `rectify` 는 쓰지 않는다.**
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
# wemeet/ai/detection.py            @AI파트
def detect(image_bgr: np.ndarray) -> list[DetectedBarcode]: ...

# wemeet/ai/geometry.py             @AI파트
def estimate_geometry(target: DetectedBarcode) -> GeometryField: ...
#   이미지를 반환하지 않는다. 반환형에 이미지 필드가 없다.
#   리스트가 아니라 하나를 받는다 — 무엇을 보정할지는 pipeline 이 정한다.

# wemeet/sw/rectify.py              @SW파트
def apply_field(
    target: DetectedBarcode,
    field: GeometryField,
    interpolation: int = cv2.INTER_CUBIC,
) -> RectifiedBarcode: ...

# wemeet/sw/decoding.py             @SW파트
def decode(image: RectifiedBarcode) -> DecodeResult: ...

# wemeet/sw/pipeline.py             @SW파트
def run(image_bgr: np.ndarray) -> PipelineResult: ...
```

`estimate_geometry()`가 리스트가 아니라 단일 객체를 받는 것이 중요합니다. "검출 결과가 여러 개일 때 무엇을 보정할지"는 AI파트가 아니라 파이프라인이 정할 문제입니다. 이 경계를 흐리면 두 파트가 같은 결정을 서로 다르게 구현합니다.

**파이프라인은 예외를 던지지 않습니다.** 어떤 상황에서도 `PipelineResult`를 반환합니다. 현장 컨베이어에서 예외가 올라오면 서비스가 멈추고, 그것이 곧 노리드 존 정체입니다. 실패는 예외가 아니라 **값**으로 표현합니다 — 실패한 `DecodeResult`에는 항상 `failure_reason`이 들어갑니다.

---

## 코드가 흐르는 순서

사진 한 장이 들어와서 번호가 나오기까지, **어느 파일의 어느 줄이 언제 실행되는지**를 따라갑니다.

> 아래 코드는 **아직 만들지 않았습니다** (2단계 산출물). 계약(`schemas.py`)에 따라 이런 형태가 될 것이라는 예시입니다.

### 호출 스택

```
server.py                     POST /api/decode 를 받는다        [SW파트]
   └─ pipeline.run()          전체를 지휘한다                   [SW파트]
        ├─ detect()           ai/detection.py 가 실행된다       [AI파트]
        ├─ decode()           1차 시도 — 보정 없이 먼저 읽어본다  [SW파트]
        ├─ estimate_geometry()  ai/geometry.py 가 실행된다      [AI파트]
        ├─ apply_field()      sw/rectify.py 가 실행된다         [SW파트]
        └─ decode()           최종 판독 + 재시도                [SW파트]
```

들여쓰기가 곧 의존 방향입니다. **안쪽이 불리는 쪽**입니다. AI 코드는 `pipeline`이 부를 때 실행되고, 스스로 아무것도 시작하지 않습니다.

### 한 줄씩

```python
# wemeet/sw/pipeline.py                                   ← SW파트가 소유한다
from wemeet.ai.detection import detect          # ① SW가 AI를 부르므로 여기서 import
from wemeet.ai.geometry import estimate_geometry  #  AI 쪽에는 이런 줄이 없어야 한다
from wemeet.schemas import DecodeResult, PipelineResult, RectifiedBarcode
from wemeet.sw.decoding import decode
from wemeet.sw.rectify import apply_field


def run(image_bgr):
    ms = {}

    # ② 검출 — 이 줄에서 AI파트 코드가 실행된다
    found = _timed(ms, "detect", detect, image_bgr)

    # ③ 판단은 SW가 한다. AI는 "찾은 것"만 돌려줬을 뿐이다
    if not found:
        return _fail(image_bgr, "not_detected", ms)
    best = max(found, key=lambda b: b.confidence)   # 2개 이상이면 1등만

    # ④ 1차 시도 — 기울기 보정만으로 읽히는지 본다.
    #    수직으로만 휜 바코드는 여기서 끝난다 (실측: 진폭 32px 까지 읽힌다)
    plain = RectifiedBarcode(
        image_gray_uint8=cv2.cvtColor(best.crop_bgr_uint8, cv2.COLOR_BGR2GRAY),
        source=best,
        field=None,                                 # None = 보정하지 않았다
    )
    result = _timed(ms, "decode_first", decode, plain)
    if result.text is not None:
        return _ok(image_bgr, best, plain, result, ms, len(found))

    # ⑤ 안 읽혔다. 얼마나 휘었는지 추정한다 — 이미지가 아니라 숫자를 받는다
    field = _timed(ms, "estimate", estimate_geometry, best)
    if field.confidence < 0.3:                      # 추정을 신뢰할 수 없다
        result.degraded = True
        return _fail_with(image_bgr, best, plain, result, ms, len(found))

    # ⑥ 보정 + 디코딩. 실패하면 보정 파라미터를 바꿔 최대 3회
    for i, opts in enumerate(_RETRIES):
        rectified = _timed(ms, "warp", apply_field, best, field, **opts)
        result = _timed(ms, "decode" if i == 0 else "retry", decode, rectified)
        result.retry_count = i
        result.candidate_count = len(found)         # ⑦ decode() 가 모르는 정보
        if result.text is not None:
            return _ok(image_bgr, best, rectified, result, ms, len(found))

    result.failure_reason = "decode_failed"
    return _fail_with(image_bgr, best, rectified, result, ms, len(found))
```

### 각 번호가 중요한 이유

| | 무엇을 보여주나 |
|---|---|
| ① | **import가 SW 쪽에만 있습니다.** AI 파일에 `from wemeet.sw...` 가 생기면 순환이 되고 CI가 막습니다 |
| ② | `detect()` 는 먼저 실행되지만 **부른 쪽은 `run()`** 입니다. "데이터는 AI → SW, 호출은 SW → AI"의 실제 모습입니다 |
| ③ | "0개면?", "2개 이상이면?" 같은 판단이 **파이프라인에 모여 있습니다.** AI 안으로 새면 두 파트가 같은 결정을 서로 다르게 구현합니다 |
| ④ | **1차 디코딩을 AI가 하지 않습니다.** AI가 디코더를 부르면 `wemeet.sw` import가 되어 규칙 위반입니다. "먼저 읽어보고 안 되면 보정한다"는 조율 판단이므로 지휘자의 일입니다 |
| ⑤ | AI가 돌려주는 것은 **제어점 좌표 몇십 개**입니다. 이미지가 아니므로 없는 바코드를 만들어낼 수 없습니다 |
| ⑥ | 재시도가 **보정을 다시 하는 것**입니다. 보간법이나 제어점 스케일을 바꿔 다시 펴고 다시 읽습니다 |
| ⑦ | `candidate_count` 를 파이프라인이 채웁니다 — 몇 개 찾았는지는 `decode()` 가 알 수 없는 정보입니다 |

### 재시도에서 무엇을 바꾸나

| 회차 | 바꾸는 것 | 근거 |
|---|---|---|
| 1 | `INTER_CUBIC` | 기본. 대부분 여기서 끝납니다 |
| 2 | `INTER_LANCZOS4` | 경계가 더 선명해집니다. 얇은 바에 유리 |
| 3 | `INTER_LINEAR` | 부드럽습니다. 노이즈가 심할 때 유리 |

제어점 스케일 조정(변위를 0.9배·1.1배)도 후보입니다. 실측 후 `docs/decisions/0006`에 순서를 확정합니다.

### 예외를 던지지 않으면 실패는 어떻게 전달되나

`try/except`로 감싼 뒤 **값으로 바꿔서** 돌려줍니다.

| 상황 | `detected` | `rectified` | `decode.text` | `failure_reason` | 비고 |
|---|---|---|---|---|---|
| 1차에서 바로 읽힘 | 있음 | `field=None` | `"123456"` | `None` | 약 150ms |
| 보정 후 읽힘 | 있음 | 있음 | `"123456"` | `None` | |
| 이미지 형식 오류 | `None` | `None` | `None` | `invalid_input` | 즉시 종료 |
| 검출 0개 | `None` | `None` | `None` | `not_detected` | 이후 단계 생략 |
| 검출 2개 이상 | 1등만 | 있음 | 보통 성공 | `None` | `candidate_count`에 총 개수 |
| 기하 추정 신뢰도 낮음 | 있음 | `field=None` | `None` | `decode_failed` | `degraded=True` |
| 재시도 후에도 실패 | 있음 | 있음 | `None` | `decode_failed` | `retry_count=3` |
| 500ms 초과 | 있음 | 있음 | 보통 성공 | `None` | **중단하지 않는다.** 시간만 기록 |

호출한 쪽(`server.py`)은 예외를 잡지 않고 `result.ok`만 봅니다. 판독 실패도 **HTTP 200**으로 보냅니다 — 실패는 서버 오류가 아니라 정상적인 처리 결과입니다.

### AI파트가 자기 코드를 단독으로 돌릴 수 있습니다

의존 방향을 이렇게 잡은 부수 효과입니다.

```python
# 노트북이나 학습 스크립트에서
from wemeet.ai.geometry import estimate_geometry

field = estimate_geometry(crop)      # 서버도 디코더도 OpenCV 보정도 필요 없다
print(field.control_points_src_norm)
```

만약 AI가 SW를 알고 있었다면, 추정 실험을 하려고 FastAPI와 디코더까지 끌어와야 합니다. 그러면 실험이 무거워지고, 서버 코드를 고칠 때 학습 스크립트가 깨집니다.

---

## 시간 예산

기획서의 0.5초를 단계별로 배분한 것입니다. `stage_ms`로 측정하고 목표 초과 시 원인을 특정합니다.

| 단계 | `stage_ms` 키 | 예산 |
|---|---|---|
| 검출 + 기울기 보정 | `detect` | 110 ms |
| **1차 디코딩** (조기 종료 지점) | `decode_first` | 40 ms |
| 기하 추정 (AI) | `estimate` | 150 ms |
| 기하 보정 (OpenCV) | `warp` | 20 ms |
| 디코딩 + 재시도 최대 3회 | `decode` / `retry` | 150 ms |
| 여유 | — | 30 ms |
| 합계 | | **500 ms** |

**1차에서 읽히면 약 150ms에 끝납니다.** 사전 실험에서 수직 방향으로만 휜 바코드는 보정 없이도 읽히는 것을 확인했으므로, 이 경로를 타는 비율이 상당할 것으로 예상됩니다.

**0.5초는 재시도를 포함한 전체 경로 기준입니다.**

예산을 초과해도 **중단하지 않습니다.** 중단하면 판독 가능한 화물을 스스로 버리는 것이 되고, 무엇보다 초과 원인을 알 수 없게 됩니다. 시간을 기록만 하고 준수 여부는 평가 단계에서 통계로 판정합니다.

---

## 복원할 수 있는 한계

기하 보정은 **있는 픽셀을 옮기는 것**이므로 정보가 소실된 것은 되살리지 못합니다. 사전 실험으로 경계를 측정했습니다.

| 가로 압축(감긴 각도) | 가장자리 모듈 폭 | 판정 |
|---|---|---|
| 35° | 4.5 px | 보정하면 읽힘 |
| 50° | 3.5 px | 보정하면 읽힘 |
| 70° | 1.9 px | 보정하면 읽힘 |
| 84° | 0.6 px | **불가 — 정보 소실** |

**가장 압축된 지점에서 모듈 폭이 1~2px 이상 남아 있어야 합니다.** 이 기준은 촬영 해상도 요구사항으로 이어집니다 (`docs/decisions/0004`).

반사도 마찬가지입니다. 픽셀을 옮겨도 하얗게 날아간 곳은 하얗습니다. 대신 보정 후 **바 높이 방향으로 여러 행을 읽어 포화된 행을 버리고 합칩니다** — 1D 바코드는 높이 방향으로 정보가 중복되므로, 있는 픽셀 중에서 고르는 것으로 상당 부분 처리됩니다. 전체 높이가 덮인 열은 실패로 보고합니다.

---

## 아직 정하지 않은 것

**추론 대상 하드웨어** — 0.5초 목표의 기준 장비가 기획서에 없습니다. 이것이 정해지지 않으면 "500ms 안에 끝났다"는 말의 의미가 확정되지 않습니다. 멘토 자문이 필요합니다.
담당: 팀장 / 기한: 통합 단계 전 (설계 문서 §20)

**기하 추정 신뢰도 임계값** — 위 코드의 `confidence < 0.3`은 근거 없는 초기값입니다.
담당: AI파트 / 기한: `docs/decisions/0003` 확정 후
