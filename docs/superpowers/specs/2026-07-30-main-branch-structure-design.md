# 메인 브랜치 구조 및 협업 규칙 설계

- 프로젝트: AI 기반 난인식 바코드 품질 인식 고도화 (WE-Meet, METRO팀)
- 저장소: `kim1034/2026-Wemeet-Barcode-Restoration` (public, 현재 빈 레포)
- 작성일: 2026-07-30
- 상태: 승인 대기

---

## 1. 개요

### 목적

8명(AI·SW·데이터 3파트)이 5개월간 병렬로 작업할 수 있는 저장소 구조와 협업 규칙을 확정한다. 이 단계가 끝나면 팀원은 `git clone` → `uv sync` → 자기 폴더에서 작업 시작이 가능하다.

### 이번 스펙의 범위

| 포함 | 제외 |
|---|---|
| 폴더 구조, 모듈 경계, 의존 규칙 | 각 스테이지의 실제 구현 |
| `schemas.py` 계약 (전체 코드) | 모델 학습 코드 |
| 실패·에러 처리 정책, 로깅 정책 | 더미 E2E 파이프라인 (2단계) |
| API 계약 (FastAPI ↔ React) | React 화면 구현 (2단계) |
| 브랜치·커밋·PR 규칙 | |
| CI 설정, 개발 환경 (uv, Vite) | |
| README의 파일별 구현 명세 형식 | |
| 데이터 수집 절차, 모델·디코더 선정 절차 | |

### 2단계 (별도 스펙)

이미지를 넣으면 끝까지 흘러가는 더미 파이프라인. `detection.py` / `restoration.py` / `decoding.py` / `pipeline.py` / `server.py` 스텁, `web/` 스캐폴드, E2E 스모크 테스트.

---

## 2. 전제와 제약

| 항목 | 내용 | 설계에 미친 영향 |
|---|---|---|
| 팀 구성 | 8명, 3파트. 3학년 4명, 2학년 2명, 1학년 2명 | 폴더를 파트 기준으로 분할 |
| 숙련도 | 높지 않음. 1~2학년이 절반 | `src/` 제거, 단계별 파일 하나, 명령어 1개 |
| 저장소 공개 | public | 실촬영 송장 데이터 커밋 불가. CI·브랜치 보호는 무료 |
| 모델 | **아직 미정** | 스캐폴드가 특정 모델을 전제하지 않음. `pyproject.toml`에 모델 라이브러리 없음 |
| 데이터 | **아직 수집 전** | 데이터 수집 절차를 스펙에 포함 |
| 일정 | 8월 레포 구축 / 9월 데이터+탐지 / 10월 복원+통합+피드백 / **11월 종료** | 뒤 단계를 압축하는 대신 범위를 축소 — §22 |
| 파이프라인 소유 | SW파트 (기획서 6.1) | 의존 방향이 `sw → ai` |

---

## 3. 레포 구조

```
2026-Wemeet-Barcode-Restoration/
├── wemeet/                    파이썬 코드는 전부 여기
│   ├── __init__.py
│   ├── schemas.py             AI ↔ SW 계약 (공용, 아무도 소유 안 함)
│   │
│   ├── ai/                    AI파트 전용
│   │   ├── __init__.py
│   │   ├── detection.py       Stage 1  탐지·크롭
│   │   ├── restoration.py     Stage 2  평탄화·반사 제거
│   │   ├── train_detection.py     탐지 모델 학습
│   │   ├── train_restoration.py   복원 모델 학습
│   │   └── configs/           학습 설정 YAML
│   │
│   ├── sw/                    SW파트 전용
│   │   ├── __init__.py
│   │   ├── decoding.py        Stage 3  디코더 + 피드백 루프
│   │   ├── pipeline.py        3단계 연결 (SW파트 소유)
│   │   └── server.py          FastAPI
│   │
│   └── data/                  데이터파트 전용
│       ├── __init__.py
│       ├── download.py        HF Hub에서 받아오기
│       ├── synthesis.py       역방향 합성
│       └── ground_truth.py    실촬영 정답 번호 확보 (sw.decoding 사용)
│
├── web/                       React 화면 (SW파트)
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api.ts             FastAPI 호출 (§23 API 계약)
│       └── assets/
├── downloads/                 실제 이미지·가중치 (git 제외, .gitkeep만 추적)
├── tests/
│   ├── conftest.py            테스트용 바코드 이미지 생성 픽스처
│   └── test_schemas.py
├── docs/
│   ├── architecture.md
│   ├── decisions/             결정 기록
│   └── superpowers/specs/     설계 문서
├── .github/
│   ├── workflows/ci.yml
│   ├── pull_request_template.md
│   └── ISSUE_TEMPLATE/task.md
├── .gitignore
├── pyproject.toml
├── README.md
└── CONTRIBUTING.md
```

### 설계 원칙

**1파트 = 1폴더.** 팀원은 폴더 하나만 본다. 학습 스크립트를 찾으려고 최상위로 나갈 필요가 없다.

**파일 이름 = 기획서의 파이프라인 단계.** `detection` / `restoration` / `decoding`은 기획서 4.1.1의 Stage 1/2/3과 일치한다. 기획서를 읽은 사람은 파일 이름만 보고 역할을 안다.

**폴더 대신 파일부터.** 각 스테이지는 디렉토리가 아니라 파일 하나로 시작한다. 500줄을 넘으면 그때 디렉토리로 분할한다. 초기부터 나누면 빈 폴더와 `__init__.py`만 늘어난다.

**`src/` 없음.** src 레이아웃의 이점(cwd로부터의 우발적 import 방지)은 초보자가 체감하지 못하고, 경로 깊이만 한 칸 늘린다.

**`web/`은 패키지 밖.** React는 자바스크립트이므로 파이썬 패키지 안에 넣을 수 없다.

**실제 데이터 폴더는 `downloads/`.** `wemeet/data/`(데이터파트 코드)와 이름이 겹치지 않게 한다. `datasets/`로 두면 초보자가 반드시 혼동한다.

---

## 4. 모듈 경계와 의존 규칙

### 의존 방향

세 파트와 계약 모듈이 4개 층을 이룬다. 화살표는 위에서 아래로만 흐른다.

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

허용되는 import는 이 다섯 개뿐이다.

| 주체 | 부를 수 있는 것 |
|---|---|
| `ai` | `schemas` |
| `sw` | `schemas`, `ai` |
| `data` | `schemas`, **`sw.decoding`** |
| `schemas` | (없음) |

반대 방향 화살표가 하나만 생겨도 순환이 완성되어 `ImportError: cannot import name ... from partially initialized module`이 발생한다. 이 오류는 import 진입점에 따라 나타났다 사라지므로 원인 추적이 어렵고, 하필 통합 시점에 터진다.

### `data → sw.decoding` 을 허용하는 이유

§14 5단계가 실촬영 테스트셋의 정답 번호를 이렇게 확보한다 — "평탄 상태로 촬영 → **디코더로 정상 판독** → 정답 번호 기록". 디코더는 `sw/decoding.py`에 있으므로 데이터파트가 그것을 불러야 한다.

`data`는 아무도 부르지 않는 최상위 층이므로 이 화살표가 순환을 만들지 않는다. 다만 **`sw.pipeline`과 `sw.server`는 부를 수 없다.** 데이터 스크립트가 서버나 파이프라인 전체를 끌어오면 무거워지고, 학습 데이터 생성이 서버 코드 변경에 깨지게 된다.

### 팀 규칙 (3줄)

> **AI파트는 자기 파일에 `wemeet.sw` 나 `wemeet.data` 라는 글자를 쓰지 않는다. 공용이 필요하면 `wemeet.schemas` 를 쓴다.**
>
> **데이터파트는 `wemeet.sw.decoding` 까지만 쓸 수 있다. `pipeline` 과 `server` 는 쓰지 않는다.**
>
> **함수 안에서 import 하지 않는다.**

세 번째 규칙의 근거: 함수 내부 import는 순환을 런타임으로 미루는 응급처치일 뿐 순환 자체를 제거하지 않는다. 다른 진입점에서 재발하며, 6개월 뒤에는 왜 그 자리에 있는지 알 수 없게 된다.

### 강제 방법

`import-linter`가 CI에서 강제한다. 문서로만 둔 규칙은 8명 팀에서 3주면 잊힌다.

```toml
[tool.importlinter]
root_package = "wemeet"

[[tool.importlinter.contracts]]
name = "ai는 sw와 data를 import하지 않는다"
type = "forbidden"
source_modules = ["wemeet.ai"]
forbidden_modules = ["wemeet.sw", "wemeet.data"]

[[tool.importlinter.contracts]]
name = "sw는 data를 import하지 않는다"
type = "forbidden"
source_modules = ["wemeet.sw"]
forbidden_modules = ["wemeet.data"]

[[tool.importlinter.contracts]]
name = "data는 sw.decoding까지만 쓴다"
type = "forbidden"
source_modules = ["wemeet.data"]
forbidden_modules = ["wemeet.ai", "wemeet.sw.pipeline", "wemeet.sw.server"]

[[tool.importlinter.contracts]]
name = "schemas는 아무것도 import하지 않는다"
type = "forbidden"
source_modules = ["wemeet.schemas"]
forbidden_modules = ["wemeet.ai", "wemeet.sw", "wemeet.data"]
```

마지막 계약이 이 구조의 바닥을 지킨다. `schemas.py`가 무언가를 부르기 시작하면 잎이 아니게 되고, 그 순간 순환을 막는 근거 자체가 사라진다.

`schemas.py`는 `numpy`와 표준 라이브러리는 import한다. "아무것도 import하지 않는다"는 **`wemeet` 내부 모듈에 한한 규칙**이다.

### 채택하지 않은 대안

| 방법 | 불채택 이유 |
|---|---|
| 계층 구조 (layers) | 모듈이 7개뿐. 수십 개로 늘면 전환 |
| 의존성 역전 (Protocol + 주입) | 초보자에게 개념 2개 추가. 더미 파이프라인이 같은 효과를 더 싸게 냄 |
| 지연 import | 순환을 숨김. 금지 규칙에 포함 |
| `TYPE_CHECKING` | `schemas.py`로 이미 해결되어 필요 없음 |
| 모듈 합치기 | AI팀·SW팀이 물리적으로 다른 사람이므로 분리가 맞음 |

---

## 5. 스테이지 간 계약 — `wemeet/schemas.py`

이 프로젝트에서 가장 중요한 파일이다. 9월 통합의 성패가 여기서 갈린다.

```python
"""AI파트와 SW파트가 주고받는 데이터 형식.

이 파일은 wemeet 안의 어떤 모듈도 import 하지 않는다.
변경하려면 AI·SW 양쪽과 먼저 상의한다 (CONTRIBUTING.md).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectedBarcode:
    """Stage 1 (탐지) → Stage 2 (복원)"""

    crop_bgr_uint8: np.ndarray   # BGR 순서, 0~255, shape (H, W, 3)
    angle_deg_ccw: float         # 원본에서 회전 보정한 각도. 도(degree), 반시계
    confidence: float            # 0.0 ~ 1.0

    def __post_init__(self) -> None:
        assert self.crop_bgr_uint8.dtype == np.uint8
        assert self.crop_bgr_uint8.ndim == 3
        assert self.crop_bgr_uint8.shape[2] == 3
        assert 0.0 <= self.confidence <= 1.0


@dataclass
class RestoredBarcode:
    """Stage 2 (복원) → Stage 3 (디코딩)"""

    image_gray_uint8: np.ndarray   # 흑백 단일 채널, 0~255, shape (H, W)
    source: DetectedBarcode

    def __post_init__(self) -> None:
        assert self.image_gray_uint8.dtype == np.uint8
        assert self.image_gray_uint8.ndim == 2


@dataclass
class DecodeResult:
    """Stage 3 (디코딩) → 최종 출력"""

    text: str | None                              # 판독 실패 시 None
    symbology: str | None                         # 예: "CODE128". 실패 시 None
    retry_count: int                              # 0 ~ 3
    stage_ms: dict[str, float] = field(default_factory=dict)
    # 예: {"detect": 42.1, "restore": 210.3, "decode": 18.7, "retry": 95.0}

    def __post_init__(self) -> None:
        assert 0 <= self.retry_count <= 3
        if self.text is None:
            assert self.symbology is None

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


@dataclass
class PipelineResult:
    """파이프라인 최종 반환형. UI가 필요한 모든 것을 담는다.

    기획서 4.1.2 5단계("복원 이미지와 디코딩 텍스트를 UI에 동시 매핑")와
    7.2("복원 이미지와 판독 결과를 동시 표출하는 투명한 파이프라인")를
    만족하려면 중간 산출물이 최종 반환형까지 살아 있어야 한다.
    """

    original_bgr_uint8: np.ndarray        # 입력 원본 (대조 표시용)
    detected: DetectedBarcode | None      # 탐지 실패 시 None
    restored: RestoredBarcode | None      # 탐지 실패 시 None
    decode: DecodeResult                  # 실패해도 항상 존재 (text=None)

    @property
    def ok(self) -> bool:
        return self.decode.text is not None
```

### 왜 `DecodeResult`만 반환하면 안 되는가

`DecodeResult`에는 이미지가 없다. 파이프라인이 그것만 반환하면 중간 산출물이 버려지고, **FastAPI가 React에 복원 이미지를 보낼 방법이 사라진다.**

기획서 7.2는 이 프로젝트가 Scandit 같은 상용 솔루션과 차별화되는 지점으로 "내부가 블랙박스가 아니라 복원 결과를 작업자에게 보여준다"를 내세운다. 그 기능이 계약 단계에서 불가능해지면 안 된다.

따라서 `pipeline.run()`의 반환형은 `PipelineResult`다.

### 스테이지 함수 시그니처

각 파트가 구현할 함수의 형태를 계약으로 고정한다. 여기서 벗어나면 통합이 깨진다.

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

`restore()`가 리스트가 아니라 단일 객체를 받는 것이 중요하다. "탐지 결과가 여러 개일 때 무엇을 복원할지"는 AI파트가 아니라 파이프라인이 정할 문제다 (아래 "실패와 예외 상황" 참고). 이 경계를 흐리면 두 파트가 같은 결정을 서로 다르게 구현한다.

### 실패와 예외 상황

기획서 6.1은 SW파트 업무로 "예외 처리 로직 구현"을 명시한다. 정상 경로만 정의하면 각 파트가 서로 다르게 가정하므로, 아래를 계약의 일부로 확정한다.

**원칙: 파이프라인은 예외를 던지지 않는다.** 어떤 상황에서도 `PipelineResult`를 반환한다. 현장 컨베이어에서 예외가 올라오면 서비스가 멈추고, 그것이 곧 노리드 존 정체다. 실패는 예외가 아니라 **값**으로 표현한다.

| 상황 | 동작 | 반환값 |
|---|---|---|
| 이미지 로드 실패·형식 오류 | 즉시 종료 | `detected=None`, `restored=None`, `decode.text=None`, `failure_reason="invalid_input"` |
| 탐지 0개 | 복원·디코딩 생략 | `detected=None`, `restored=None`, `failure_reason="not_detected"` |
| **탐지 2개 이상** | **confidence 최상위 1개만** 처리 | `detected`에 1등. `decode.candidate_count`에 총 개수 기록 |
| 복원 모델 추론 실패(예외) | 예외를 잡고 **원본 크롭으로 대체** 진행 | `restored`에 크롭 원본, `decode.degraded=True` |
| 1차 디코딩 실패 | 피드백 루프 최대 3회 재시도 | `retry_count`에 실제 시도 횟수 |
| 재시도 후에도 실패 | 정상 종료 | `decode.text=None`, `failure_reason="decode_failed"` |
| 500ms 예산 초과 | **중단하지 않고 완료** 후 기록 | `stage_ms`에 실제 시간. 초과는 지표로 판정 |

두 항목만 부연한다.

**탐지 2개 이상은 물류 현장에서 흔하다.** 박스에 송장 바코드와 상품 바코드가 같이 붙어 있다. 1등만 처리하는 것이 이번 범위이고, 총 개수를 기록해두면 나중에 "여러 개 처리가 필요한가"를 데이터로 판단할 수 있다.

**500ms 초과 시 중단하지 않는다.** 중단하면 판독 가능한 화물을 스스로 버리는 것이 되고, 무엇보다 초과 원인을 알 수 없게 된다. 시간을 기록만 하고 예산 준수 여부는 평가 단계에서 통계로 판정한다 (§17).

이를 위해 `DecodeResult`에 세 필드를 추가한다.

```python
@dataclass
class DecodeResult:
    text: str | None
    symbology: str | None
    retry_count: int
    stage_ms: dict[str, float] = field(default_factory=dict)

    failure_reason: str | None = None    # invalid_input / not_detected / decode_failed
    candidate_count: int = 0             # 탐지된 바코드 총 개수
    degraded: bool = False               # 복원 실패로 원본 크롭을 쓴 경우

    def __post_init__(self) -> None:
        assert 0 <= self.retry_count <= 3
        if self.text is None:
            assert self.symbology is None
            assert self.failure_reason is not None    # 실패에는 항상 이유가 있다
        else:
            assert self.failure_reason is None
```

마지막 assert가 중요하다. **실패했는데 이유가 없는 상태를 계약 수준에서 금지한다.** 이유 없는 실패가 쌓이면 §17의 훼손 유형별 세분화 평가를 할 수 없다.

### 로깅 정책

현장 운영을 가정한 시스템이므로 무엇을 남길지 정한다. 그리고 §7의 개인정보 정책과 충돌하지 않아야 한다.

**남긴다**: 처리 시각, `stage_ms`, `retry_count`, `failure_reason`, `candidate_count`, `degraded`, 성공/실패 여부.

**남기지 않는다**: 입력 이미지, 크롭 이미지, **디코딩된 바코드 번호**.

바코드 번호는 화물을 특정하는 식별자다. 로그 파일이 저장소나 공유 폴더에 올라가는 순간 §7의 정책이 무의미해진다. 디버깅에 이미지가 필요하면 개발 환경에서만 켜지는 별도 플래그로 처리하고, 그 출력 경로는 `downloads/` 아래로 둔다 (git 제외 대상).

### 설계 결정 두 가지

**이름에 단위를 박는다.** `angle`이라고만 쓰면 라디안을 넣는 사람이 나온다. `angle_deg_ccw`면 나오지 않는다. `crop_bgr_uint8`도 같은 이유다 — OpenCV는 BGR, PyTorch는 RGB이므로 이 실수가 가장 자주 발생한다. `dataclass`의 타입 힌트는 "이건 `float`이다"까지만 보장하고 단위는 보장하지 않으므로, 단위는 이름과 `__post_init__` assert로 강제한다.

**`stage_ms`로 단계별 시간을 기록한다.** 기획서 2.1은 "전 과정 0.5초 이내", 4.3.3은 "최대 3회 재시도"라고 하여 재시도가 예산에 포함되는지 정의가 충돌한다. 단계별로 시간을 찍어두면 어느 단계가 예산을 초과했는지 즉시 드러난다.

### 시간 예산 배분

기획서의 0.5초를 단계별로 배분한다. `stage_ms`로 측정하고 목표 초과 시 원인을 특정한다.

| 단계 | 예산 |
|---|---|
| 탐지 (detect) | 100 ms |
| 복원 (restore) | 250 ms |
| 디코딩 1차 (decode) | 50 ms |
| 재시도 여유 (retry, 최대 3회) | 100 ms |
| 합계 | **500 ms** |

정의: **0.5초는 재시도를 포함한 전체 경로 기준**이다. 기획서 2.1의 표현을 이렇게 확정한다.

---

## 6. 오너십 — CODEOWNERS → **폐기 (2026-08-04)**

> **이 절은 폐기됐다. 아래 내용은 구현하지 않는다.**
>
> **왜 폐기했나**: 팀 규모에 비해 장치가 복잡하고, PR을 올리는 인원이 고학년이라 오너십을 파일로 강제할 필요가 없다고 판단했다. 재도입 계획 없음.
>
> **대신 무엇을 하나**: 리뷰어는 PR 작성자가 직접 지정한다. `main` 브랜치 보호는 "PR 필수 + 승인 1개 + CI 통과"만 켜고 `Require review from Code Owners`는 켜지 않는다. 파트별 담당 범위는 README의 파트 표와 `CONTRIBUTING.md`에 글로 적는다.
>
> **잃은 것 (알고 폐기했다)**: `wemeet/schemas.py` 변경에 AI·SW 양쪽 승인을 **강제**하는 장치가 없어졌다. 아래 본문이 "구조적으로 차단된다"고 한 부분이 사라진다. 대체는 PR 템플릿 체크박스와 `CONTRIBUTING.md`의 규칙뿐이므로, 한쪽이 말없이 계약을 바꾸는 사고는 사람이 막아야 한다. 9월 통합에서 계약 불일치가 나오면 이 결정을 먼저 의심할 것. `pyproject.toml`·`.github/` 무단 변경 차단도 같이 사라졌다.
>
> 아래 원문은 지우지 않고 기록으로 남긴다 — 11월 보고서에서 "왜 이 방식을 안 썼나"에 답할 근거다.

```
# .github/CODEOWNERS   ← 폐기. 이 파일은 만들지 않는다

# 기본값 — 아래 규칙에 걸리지 않는 모든 파일
*                          @시선

# 파트별 담당
/wemeet/ai/                @시선 @대원
/wemeet/sw/                @종연 @도훈
/web/                      @종연 @도훈
/wemeet/data/              @강민 @아라
/docs/decisions/           @시선

# 공용 — 양쪽 승인이 필요한 계약
/wemeet/schemas.py         @시선 @종연

# 빌드·CI 설정 — 아무나 바꾸면 안 된다
/pyproject.toml            @시선 @종연
/.github/                  @시선 @종연
/.gitignore                @시선 @종연
```

(이름 자리에 실제 GitHub 아이디가 들어간다)

**첫 줄 `*`가 중요하다.** CODEOWNERS는 어떤 규칙에도 걸리지 않는 파일에 대해 오너를 요구하지 않는다. 즉 기본값이 없으면 `README.md`, `pyproject.toml`, `.github/workflows/ci.yml` 같은 파일을 **아무나 승인해서 머지할 수 있다.** CI 설정을 누구든 바꿀 수 있다는 뜻이다.

`pyproject.toml`과 `.github/`를 따로 지정한 이유도 같다. 여기에는 의존성 목록, `import-linter` 계약, CI 검사 항목이 들어 있다. 이 파일이 뚫리면 §4의 의존 규칙 강제가 무력화된다.

마지막 줄이 핵심이다. 계약을 변경하려면 AI·SW 양쪽 승인이 필요하므로, **한쪽이 말없이 출력 형식을 바꾸는 사고가 구조적으로 차단된다.**

Organization 팀(`@org/team`)은 개인 계정 레포에서 쓸 수 없으므로 개인 아이디를 나열한다. public 레포이므로 브랜치 보호(담당자 승인 전 머지 차단)는 무료 계정에서 동작한다 — 유료가 필요한 것은 private 레포다.

### 선행 작업

1. 팀원 8명을 레포 Collaborator로 초대
2. 각자의 GitHub 아이디 수집

### 1학년 배치 — **이 부분은 유효하다** (위 폐기 대상 아님)

기획서 6.1에 개인별 파트 매핑이 없다. 1학년 2명(이다현·이아침)은 데이터 구축 단계의 OBB 재라벨링 작업(§14 3단계)에 배치하고, 이후 각자 전공(AI/SW) 파트에 편입한다. 편입 시점에 README의 파트 표를 갱신한다.

---

## 7. 데이터와 가중치 관리

Git에는 **스크립트만** 커밋한다. 실제 파일은 Hugging Face Hub에 둔다.

```
wemeet/data/download.py   →   downloads/datasets/    (git 제외)
                              downloads/weights/     (git 제외)
```

팀원 실행 명령: `uv run python -m wemeet.data.download`

### HF Hub 레포 구성

| 용도 | 공개 여부 |
|---|---|
| 오픈소스 데이터 + 합성 데이터 + 모델 가중치 | public |
| **실촬영 송장 데이터** | **private** |

### 개인정보 (필수)

실제 송장에는 수취인 이름·주소·연락처가 인쇄되어 있다. 이 레포는 public이므로 해당 이미지는 커밋할 수 없고, public HF 레포에도 올릴 수 없다.

**촬영 단계에서 개인정보 영역을 물리적으로 차폐한다** (테이프·마스킹지). 촬영 후 블러 처리보다 촬영 전 차폐가 확실하다. 이 절차를 §14 5단계 작업 지침에 포함한다. 준비물과 순서는 `docs/external/README.md` §6에 정리했다.

### `.gitignore`

실제 파일과 동일하다. 두 예외가 중요하다 — `downloads/*`(디렉토리가 아니라 내용)로 써야 `!downloads/.gitkeep`이 동작하고, `!docs/reports/*.pdf`가 없으면 기획서 2.2의 성과 보고서가 조용히 무시된다.

```
# 데이터와 가중치는 Hugging Face Hub에 둔다 (docs/external/huggingface.md)
#
# "downloads/" 가 아니라 "downloads/*" 인 이유:
# 디렉토리 자체를 제외하면 git이 안으로 들어가지 않아서
# 아래 !downloads/.gitkeep 예외가 무시된다.
downloads/*
!downloads/.gitkeep

*.pt
*.pth
*.onnx

# 송장 사진이 실수로 커밋되는 것을 막는다.
# 문서용 이미지와 React 정적 자원은 예외로 허용한다.
*.jpg
*.jpeg
*.png
!docs/**/*.png
!docs/**/*.jpg
!web/src/assets/**

# 토큰과 API 키 (HF_TOKEN, WANDB_API_KEY)
.env
.env.*
!.env.example

# 계획서 원본은 커밋하지 않는다.
# 1페이지에 팀원 8명의 학번이 있고 이 저장소는 public이다.
# 팀 내부에서 별도로 공유한다.
# 단 성과 보고서는 산출물이므로 예외로 허용한다 (기획서 2.2).
*.pdf
!docs/reports/*.pdf

# 파이썬
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/

# 실험 기록 로컬 캐시
wandb/
runs/

# 에디터·OS
.vscode/
.idea/
.DS_Store
Thumbs.db
```

### 채택하지 않은 대안

| 방법 | 불채택 이유 |
|---|---|
| Git LFS | 무료 한도가 저장 1GB·대역폭 월 1GB. 수만 장이면 8명이 clone하는 첫 주에 초과 |
| DVC | 원격 스토리지 별도 설정 필요. 1~2학년이 `dvc pull`과 git 명령을 혼동 |

---

## 8. 개발 환경 — uv

팀원이 외울 명령어는 하나다.

```
uv sync
```

파이썬 버전을 자동으로 맞추고, 가상환경 활성화가 필요 없다(`uv run python ...`). `uv.lock`이 생성되어 8명 환경이 동일해진다. CUDA용 PyTorch 인덱스는 `pyproject.toml`에 선언한다.

pip 방식은 윈도우에서 `venv` 생성 → 활성화(OS별로 명령이 다름) → 설치 → CUDA 별도 설치의 4단계이며, 활성화 단계에서 가장 많이 막힌다.

### 초기 의존성

모델이 미정이므로 **모델 라이브러리는 넣지 않는다.**

| 분류 | 패키지 |
|---|---|
| 코어 | `numpy`, `opencv-python`, `pillow` |
| 디코더 후보 | `pyzbar`, `zxing-cpp` (§14 2단계 비교용. 선정 후 하나만 남긴다) |
| 데이터 생성 | `python-barcode` (테스트 픽스처와 합성 원본 렌더링) |
| 서버 | `fastapi`, `uvicorn` |
| 실험 | `wandb` |
| 개발 | `pytest`, `ruff`, `import-linter` |

`ultralytics` / `torch`는 `docs/decisions/0002`, `0003` 확정 후 추가한다.

세 번째 디코더 후보인 OpenCV `barcode` 모듈은 별도 패키지가 아니다. `cv2.barcode.BarcodeDetector`는 **OpenCV 4.8부터 main 패키지(`opencv-python`)에 포함**된다. 그 이전 버전은 `opencv-contrib-python`이 필요하므로 `pyproject.toml`에 `opencv-python>=4.8` 하한을 걸어둔다. 이걸 놓치면 §14 2단계에서 "왜 `cv2.barcode`가 없지"로 시간을 쓴다.

---

## 9. 협업 규칙

### 브랜치

`main`에 직접 push하지 않는다. 항상 브랜치를 만들어 PR을 올린다.

```
main
├── ai/detection-baseline
├── sw/pipeline-stub
└── data/synth-v1
```

이름 형식은 `<파트>/<내용>`이다. 브랜치 목록만 보고 누가 무엇을 하는지 파악된다.

### 커밋 메시지

`[파트] 무엇을 했다` 형식.

```
[AI] YOLO-OBB 학습 스크립트 추가
[SW] 디코딩 재시도 루프 구현
[데이터] 곡률 왜곡 합성 함수 추가
[문서] 탐지 모델 선정 결과 기록
```

Conventional Commits는 채택하지 않는다. 영어 타입 키워드를 외우는 부담이 학생 팀에서 얻는 이익보다 크다.

### main 브랜치 보호

- PR 없이 머지 금지
- **승인 1개 이상** — 리뷰어는 PR 작성자가 직접 지정한다. 담당 파트원인지는 검사하지 않는다 (§6 폐기)
- CI 통과 필수

### Issue 라벨

- 파트: `part:ai` / `part:sw` / `part:data`
- 마일스톤: `2026-08` ~ `2026-11`

팀원이 자기 파트 라벨로 필터링하면 할 일 목록이 나온다.

---

## 10. CI

`.github/workflows/ci.yml` 하나로 PR마다 자동 실행한다. public 레포이므로 GitHub Actions는 무료·무제한이다.

| 검사 | 하는 일 |
|---|---|
| `import-linter` | AI파트가 `wemeet.sw`를 부르면 차단 |
| `ruff check` + `ruff format --check` | PEP8 준수 및 스타일 자동 통일 |
| `pytest` | 테스트 통과 확인 |

**모델 학습은 CI에서 돌리지 않는다.** GitHub 무료 러너에 GPU가 없다. 2단계에서 만들 더미 파이프라인은 CPU로 동작하므로 E2E 흐름은 매 PR마다 검사할 수 있다.

### 초보 팀에서 CI가 필요한 이유

1. 초보자는 자기가 무엇을 망가뜨렸는지 모른다. CI가 5분 안에 알려주면 학습이 일어난다. 없으면 2주 뒤에 남이 발견하고, 그때는 원인 추적이 어렵다.
2. 3학년의 리뷰 시간을 지킨다. `ruff`가 스타일을 자동 통일하면 공백·import 순서 지적이 사라지고, 리뷰가 로직에 집중된다.

---

## 11. 테스트

1단계에서는 `tests/test_schemas.py` 하나를 만든다. 계약이 작동하는지 검증한다.

```python
import numpy as np
import pytest
from wemeet.schemas import DetectedBarcode, DecodeResult


def test_잘못된_dtype은_거부된다():
    with pytest.raises(AssertionError):
        DetectedBarcode(
            crop_bgr_uint8=np.zeros((10, 10, 3), dtype=np.float32),
            angle_deg_ccw=0.0,
            confidence=0.5,
        )


def test_판독_실패시_symbology도_None이어야_한다():
    with pytest.raises(AssertionError):
        DecodeResult(text=None, symbology="CODE128", retry_count=0)


def test_total_ms는_단계별_시간의_합이다():
    r = DecodeResult(
        text="123", symbology="CODE128", retry_count=0,
        stage_ms={"detect": 40.0, "restore": 200.0, "decode": 20.0},
    )
    assert r.total_ms == 260.0
```

테스트 이름은 한국어로 작성한다. 실패 시 출력만 보고 무엇이 깨졌는지 파악된다.

### 테스트 이미지는 코드로 만든다

`downloads/`는 git 제외 대상이라 **GitHub 러너에는 존재하지 않는다.** 테스트가 실제 이미지 파일을 읽으면 CI에서 전부 실패한다. 로컬에서는 통과하고 CI에서만 깨지는, 원인 찾기 가장 짜증나는 형태다.

따라서 테스트용 바코드 이미지를 **코드로 생성**한다.

```python
# tests/conftest.py
import numpy as np
import pytest


@pytest.fixture
def clean_barcode_bgr() -> np.ndarray:
    """정상 Code128 바코드 이미지. python-barcode로 렌더링한다."""
    ...


@pytest.fixture
def warped_barcode_bgr(clean_barcode_bgr) -> np.ndarray:
    """곡률 왜곡을 준 바코드. wemeet.data.synthesis 를 사용한다."""
    ...


@pytest.fixture
def glared_barcode_bgr(clean_barcode_bgr) -> np.ndarray:
    """반사 효과를 준 바코드."""
    ...
```

이 픽스처는 §14 4단계의 합성 파이프라인과 같은 코드를 쓴다. 미리 만들면 두 곳에서 재사용되고, 합성 코드 자체가 테스트로 검증된다.

**정답 번호를 아는 이미지를 만들 수 있다는 점이 중요하다.** 렌더링할 때 넣은 번호가 곧 정답이므로, 디코딩 성공/실패를 CI에서 자동 판정할 수 있다.

### 재현성 — 시드 고정

§16에서 W&B config에 `seed`를 기록하기로 했지만, 기록만 하고 실제로 고정하지 않으면 재현이 안 된다. 고정 유틸을 `wemeet/schemas.py`가 아닌 별도 위치에 둔다.

```python
# wemeet/ai/train_detection.py 등 학습 스크립트 진입부에서 호출
def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    # torch 도입 후: torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)
```

합성 데이터 생성(`wemeet/data/synthesis.py`)에도 같은 시드를 적용한다. 데이터가 매번 달라지면 모델 비교가 성립하지 않는다.

### 2단계에서 추가할 것

E2E 스모크 테스트 — 위 픽스처로 만든 이미지를 `pipeline.run()`에 넣어 `PipelineResult`가 나오는지 확인한다. 더미 구현이라도 통과해야 하며, §5의 실패 정책(탐지 0개, 2개 이상, 예산 초과)도 여기서 검증한다.

---

## 12. 팀원 안내 문서 — README + 파트별 문서

### 문서를 두 층으로 나눈다

파일별 구현 명세를 README에 다 넣으면 세 파트 내용이 뒤섞여 길어지고, 1~2학년이 자기 부분을 찾지 못한다. 두 층으로 나눈다.

| 문서 | 내용 | 분량 목표 |
|---|---|---|
| `README.md` | 무슨 문제를 푸는지, 파이프라인 3단계, 시작하는 방법, **내 파트 링크** | 한 화면에 핵심이 들어오게 |
| `docs/parts/ai.md` | AI파트가 담당 파일에서 무엇을 구현하는지 전부 | 링크 하나로 자기 할 일 완결 |
| `docs/parts/sw.md` | SW파트 동일 | |
| `docs/parts/data.md` | 데이터파트 동일 | |

**팀원은 링크 하나만 연다.** "1파트 = 1폴더" 원칙과 같은 논리를 문서에도 적용한 것이다.

동일 내용을 해당 파이썬 파일의 docstring에도 넣는다 — 문서를 열지 않고 파일부터 여는 팀원이 있다.

### README에 반드시 들어갈 것

1. **무슨 문제를 푸는가** — 노리드 존에서 작업자가 공정을 거꾸로 걸어가는 상황. 왜 하는지 모르면 동기가 안 생긴다
2. **우리가 하지 않는 것** — 정렬은 기존 설비가 한다. "AI가 다 해결한다"가 아니라는 것을 명시
3. 파이프라인 3단계 도식 + **각 단계의 담당 파트**
4. 시작하기 3단계 (`clone` → `uv sync` → `download`)
5. **파트별 시작 지점 표** (아래)
6. 외워야 할 것 4개 (아래)
7. 브랜치·커밋·PR 규칙
8. **지금 코드가 없다는 사실** — 8월에 골격을 만든다는 상태 안내. 없으면 문서에 적힌 파일을 찾다가 혼란스러워한다

### 파트별 문서에 반드시 들어갈 것

각 파일마다 아래 항목을 채운다.

| 항목 | 왜 |
|---|---|
| 무엇을 하는 파일인가 | 한두 문장, 비유 가능 |
| 구현할 함수 시그니처 | §5의 계약 그대로 |
| 입력·출력·실패 시 반환 표 | 애매하면 파트마다 다르게 구현한다 |
| 지켜야 할 것 (시간 예산 포함) | |
| 아직 안 정해진 것 + `decisions/` 링크 | 모델이 미정인 상태를 숨기지 않는다 |
| **막히면 이렇게 시작하세요** | 아래 참고 |
| 절대 하면 안 되는 것 | import 규칙 |
| 자주 헷갈리는 것 (Q&A) | |

**"막히면 이렇게 시작하세요"가 핵심이다.** 동작하는 가짜 구현을 코드로 제시한다. 1~2학년이 빈 파일 앞에서 멈추는 것을 막고, 가짜라도 넣으면 파이프라인 전체가 돌아 자기 작업이 어디에 쓰이는지 눈으로 확인한다.

### 파일별 명세 형식 예시

````markdown
### `wemeet/ai/detection.py` — Stage 1 탐지
**담당: AI파트 (김시선, 김대원)**

#### 무엇을 하는 파일인가
사진 한 장을 받아서 바코드가 있는 부분만 잘라냅니다.
비스듬히 놓인 바코드는 똑바로 세워서 잘라냅니다.

#### 구현할 함수
```python
def detect(image_bgr: np.ndarray) -> list[DetectedBarcode]:
```

| | 내용 |
|---|---|
| 입력 | BGR 이미지, 0~255, shape `(H, W, 3)`, dtype `uint8` |
| 출력 | `DetectedBarcode` 리스트 |
| 못 찾으면 | 빈 리스트 `[]` (예외를 던지지 말 것) |
| 여러 개 찾으면 | `confidence` 내림차순 정렬 |

#### 지켜야 할 것
- 잘라낸 이미지는 회전을 보정해서 바코드가 수평이 되게
- `wemeet.sw` 를 import하지 말 것 (CI가 차단합니다)
- 100ms 안에 끝내기 — 전체 예산 500ms 중 이 단계 배정분

#### 아직 안 정해진 것
어떤 모델을 쓸지 미정 → `docs/decisions/0002-탐지-모델-선정.md`

#### 막히면 이렇게 시작하세요
`TODO` 자리에 아래를 넣으면 파이프라인이 일단 돕니다.

```python
h, w = image_bgr.shape[:2]
crop = image_bgr[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
return [DetectedBarcode(crop_bgr_uint8=crop, angle_deg_ccw=0.0, confidence=0.5)]
```
````

"막히면 이렇게 시작하세요" 항목이 핵심이다. 1~2학년이 빈 파일 앞에서 멈추는 것을 막고, 가짜 구현이라도 넣으면 파이프라인 전체가 돌아 자기 작업이 어디에 쓰이는지 눈으로 확인한다.

### 파트별 시작 지점 표

README 최상단에 배치한다.

| 파트 | 열어볼 폴더 | 첫 파일 |
|---|---|---|
| AI | `wemeet/ai/` | `detection.py` 의 `TODO` |
| SW | `wemeet/sw/` + `web/` | `pipeline.py` |
| 데이터 | `wemeet/data/` | `synthesis.py` 의 `TODO` |

### 외워야 할 것 4개

README에 명시한다.

- 명령어 1개: `uv sync`
- 내 폴더 1개: `wemeet/ai/` 또는 `wemeet/sw/` 또는 `wemeet/data/`
- 규칙 1개: 다른 파트 폴더를 import하지 않는다 (예외는 파트별 문서에)
- 상의해야 하는 파일 1개: `schemas.py`

규칙을 파트별로 정확히 쓰면 4줄이 되므로, README에는 한 줄로 줄이고 정확한 범위는 파트별 문서에 적는다. 어기면 CI가 막아주므로 외우지 못해도 사고는 나지 않는다.

---

## 13. 결정 기록 — `docs/decisions/`

모델과 데이터가 미정이므로, 결정할 항목마다 문서를 **"미결정" 상태로 미리 만들어둔다.**

```
docs/decisions/
├── 0001-바코드-심볼로지-범위.md
├── 0002-탐지-모델-선정.md
├── 0003-복원-모델-선정.md
├── 0004-데이터-수집-및-GT-프로토콜.md
├── 0005-평가-지표-체계.md
└── 0006-디코더-선정.md
```

각 문서 형식:

```markdown
# 0002. 탐지 모델 선정
상태: 미결정
결정자:
결정일:

## 무엇을 정하는가
## 후보
## 비교 기준
## 실험 결과
## 결정과 이유
```

기획서 2.2의 산출물에 "기술 문서 및 성능 평가 결과 보고서"가 있다. 결정 문서를 누적하면 11월 보고서 작성 시 "왜 이 모델을 선택했나"에 대한 근거가 이미 확보되어 있다.

---

## 14. 데이터 수집 절차

순서가 중요하다. 앞 단계 결과 없이 다음 단계로 진행하면 재작업이 발생한다.

**착수 시점은 2026년 9월이다.** 기획서 5.1은 이 작업을 8월로 계획했으나 한 달 늦어졌다. 아래는 착수일 기준 **상대 주차**로 기술한다 — 달력 날짜에 묶어두면 일정이 또 밀릴 때 문서 전체를 고쳐야 한다. 후속 단계(복원·통합·최적화·평가) 일정 재조정은 §22에서 다룬다.

### 0단계. 심볼로지 확정 (1주차, 최우선)

실제 인천세관 송장의 바코드 종류를 확인한다 (Code128 / ITF-14 / GS1-128 등). 멘토 문의 또는 현장 사진으로 확인한다.

이것이 확정되지 않으면 데이터 필터링 기준, 합성 시 렌더링할 심볼로지, 평가 대상이 모두 미정이 되어 이후 작업이 무효화될 수 있다. 기획서에 1D/2D 타겟이 정의되지 않은 문제도 여기서 해결한다 — 문제 정의(스캔라인, 바 간격 비율, 명암비, 평탄화)가 전부 1D 논리이므로 **1D 우선, QR은 범위 외**로 확정하는 것을 권고한다. QR은 Reed-Solomon 오류정정이 내장되어 훼손 내구성과 복원 전략이 근본적으로 다르다.

결과를 `decisions/0001`에 기록한다.

### 1단계. 오픈소스 데이터 실태 파악 (1주차)

3종을 다운로드해 실제로 열어보고 아래를 표로 기록한다.

- 실제 장수
- 1D / QR 비율
- 라벨이 HBB인지 OBB인지
- 해상도 분포
- 정답 바코드 번호 유무

기획서는 "952장 JSON+JPG"로만 기술되어 있어 OBB 라벨 여부가 불확실하다. 이 표 없이 3단계로 진행하면 안 된다. 결과를 `decisions/0004`에 기록한다.

### 2단계. 디코더 베이스라인 측정 (1주차, 1단계와 병렬)

데이터 구축 단계의 최우선 작업이다. 기획서에는 평가 방법(4.6.2)으로만 기술되어 있으나 순서상 맨 앞에 와야 한다.

1. 기준선 없이는 "90% 달성"을 주장할 수 없다. 현재 성공률을 아무도 모른다.
2. 이 작업이 곧 평가 코드가 된다. 이후 모델 비교와 11월 최종 평가에 그대로 사용한다.
3. AI 모델 없이 SW파트가 즉시 착수할 수 있다.

결과가 예상과 다를 수 있다. 예상보다 잘 나오면 목표를 재설정해야 하고, 못 나오면 그것이 프로젝트의 근거가 된다. 어느 쪽이든 먼저 알아야 한다.

**디코더를 3개 나란히 측정한다.** 기획서는 pyzbar를 전제하지만, §15에서 탐지·복원 모델을 후보 비교로 정하기로 한 것과 같은 논리가 디코더에도 적용된다.

| 후보 | 비고 |
|---|---|
| **pyzbar** (ZBar 바인딩) | 기획서 안. 오래된 라이브러리이고 회전·왜곡에 특히 약하다고 보고된다 |
| **zxing-cpp** | ZXing의 C++ 재작성판. 파이썬 바인딩 제공 |
| **OpenCV `barcode` 모듈** | §17에서 확인한 Nature 논문이 실제로 사용한 것. 그 논문의 디코딩 정확도는 양호 조명 98.5% |

**지금 하는 이유가 중요하다.** 디코더만 바꿔서 성공률이 크게 오르면, 기획서 4.6.2가 측정하려는 "AI 복원이 기여한 순수 향상폭"이 실제보다 부풀려진다. 극단적으로는 **AI 복원 없이 디코더 교체만으로 목표를 달성**할 수도 있는데, 그렇다면 프로젝트 전제 자체를 다시 봐야 한다. 11월 심사에서 "왜 pyzbar만 썼나"라는 질문에 답할 근거도 필요하다.

비용은 거의 없다. 평가 코드는 같고 디코더 호출부만 바꾼다. 세 개를 재는 데 하루면 충분하다.

측정 조건은 기획서 4.6.2의 3조건 비교를 디코더 축으로 확장한 것이다.

```
① 디코더 단독                            (후보 3개 각각)
② 전통 영상처리(CLAHE + 적응형 이진화) + 디코더   (후보 3개 각각)
③ 전통 영상처리 탐지 → ②
④ (9월 4주차 이후) AI 파이프라인 → 최적 디코더
```

결과는 `decisions/0006-디코더-선정.md`에 기록한다. **여기서 정한 디코더를 이후 모든 단계에 고정한다.** 그래야 AI 복원의 기여도만 분리해서 측정할 수 있다.

`sw/decoding.py`는 디코더를 교체할 수 있게 작성한다. 세 후보를 비교하려면 어차피 필요하고, 최종 선택 후에도 재현 실험에 쓴다.

### 3단계. 라벨링 (2주차)

1단계 결과로 OBB 재라벨링 필요 여부를 판단한다. HBB→OBB는 자동 변환되지 않는다. 필요하면 1~2학년 4명에게 분배한다 — 952장 ÷ 4 = **1인당 238장**. 기획서 5.1에 이 공수가 반영되어 있지 않았다.

**도구.** `labelImg`는 회전 박스(OBB)를 지원하지 않으므로 쓸 수 없다. 아래에서 고른다.

| 도구 | 비고 |
|---|---|
| **CVAT** | 회전 박스 지원, 무료 자체 호스팅 또는 무료 클라우드. 여러 명 동시 작업과 검수 기능이 있다 |
| **Roboflow** | 사용이 가장 쉽고 YOLO-OBB 형식으로 바로 내보내진다. 무료 플랜은 용량 제한이 있다 |

1~2학년 4명이 동시에 작업하고 검수까지 필요하므로 **CVAT를 권한다.** 선정 결과는 `decisions/0004`에 기록한다.

**품질 관리.** 라벨이 부정확하면 mAP 목표 달성이 불가능하다. 라벨링을 시작하기 전에 아래를 준비한다.

1. **라벨링 가이드라인 1장** — 바코드의 어디까지를 박스에 넣는지(여백 포함 여부), 부분적으로 가려진 바코드는 어떻게 하는지, 각도 기준선을 어디로 두는지. 이 문서 없이 4명이 각자 판단하면 라벨이 일관되지 않는다
2. **공동 작업 20장** — 처음 20장은 4명이 같은 이미지를 라벨링하고 결과를 비교한다. 기준이 어긋나는 지점을 여기서 잡는다
3. **교차 검수 10%** — 각자 작업분의 10%를 다른 사람이 확인한다. 3학년이 최종 표본 검수
4. **검수 결과를 `decisions/0004`에 기록** — 불일치율을 남겨두면 나중에 mAP가 낮게 나올 때 원인이 모델인지 라벨인지 판단할 수 있다

### 4단계. 합성 파이프라인 (2~3주차)

**기획서 4.2.1을 정정한다.** 기획서는 HF `barcodes-google-ocr`를 "역방향 합성의 원본 재료"로 지정했으나, 확인 결과 이 데이터는 416px 실사 촬영본이며 스키마가 `pixel_values` / `label`(1 class) / `ocr`(가변 길이 리스트)이다. 즉 정답 바코드 번호가 보장된 필드가 아니고 이미지도 이미 열화되어 있어 **깨끗한 GT가 아니다.** 열화된 정답에 훼손을 덧씌우면 복원 모델의 학습 목표 자체가 흐려진다.

대신 `python-barcode` 또는 `treepoem`으로 확정된 심볼로지를 직접 렌더링해 픽셀 완벽한 원본을 생성한다. HF 데이터는 원본 재료가 아니라 **탐지 모델 학습용**으로 사용한다.

증강은 기획서 4.2.2대로 Albumentations의 `GridDistortion` / `ElasticTransform`(비닐 주름)과 인위적 광원 효과(반사)를 적용한다.

**또한 기획서 4.1.1의 서술을 정정한다.** "모델 학습에 최적화되도록 명암비 조절 및 기하학적 평탄화 전처리를 수행한다"는 문장에서 기하학적 평탄화는 Stage 2가 학습해서 출력해야 하는 대상이다. 전처리로 미리 평탄화하면 모델이 학습할 것이 없어진다. GT 정규화를 의도한 것으로 보이므로 문서를 수정한다.

### 5단계. 실촬영 테스트셋 (3~4주차) — 순서 필수

기획서 4.6.2는 "출력 텍스트가 정답 번호와 완전히 일치"로 판정한다고 했으나 정답 번호 확보 절차가 없다. 아래 순서를 반드시 지킨다.

```
① 송장을 평탄한 상태로 촬영  →  디코더로 정상 판독  →  정답 번호 확보·기록
② 같은 송장을 구기고 비닐 씌우고 반사 유도  →  재촬영
③ 두 이미지를 같은 ID로 짝지어 저장
```

순서를 바꾸면 구겨진 사진의 정답을 확인할 방법이 없어 테스트셋이 무효가 된다.

①의 자동화 코드는 `wemeet/data/ground_truth.py`에 둔다. 이 파일이 `sw.decoding`을 부르는 유일한 데이터파트 코드다 (§4).

이 단계에서 개인정보 영역을 물리적으로 차폐한다.

### `dev` / `eval` 두 분할로 촬영한다

기획서 4.2.1은 실촬영 데이터를 "학습에 일절 관여시키지 않는 독립 평가 데이터"로 규정한다. 그런데 §22의 10월 중간 평가가 이 원칙과 충돌한다 — 최종 평가셋을 10월에 미리 보면 그 결과를 보고 11월까지 튜닝하게 되고, 그 순간 독립성이 깨진다.

따라서 **촬영 시점부터 두 묶음으로 나눈다.**

| 분할 | 용도 | 열어보는 시점 |
|---|---|---|
| `dev` | 중간 점검, 도메인 갭 측정, 디버깅 | 언제든 |
| `eval` | **최종 평가 전용** | 11월 2주차에 딱 한 번 |

비율은 `dev` 40% / `eval` 60%를 권한다. 같은 송장을 두 분할에 나눠 넣지 않는다 — 같은 송장의 다른 각도 사진이 양쪽에 있으면 사실상 유출이다. **송장 단위로 나눈다.**

§18의 도메인 갭 대응에서 "실촬영 데이터 일부를 학습에 투입"하는 경우에도 `dev`만 쓴다. `eval`은 어떤 경우에도 학습·튜닝에 노출되지 않는다.

### 데이터 구축 단계 작업 순서 (착수 = 2026년 9월)

```
1주차 ├─ 0단계  심볼로지 확정                          전원
      ├─ 1단계  오픈소스 데이터 실태 파악              데이터파트
      └─ 2단계  디코더 3종 베이스라인 측정             SW파트
               + 전통 영상처리 탐지 시도               AI파트

2주차 ├─ 3단계  OBB 재라벨링 (1인 238장)               1~2학년 4명
      └─ 4단계  합성 파이프라인                        데이터파트
               + 탐지 모델 학습 시작                   AI파트

3~4주차├─ 5단계 실촬영 테스트셋 (GT 프로토콜 준수)      전원
      └─ 탐지 모델 비교·선정 → decisions/0002 확정     AI파트
```

---

## 15. 모델 선정 절차

### 원칙

후보를 정해놓고 코드를 쓰지 않는다. **인터페이스를 고정하고 후보를 갈아끼워 비교한다.** `detect()` 시그니처가 고정되어 있으면 모델 교체는 파일 하나 수정이며 `pipeline.py` / `schemas.py` / 테스트는 건드리지 않는다. 따라서 모델이 미정인 현재 상태에서도 SW파트는 작업을 시작할 수 있다.

### Stage 1 — 탐지 후보

| 후보 | 라이선스 | 장점 | 단점 |
|---|---|---|---|
| Ultralytics YOLO-OBB (기획서 안) | **AGPL-3.0** | 문서·예제 풍부, OBB 기본 지원 | 라이선스가 상용화와 충돌 |
| YOLOX / MMRotate | Apache-2.0 | 라이선스 자유 | 설치·학습 난이도 높음 |
| 전통 영상처리 (MSER + 최소외접사각형) | 제약 없음 | 학습 불필요, 즉시 동작 | 배경이 복잡하면 취약 |

**진행 순서**: 1주차에 3번(전통 영상처리)을 먼저 구현한다. 학습이 필요 없어 며칠이면 되고, 베이스라인과 실제 Stage 1 구현을 동시에 확보한다. 이후 1번으로 학습해 비교하고, 라이선스가 문제가 되면 2번으로 전환한다.

3번이 충분한 성능을 낸다면 딥러닝 모델을 쓰지 않는다. "이 단계는 딥러닝이 필요 없었다"는 결과도 보고서에 기술할 가치가 있다.

### Stage 2 — 복원 후보

| 후보 | 장점 | 단점 |
|---|---|---|
| U-Net (기획서 안) | 단순, 학습 빠름, 스킵 커넥션이 바 경계 보존에 유리 | 심한 훼손에 한계 |
| Pix2Pix (조건부 GAN) | 결과가 선명함 | 학습 불안정, 소요 시간 김 |
| Restormer / NAFNet | 성능 우수 | 무거워 0.5초 예산 초과 위험 |

**진행 순서**: U-Net부터. 복원 단계에서 SSIM이 목표에 미달하면 Pix2Pix를 실험한다. 3번은 예산 제약으로 후순위.

손실 함수는 기획서 4.3.2대로 L1 + Edge/Gradient + SSIM 가중 결합을 사용한다.

### Stage 3 — 디코더 후보

§14 2단계에서 pyzbar / zxing-cpp / OpenCV `barcode` 세 개를 측정하고 하나로 고정한다. 상세는 §14 2단계, 기록은 `decisions/0006`.

`sw/decoding.py`는 디코더를 갈아끼울 수 있게 작성한다. 비교하려면 어차피 필요하고, 최종 선정 후에도 재현 실험에 쓴다.

### 재시도 파라미터 탐색 순서

기획서 4.3.3은 "사전 정의된 탐색 순서에 따라 변경하며 최대 3회까지 재시도"라고만 하고 그 순서를 정의하지 않았다. 순서가 성능과 지연을 모두 좌우하므로 아래를 초기값으로 확정한다.

| 시도 | 변경하는 것 | 노리는 실패 원인 |
|---|---|---|
| 0차 (최초) | CLAHE + 적응형 이진화 (기본값) | — |
| 1차 재시도 | **이진화 임계값을 낮춤** | 반사로 밝아진 영역에서 바가 사라진 경우 |
| 2차 재시도 | **이진화 임계값을 높임** | 그림자로 어두워진 영역에서 흰 여백이 사라진 경우 |
| 3차 재시도 | **모폴로지 커널 크기 확대 + 대비 강화** | 주름으로 바가 끊어진 경우 |

임계값을 먼저 낮추는 이유: 이 프로젝트의 1순위 훼손 유형이 **비닐 반사**이며, 반사는 밝아지는 방향의 손실이다. 가장 흔한 원인부터 시도하면 평균 재시도 횟수가 줄고 지연 예산에 유리하다.

**재시도할 때 복원 모델을 다시 돌리지 않는다.** 이진화와 디코딩만 다시 한다. 기획서 4.3.3도 "학습된 모델의 출력 이후 단계에서 작동하는 규칙 기반 보완 장치"로 규정한다. 복원을 재시도마다 반복하면 250ms가 4번 들어가 500ms 예산이 즉시 무너진다.

이 순서는 초기값이며, §17의 "피드백 루프 기여도 분석"(재시도 횟수별 추가 성공률)으로 검증해 조정한다. 조정 결과는 `decisions/0006`에 함께 기록한다.

### 비교 기준

기획서 4.6.1은 정확도 지표만 규정한다. 두 항목을 추가한다.

| 기준 | 근거 |
|---|---|
| 정확도 (mAP / SSIM) | 기획서에 있음 |
| **추론 시간 (ms)** | 0.5초 예산 충족 여부. 정확해도 느리면 탈락 |
| **라이선스** | AGPL이면 상용화 목표와 충돌 |

동일 데이터·동일 지표·동일 표로 비교하고 `docs/decisions/`에 기록한다.

---

## 16. 실험 기록 — Weights & Biases

기획서 4.6.3은 "모델 버전별 성능 이력 관리"를 요구하나 4.4의 도구 목록에는 Git/GitHub만 있다.

W&B를 사용한다. 학생 무료이며 학습 코드에 3줄을 추가하면 정확도·손실 그래프가 자동으로 누적되고 8명이 같은 대시보드를 본다.

```python
import wandb
wandb.init(project="wemeet-barcode", name="detection-yolo-obb-v1", config=cfg)
wandb.log({"mAP50": 0.91, "mAP50-95": 0.62, "inference_ms": 38.4})
```

- 프로젝트명: `wemeet-barcode`
- 실행 이름 규칙: `<stage>-<model>-v<n>` (예: `restoration-unet-v3`)
- `wandb/`는 `.gitignore`에 포함
- API 키는 각자 로컬에 두고 커밋하지 않는다

---

## 17. 기획서 대비 정정 사항

이 스펙에 반영한 계획서 수정 항목이다. 계획서 본문도 함께 고쳐야 한다.

### 문서만 수정하면 되는 것

| 위치 | 현재 | 수정 |
|---|---|---|
| p.21 참고문헌 ③ | "Robust Barcode Recognition under Reflection and Packaging Distortion" | 해당 URL(DOI `10.3390/app9163268`)의 실제 논문은 **"1D Barcode Detection via Integrated Deep-Learning and Geometric Approach"**. 기재된 제목은 검색으로 확인되지 않음 |
| p.20 참고문헌 ① | "...using drone imagery simulation" | **"...using simulated UAV imagery"** (논문 자체는 실재) |
| p.11 표 3행 | 출처 "Kaggle" | **"Hugging Face"** (본문 4.2.1은 올바름) |
| p.11 | "흑백 명암비비(Contrast)" | "흑백 명암비(Contrast)" |
| p.2 목차 / p.20 본문 | "7. 관련 연구 및 문헌 조사" / "7 관련 연구 및 선행 조사" | 제목 통일 |
| p.12 | 내용 없음 | 그림 자리인지 확인 |

참고문헌 ①은 인용을 강화할 수 있다. 해당 논문의 실측치는 mAP@0.5 **92.4%**, mAP@0.5:0.95 64.5%, 디코딩 98.5%(양호 조명)/78.3%(부분 가림)이며, 저자들이 *"severely damaged or reflective barcodes were not explicitly tested"*라고 명시한다. 기획서 7.1(4)의 한계 주장을 저자 인용으로 뒷받침할 수 있다.

### 평가 설계 — 발견된 문제와 제안 (미확정)

> **이 항목은 이 스펙에서 확정하지 않는다.** 기획서의 평가 지표를 바꾸는 내용이라 팀 논의와 멘토·지도교수 확인이 필요하다. 아래는 검토에서 발견한 문제와 제안이며, 확정은 `docs/decisions/0005-평가-지표-체계.md`에서 한다.
>
> 다만 첫 번째 항목(SSIM)은 의견 문제가 아니라 물리적 제약이므로, 어떤 결론으로 가든 지표 표에 측정 대상 데이터셋을 명시할 필요는 남는다.

**문제 1. SSIM은 실촬영 테스트셋에서 측정할 수 없다.** 4.6.1은 SSIM을 "복원 이미지와 원본(GT) 간" 비교로 정의하고 4.6.2는 "실촬영 데이터만으로 최종 평가"라고 하는데, 실제로 구겨진 비닐 사진에는 픽셀 정렬된 평탄한 정답 이미지가 물리적으로 존재하지 않는다. 같은 송장을 펴서 재촬영해도 픽셀이 정렬되지 않는다.

제안: 지표 표에 "측정 대상 데이터셋" 열을 추가해 분리한다.

| 지표 | 측정 대상 | 목표 |
|---|---|---|
| mAP@0.5, mAP@0.5:0.95, 각도 오차(°) | 합성 검증셋 | 2주차 확정 (§20) |
| SSIM (주), PSNR (보조) | **합성 검증셋 전용** | SSIM 0.85 |
| 종합 디코딩률 | 실촬영 테스트셋 | 2주차 확정 (§20) |
| 난인식 구제율 | 실촬영 테스트셋의 난인식 부분집합 | 2주차 확정 (§20) |
| **오독률 (false decode)** | 실촬영 테스트셋 | **0%** |
| E2E 지연 (p50 / p95) | 실촬영 테스트셋 | 500ms |

**문제 2. 최종 KPI의 분모 정의.** 4.6.1은 분모를 "pyzbar 단독 실패 샘플"로 정의하는데, 이 정의로는 4.6.2의 3조건 비교에서 조건(1) pyzbar 단독이 정의상 0%가 되어 비교가 성립하지 않는다. 제안: 지표를 셋으로 분리한다.

1. 전체 테스트셋 기준 **종합 디코딩률**
2. 난인식 부분집합 **구제율(recovery rate)**
3. **오독률** — 0%여야 한다. 오독을 성공으로 집계하는 오류를 방지한다 (4.6.2의 완전 일치 원칙 유지)

**문제 3. mAP 0.95 목표의 근거.** 기획서의 0.95는 근거가 제시되지 않았다. 위 논문이 창고 환경 YOLOv8로 mAP@0.5 92.4%다. 또한 OBB는 각도 오차가 크롭 품질을 직접 좌우하므로 mAP@0.5만으로는 불충분하다 — 각도가 틀려도 IoU 0.5는 통과한다. **mAP@0.5:0.95와 각도 오차(°)를 함께 사용**하고, 구체적 목표는 §14 1단계 실태 파악 후 확정한다.

### 누락 사항 반영

| 항목 | 반영 위치 |
|---|---|
| 개인별 역할 매핑 없음 | §6 (1학년 배치), README 파트 표 |
| 리스크 대응 계획 없음 | §18 (합성↔실측 도메인 갭) |
| 라이선스 검토 없음 | §15 비교 기준, §20 |
| 개인정보 정책 없음 | §7 |
| 실험 관리 도구 없음 | §16 |
| 하드웨어 전제 없음 | §20 |
| OBB 재라벨링 공수 | §14 3단계 |

---

## 18. 리스크

기획서에 리스크 관리 섹션이 없다. 최대 리스크와 대응을 명시한다.

**합성 데이터와 실제 훼손 사이의 도메인 갭.** 기획서는 "최종 테스트로 측정"까지만 규정하고 갭이 클 경우의 대응이 없다.

대응 순서를 정한다.

1. §14 5단계 실촬영 테스트셋으로 갭을 정량 측정한다 (합성 검증 성능 − 실촬영 성능)
2. 갭이 크면 실촬영 데이터 일부를 학습에 투입하되, **평가 전용 분할은 절대 학습에 넣지 않는다** (촬영 시점부터 학습용/평가용을 분리 수집)
3. 그래도 부족하면 합성 파라미터를 실측 분포에 맞춰 재조정한다 (주름 강도, 반사 면적 비율)
4. 최종적으로 시스템의 정상 작동 보장 범위를 명시적으로 문서화한다 (기획서 4.2.3의 "픽셀 손실률 40% 이하" 기준을 실측으로 검증·조정)

---

## 19. 1단계 산출물

```
.github/workflows/ci.yml
.github/pull_request_template.md
.github/ISSUE_TEMPLATE/task.md

wemeet/__init__.py
wemeet/schemas.py                 ← 내용 있음 (§5). PipelineResult 포함
wemeet/ai/__init__.py
wemeet/sw/__init__.py
wemeet/data/__init__.py

tests/conftest.py                 ← 내용 있음 (§11). 바코드 이미지 생성 픽스처
tests/test_schemas.py             ← 내용 있음 (§11)

docs/architecture.md              ← 구조 + 의존 규칙 + 시간 예산
docs/parts/ai.md                  ← 작성 완료 (AI파트 할 일)
docs/parts/sw.md                  ← 작성 완료 (SW파트 할 일)
docs/parts/data.md                ← 작성 완료 (데이터파트 할 일)
docs/decisions/0001-바코드-심볼로지-범위.md
docs/decisions/0002-탐지-모델-선정.md
docs/decisions/0003-복원-모델-선정.md
docs/decisions/0004-데이터-수집-및-GT-프로토콜.md
docs/decisions/0005-평가-지표-체계.md
docs/decisions/0006-디코더-선정.md
docs/external/README.md           ← 작성 완료 (코드 밖 작업 전체 체크리스트)
docs/external/huggingface.md      ← 작성 완료
docs/external/wandb.md            ← 작성 완료
docs/reports/.gitkeep             ← 성과 보고서 자리 (PDF 예외 경로)

downloads/.gitkeep               ← .gitignore 에 !downloads/.gitkeep 예외 필요
.gitignore
pyproject.toml                    ← uv + ruff + import-linter 설정
README.md                         ← 작성 완료 (개요 + 시작하기 + 파트별 링크)
CONTRIBUTING.md                   ← 브랜치·커밋·PR 규칙
```

`detection.py` / `restoration.py` / `decoding.py` / `pipeline.py` / `server.py` / `train_detection.py` / `train_restoration.py` / `download.py` / `synthesis.py` / `ground_truth.py` 스텁과 `web/` 스캐폴드는 2단계에서 만든다.

### 검증 기준

1단계 완료 판정 조건이다.

1. 빈 상태에서 `git clone` → `uv sync` → `uv run pytest` 가 통과한다
2. `uv run lint-imports` 가 통과한다
3. 아래 세 가지를 일부러 넣으면 `lint-imports` 가 **실패한다** — 규칙이 적혀만 있는지 실제로 작동하는지를 가른다
   - `wemeet/ai/` 에 `import wemeet.sw`
   - `wemeet/data/` 에 `import wemeet.sw.pipeline`
   - `wemeet/schemas.py` 에 `import wemeet.ai`
4. `wemeet/data/` 에 `import wemeet.sw.decoding` 은 **통과한다** (허용된 예외가 막히지 않음을 확인)
5. `pytest` 가 `downloads/` 없이 통과한다 — 픽스처가 이미지를 코드로 만드는지 확인
6. `main`에 직접 push가 거부된다
7. 승인 없는 PR은 머지 버튼이 비활성화된다 (리뷰어 자동 지정은 검증 대상이 아니다 — §6 폐기)

4번이 특히 중요하다. 금지 규칙만 검증하면 실수로 너무 넓게 막아놓은 것을 놓친다. §14 5단계의 GT 확보 코드가 막히면 9월에 발견한다.

---

## 20. 이 스펙에서 확정하지 않는 항목

각각 담당과 기한을 명시해 미정 상태로 방치되지 않게 한다.

기한은 착수일(2026년 9월) 기준 상대 주차다.

| 항목 | 왜 지금 정하지 않는가 | 담당 / 기한 | 기록 위치 |
|---|---|---|---|
| **바코드 심볼로지** | 현장 확인이 필요하다. 이것이 막히면 이후 데이터 작업이 무효가 될 수 있어 최우선이다 | 팀장 / 1주차 | `decisions/0001` |
| **평가 지표 체계** | 기획서 4.6의 지표를 바꾸는 내용이라 멘토·지도교수 확인이 필요하다. §17에 문제와 제안만 정리했다 | 팀장 + 전원 / 2주차 | `decisions/0005` |
| **mAP·디코딩률 목표 수치** | 베이스라인 측정(§14 2단계) 전에는 근거가 없다 | AI·SW파트 / 2주차 | `decisions/0005` |
| **탐지·복원 모델** | §15 절차에 따라 실험으로 결정한다 | AI파트 / 4주차, 복원 단계 3주차 | `decisions/0002`, `0003` |
| **LICENSE 내용** | AGPL 모델(Ultralytics)을 쓸지가 `decisions/0002`에 달려 있다. 모델 선정 전에는 결정할 수 없다 | 팀장 / 0002 확정 직후 | `LICENSE` |
| **추론 대상 하드웨어** | 0.5초 목표의 기준 장비가 기획서에 없다. 멘토 자문 필요 | 팀장 / 통합 단계 전 | `docs/architecture.md` |
| **학습용 GPU 확보 경로** | 연구실 서버 / 학교 클러스터 / Colab 중 무엇을 쓸지 미정 | AI파트 / 1주차 | `docs/external/README.md` |
| **디코더** | pyzbar / zxing-cpp / OpenCV 3종을 실측 비교해 정한다 (§14 2단계) | SW파트 / 1주차 | `decisions/0006` |
| **라벨링 도구** | CVAT / Roboflow. 1단계 실태 파악 결과에 따라 재라벨링 규모가 정해진다 | 데이터파트 / 1주차 | `decisions/0004` |
| **재시도 파라미터 순서** | §15에 초기값을 정했으나 피드백 루프 기여도 분석으로 조정한다 | SW파트 / 10월 4주차 | `decisions/0006` |

---

## 21. 확정 사항 요약

| 항목 | 결정 |
|---|---|
| 레포 구조 | 파트 기준 단일 패키지, `src/` 없음, 단계별 파일 하나 |
| 계약 위치 | `wemeet/schemas.py` (최상위, 아무것도 import 안 함) |
| 의존 방향 | `data → sw → ai → schemas` 4층. `data → sw.decoding` 만 예외 허용 |
| 파이프라인 소유 | SW파트 (`wemeet/sw/pipeline.py`) |
| 최종 반환형 | `PipelineResult` — 원본·탐지·복원 이미지 + 디코딩 결과 (§5) |
| 실패 처리 | 예외를 던지지 않고 값으로 반환. 실패에는 항상 `failure_reason` |
| 탐지 다중 검출 | confidence 최상위 1개만 처리, 총 개수는 기록 |
| API 계약 | `POST /api/decode`, 판독 실패도 HTTP 200, 이미지는 base64 (§23) |
| 프론트엔드 | Vite + TypeScript + npm, Node 22 LTS, UI 라이브러리 없음 (§24) |
| 디코더 | pyzbar / zxing-cpp / OpenCV 3종 비교 후 고정 (§14 2단계) |
| 오너십 | 강제 장치 없음. README 파트 표 + `CONTRIBUTING.md` 규칙 (§6 폐기, 2026-08-04) |
| 데이터·가중치 | HF Hub. Git엔 스크립트만. 실촬영은 private |
| 개발 환경 | uv (`uv sync`) |
| CI | import-linter + ruff + pytest + tsc --noEmit + npm run build |
| 브랜치 | `<파트>/<내용>`, main 보호, PR 승인 1명 |
| 커밋 | `[파트] 무엇을 했다` |
| 실험 기록 | Weights & Biases |
| 시간 예산 | 탐지 100 / 복원 250 / 디코딩 50 / 재시도 100 = 500ms |
| 일정 | 선택지 B — 11월 종료 유지, 뒤 단계 압축 (§22) |
| 범위 축소 | 경량화 조건부 / GUI 최소 기능 / QR 조건부 제외 / Stage 1 조기 확정 규칙 |
| 실촬영 분할 | `dev` 40% (중간 점검) / `eval` 60% (11월 2주차에 한 번만) |
| 라벨링 | CVAT 권장. 가이드라인 + 공동 20장 + 교차 검수 10% |

---

## 22. 일정 — 선택지 B 확정 (11월 종료 유지)

사업단 제출 기한이 **11월**로 확인되었다. 데이터 구축 착수는 9월이므로 뒤 단계를 압축한다.

### 확정 일정

| 월 | 작업 | 산출물 |
|---|---|---|
| **8월** | 레포 구조(1단계) + 골격(2단계) | 실행되는 더미 파이프라인, 협업 규칙 |
| **9월** | 데이터 구축 + 탐지 모델 | 데이터셋 v1, 탐지 모델 확정, 베이스라인 수치 |
| **10월** | 복원 모델 + 통합 + 피드백 디코딩 | 복원 모델, 통합 파이프라인, 재시도 루프 |
| **11월** | GUI + 최종 평가 + 보고서 | 시연 프로그램, 평가 보고서, 발표 자료 |

기획서 5.1은 7월에 설계, 8월에 데이터를 계획했다. 설계가 7월 말로 밀린 대신 **8월이 레포 구축에 온전히 쓸 수 있는 달로 확보되었다.** 9월 1주차에 8명이 곧바로 병렬 착수할 수 있도록 8월 안에 골격까지 끝낸다.

### 월별 상세

**8월 — 레포 구축**

1단계(구조와 규칙)와 2단계(골격)를 완료한다.

§14의 데이터 작업을 8월로 앞당기는 방안은 검토했으나 채택하지 않았다. 8월은 레포 구축에만 쓰고, 데이터는 9월 1주차에 착수한다.

**9월 — 데이터 + 탐지** (§14 상세)

| 주차 | 작업 |
|---|---|
| 1주차 | 심볼로지 확정 / 데이터 실태 파악 / 디코더 3종 베이스라인 / 전통 영상처리 탐지 |
| 2주차 | OBB 재라벨링 / 합성 파이프라인 / 평가 지표 체계 확정(`0005`) |
| 3~4주차 | 실촬영 테스트셋 / 탐지 모델 비교·선정(`0002`) |

**10월 — 복원 + 통합 + 피드백**

| 주차 | 작업 |
|---|---|
| 1~2주차 | U-Net 복원 모델 학습 |
| 2~3주차 | 파이프라인 통합 — 더미를 실물로 교체 |
| 3주차 | 동적 피드백 루프 구현 |
| 4주차 | E2E 지연 측정, 500ms 예산 검증 → 경량화 필요 여부 판정 |

통합이 이 달에 들어가지만, 8월 골격 덕분에 "통합"이라는 별도 작업이 없다. `detection.py` / `restoration.py`의 더미를 실물로 바꿔 넣는 것이 전부이고 `pipeline.py`는 건드리지 않는다. 이것이 압축을 감당하게 해주는 핵심 장치다.

**11월 — GUI + 평가 + 보고서**

| 주차 | 작업 |
|---|---|
| 1주차 | GUI 최소 기능 / (필요 시) 경량화 |
| 2주차 | 최종 평가 — 실촬영 테스트셋 |
| 3주차 | 보고서 + 발표 자료 |
| 4주차 | 예비 — 초과분 흡수 |

### 범위 축소 — 확정 사항

압축을 감당하기 위해 아래를 확정한다.

**1. ONNX 변환·경량화를 선택 항목으로 강등** (기획서 4.3.4)

10월 4주차에 E2E 지연을 측정하고, **500ms를 이미 만족하면 수행하지 않는다.** 기획서의 목표는 0.5초이며 그것이 충족되면 경량화는 목적을 이미 달성한 것이다. 미달 시에만 11월 1주차에 수행한다.

판정 기준: 실촬영 테스트셋의 p95 지연이 500ms 이하이면 생략.

**2. GUI를 최소 기능으로 한정** (기획서 2.2의 3번)

포함하는 것 — 기획서가 명시한 항목이다.

- 이미지 로드
- 탐지 영역(Bounding Box) 표시
- 원본과 복원 결과 대조 표시
- 디코딩 텍스트와 처리 시간(ms) 표시

제외하는 것.

- 실시간 컨베이어 모니터링 화면
- 처리 이력 조회·통계 대시보드
- 사용자 관리, 설정 화면

기획서 2.2가 요구하는 것은 "현장 도입 시뮬레이션을 위한 데모"이며 위 4개로 충족된다.

**3. QR 지원 제외 — 조건부**

§14 0단계에서 실제 송장 심볼로지를 확인하고, 1D라면 QR을 범위 외로 확정한다 (`decisions/0001`). QR은 Reed-Solomon 오류정정이 내장되어 복원 전략이 근본적으로 다르므로, 한 모델로 둘 다 다루면 양쪽 성능이 모두 애매해진다.

현장이 실제로 QR을 사용한다면 제외할 수 없다. 그 경우 범위 축소는 4번에서 더 확보해야 한다.

**4. Stage 1 조기 확정 규칙**

§15에서 전통 영상처리(MSER + 최소외접사각형)를 먼저 구현한다. 아래 조건을 만족하면 **YOLO 학습 단계를 건너뛰고 전통 영상처리로 확정한다.**

- 실촬영 테스트셋에서 탐지 누락률 5% 이하
- 크롭 각도 오차 5° 이하
- 추론 시간 100ms 이하

시간이 가장 크게 절약되는 항목이다. 학습·튜닝·재라벨링이 모두 사라진다. 조건 미달 시에만 YOLO-OBB 학습으로 진행한다.

판정 시점: 9월 3주차. 판정 결과를 `decisions/0002`에 기록한다.

### 11월 과부하 완화 장치

압축의 최대 위험은 11월에 평가와 보고서가 겹치는 것이다. 두 가지로 완화한다.

**보고서를 11월에 몰아 쓰지 않는다.** `docs/decisions/` 문서를 매 결정 시점에 채워두면 "왜 이 모델을 골랐나", "왜 이 지표인가"에 대한 근거가 이미 문서로 존재한다. 11월 3주차의 작업은 새로 쓰는 것이 아니라 **누적된 결정 기록을 엮는 것**이 된다.

**평가 코드를 9월에 완성한다.** §14 2단계의 디코더 베이스라인 측정 코드가 그대로 최종 평가 코드가 된다. 11월에는 데이터만 바꿔 실행한다. 평가 도구를 11월에 만들기 시작하면 압축을 감당할 수 없다.

### 남은 위험

11월 2주차 최종 평가에서 목표 미달이 나오면 개선할 시간이 4주차 예비 1주뿐이다. 이를 줄이려면 **10월 4주차에 `dev` 분할로 중간 평가를 한 번 수행한다.** 조기 경보용이며, 이 시점에 목표와의 격차를 알면 11월 1주차 작업 우선순위를 조정할 수 있다.

**`eval` 분할은 열지 않는다.** 최종 평가셋을 미리 보면 그 결과를 보고 튜닝하게 되어 독립성이 깨진다. 그래서 §14 5단계에서 실촬영 데이터를 `dev` / `eval` 두 묶음으로 나눠 촬영한다.

---

## 23. API 계약 — FastAPI ↔ React

`schemas.py`로 AI와 SW 사이 계약을 못 박았다. 같은 문제가 백엔드와 프론트엔드 사이에서 재발하지 않도록 여기서도 계약을 고정한다. 기획서 4.5의 SW 멘토링 항목에도 "React 프론트엔드와 백엔드 간 실시간 데이터 통신 및 API 설계"가 있다.

### 엔드포인트

| 메서드 | 경로 | 용도 |
|---|---|---|
| `POST` | `/api/decode` | 이미지 한 장을 처리하고 결과를 반환 |
| `GET` | `/api/health` | 서버 상태와 로드된 모델 버전 |

두 개로 시작한다. 기획서 2.2의 데모 요구사항(이미지 로드 → 대조 표시 → 텍스트·시간 표시)은 이것으로 충족된다.

### `POST /api/decode`

요청은 `multipart/form-data`로 파일 하나를 보낸다.

```
Content-Type: multipart/form-data
file: <이미지 파일>
```

응답은 JSON이다. `PipelineResult`(§5)를 그대로 직렬화한 형태다.

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

실패 시에도 **HTTP 200을 반환한다.**

```json
{
  "ok": false,
  "text": null,
  "symbology": null,
  "retry_count": 3,
  "failure_reason": "decode_failed",
  "candidate_count": 1,
  "degraded": false,
  "stage_ms": { "detect": 40.0, "restore": 205.0, "decode": 61.0 },
  "total_ms": 306.0,
  "images": {
    "original": "data:image/png;base64,...",
    "detected_box": [[120, 340], [520, 350], [518, 430], [118, 420]],
    "restored": "data:image/png;base64,..."
  }
}
```

### 왜 판독 실패에 200을 쓰는가

**판독 실패는 서버 오류가 아니다.** 이 시스템의 정상적인 결과 중 하나다. 4xx/5xx를 쓰면 프론트엔드가 `catch` 블록에서 처리하게 되고, 그러면 실패한 경우에 **복원 이미지를 화면에 못 띄운다.**

그런데 이 프로젝트에서 가장 중요한 화면이 바로 그 경우다. 작업자는 "AI가 실패했다"만 보는 게 아니라 "AI가 여기까지 복원했는데 안 읽혔다"를 보고 즉시 수동 대처를 판단한다. 기획서 7.2가 Scandit과의 차별점으로 내세운 지점이다.

4xx/5xx는 진짜 오류에만 쓴다.

| 상태 | 상황 |
|---|---|
| `400` | 파일이 없음, 이미지로 해석 불가 (`failure_reason="invalid_input"`) |
| `413` | 파일 크기 초과 |
| `500` | 서버 내부 오류 (모델 로드 실패 등) |

### 이미지 전달 방식

base64 data URI로 JSON에 실어 보낸다. 별도 파일 서버나 임시 URL을 만들지 않는다.

- 요청 하나에 응답 하나로 끝나 프론트엔드 코드가 단순하다
- 서버에 이미지를 저장하지 않으므로 §7의 개인정보 정책과 충돌하지 않는다 — **송장 이미지가 디스크에 남지 않는다**
- 크롭된 바코드 이미지는 작으므로 크기 부담이 적다

원본은 클 수 있으니 서버에서 긴 변 1280px로 줄여 보낸다. 화면 대조용이므로 원해상도가 필요 없다.

`detected_box`는 원본 좌표계의 네 점이다. React가 원본 위에 그대로 그린다. OBB이므로 사각형 네 점이며, 회전을 표현할 수 있다.

### 계약 변경 규칙

이 응답 형식을 바꾸려면 `schemas.py`와 같은 취급을 한다 — **`server.py`와 `web/src/api.ts`를 같은 PR에서 함께 고친다.** 한쪽만 바꾸면 화면이 조용히 깨진다.

`PipelineResult`가 바뀌면 이 응답도 바뀌므로, `schemas.py` 변경 PR에는 API 응답 확인이 포함된다.

---

## 24. 프론트엔드 환경

파이썬은 `uv sync` 한 줄로 정리했다. 프론트엔드도 같은 수준으로 단순화한다.

| 항목 | 선택 | 이유 |
|---|---|---|
| 빌드 도구 | **Vite** | 설정 파일 하나, 시작이 빠르다. CRA는 유지보수가 중단됐다 |
| 언어 | **TypeScript** | API 응답 형식을 타입으로 못 박을 수 있다. `schemas.py`와 같은 효과 |
| 패키지 매니저 | **npm** | Node에 기본 포함. yarn/pnpm은 설치 단계가 하나 더 늘어난다 |
| Node 버전 | **22 LTS** | `web/.nvmrc`에 고정 |
| UI 라이브러리 | **없음** | 화면이 하나뿐이다. CSS로 충분하다 |
| 상태 관리 | **없음** | `useState` 로 충분하다 |

팀원이 외울 명령어는 둘이다.

```
cd web && npm install      # 처음 한 번
cd web && npm run dev      # 개발 서버
```

### API 응답 타입

`web/src/api.ts`에 §23의 응답을 타입으로 적는다. 이게 프론트엔드 쪽 계약이다.

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
    original: string;              // data URI
    detected_box: [number, number][] | null;
    restored: string | null;       // data URI
  };
};
```

### CI에서 프론트엔드 검사

`.github/workflows/ci.yml`에 단계를 추가한다. 파이썬 검사와 병렬로 돈다.

| 검사 | 명령 |
|---|---|
| 타입 검사 | `npx tsc --noEmit` |
| 빌드 확인 | `npm run build` |

프론트엔드 테스트는 넣지 않는다. 화면이 하나이고 11월 일정이 압축된 상황에서 비용 대비 이익이 낮다. 타입 검사와 빌드 통과만으로 "머지했는데 화면이 안 뜨는" 사고는 막힌다.

### 2단계에서 만들 것

`web/` 스캐폴드는 2단계(골격)에서 만든다. 그때 §23의 응답을 반환하는 더미 엔드포인트와 함께 붙여서, **8월에 이미 화면이 뜨는 상태**를 만든다. AI 모델 없이도 원본·복원(더미)·텍스트가 화면에 나오면 프론트엔드 통합 위험이 8월에 소진된다.
