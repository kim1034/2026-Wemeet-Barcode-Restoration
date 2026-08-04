# 1단계 레포 골격 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `git clone` → `uv sync` → `uv run pytest` → `uv run lint-imports` 가 전부 통과하는 레포 골격을 만든다. 파트 간 import 규칙이 문서가 아니라 CI로 강제되는 상태까지 간다.

**Architecture:** 파이썬 단일 패키지 `wemeet/` 아래에 파트별 서브패키지(`ai`/`sw`/`data`)와 공용 계약(`schemas.py`)을 둔다. 계약 모듈은 아무것도 import하지 않는 잎(leaf)이고, 의존은 `data → sw → ai → schemas` 방향으로만 흐른다. 이 방향을 `import-linter` 계약으로 CI에서 강제한다.

**Tech Stack:** Python 3.12, uv (패키지·환경), hatchling (빌드), pytest, ruff, import-linter, numpy, pillow, python-barcode

**설계 근거:** `docs/superpowers/specs/2026-07-30-main-branch-structure-design.md` (이하 "설계 문서"). §4 의존 규칙, §5 계약, §8 의존성, §10 CI, §11 테스트, §19 산출물.

---

## 범위

### 만든다

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | 의존성, ruff 설정, import-linter 계약 |
| `.python-version` | 파이썬 버전 고정 (uv가 읽는다) |
| `uv.lock` | 8명 환경 동일화 |
| `wemeet/__init__.py` | 패키지 선언 (내용 없음) |
| `wemeet/schemas.py` | AI ↔ SW 계약. 이 계획의 핵심 산출물 |
| `wemeet/ai/__init__.py` | 패키지 선언 (내용 없음) |
| `wemeet/sw/__init__.py` | 패키지 선언 (내용 없음) |
| `wemeet/data/__init__.py` | 패키지 선언 (내용 없음) |
| `tests/conftest.py` | 바코드 이미지를 코드로 생성하는 픽스처 |
| `tests/test_schemas.py` | 계약이 잘못된 값을 거부하는지 검증 |
| `.github/workflows/ci.yml` | ruff + import-linter + pytest |
| `docs/architecture.md` | 구조 + 의존 규칙 + 시간 예산 |
| `docs/reports/.gitkeep` | 성과 보고서 자리 (PDF 예외 경로) |

### 만들지 않는다

`detection.py` / `restoration.py` / `decoding.py` / `pipeline.py` / `server.py` / `train_detection.py` / `train_restoration.py` / `download.py` / `synthesis.py` / `ground_truth.py` 스텁, `web/` 스캐폴드, 더미 E2E 파이프라인, E2E 스모크 테스트. **전부 2단계다.**

## Global Constraints

- Python `>=3.12`. `.python-version` 에 `3.12` 를 적는다
- `opencv-python>=4.8` — 하한을 반드시 건다. 4.8부터 `cv2.barcode`가 main 패키지에 포함된다 (설계 §8)
- 모델 라이브러리(`torch`, `ultralytics`)는 **넣지 않는다**. `docs/decisions/0002`, `0003` 확정 후 추가한다
- 테스트는 `downloads/` 아래 파일을 읽지 않는다. GitHub 러너에 그 디렉토리가 없으므로 반드시 코드로 이미지를 만든다 (설계 §11)
- 테스트 함수 이름은 한국어로 쓴다. 실패 출력만 보고 무엇이 깨졌는지 알 수 있게 한다 (설계 §11)
- 커밋 메시지는 `[파트] 무엇을 했다` 형식. 이 계획의 커밋은 전부 `[SW]` 또는 `[문서]`
- ruff `line-length = 100`. 한국어 주석·docstring이 88자를 자주 넘는다
- `wemeet/schemas.py` 는 `numpy`와 표준 라이브러리만 import한다. `wemeet` 내부 모듈은 하나도 import하지 않는다

## 이 계획에서 새로 정한 것

설계 문서에 답이 없어서 여기서 정한다. 뒤집고 싶으면 이 절을 고친다.

| 항목 | 결정 | 이유 |
|---|---|---|
| 파이썬 버전 | **3.12** | 9월에 torch·ultralytics를 추가할 때 휠이 가장 확실하게 있는 버전. 3.13은 일부 CV 패키지가 늦다 |
| 빌드 백엔드 | **hatchling** (packaged project) | `[build-system]` 이 없으면 uv가 프로젝트를 virtual project로 보고 설치하지 않는다. 그러면 `import wemeet` 이 실행 위치(cwd)에 의존하게 된다. 명시적으로 설치되는 편이 8명 팀에서 안전하다 |
| dev 의존성 선언 | **`[dependency-groups] dev`** (PEP 735) | `uv sync` 가 기본으로 설치한다. `--extra dev` 를 외울 필요가 없다 |
| `wandb` 위치 | **dev 그룹** (설계 §8은 런타임 의존성으로 분류했다) | 학습 실험 기록 도구이므로 파이프라인 런타임에 필요하지 않다. **단, uv는 dev 그룹을 기본 설치하므로 8명 전원이 여전히 받는다** — 설치 용량을 줄이려는 것이 아니라 "런타임이 아니다"를 선언하는 것이다. 일부러 이렇게 뒀다. 별도 그룹(`--group train`)으로 빼면 AI파트가 명령을 하나 더 외워야 하고, README가 약속한 "명령어 1개"가 깨진다 |
| ruff 규칙 집합 | `E`, `F`, `I`, `UP` | 스타일·미사용·import 정렬·구버전 문법만 본다. 네이밍(`N`)·docstring(`D`) 규칙은 한국어 테스트 이름과 싸우므로 넣지 않는다 |
| CI 잡 이름 | **`ci`** | 브랜치 보호의 status check 목록에 뜨는 이름이 잡 이름이다. `docs/external/README.md` §1이 "`ci` 선택"이라고 적어뒀으므로 그 이름을 쓴다 |

## 설계 문서와 달라지는 3가지 (중요)

구현 순서 때문에 설계 문서를 그대로 따를 수 없는 지점이다. **넘어가지 말고 읽을 것.**

**1. `tests/conftest.py` 는 픽스처 3개 중 1개만 만든다.**
설계 §11은 `clean_barcode_bgr` / `warped_barcode_bgr` / `glared_barcode_bgr` 3개를 제시하는데, 뒤의 두 개는 `wemeet.data.synthesis` 를 쓴다. 그 파일은 2단계 산출물이다. 지금 만들면 `ImportError`로 전체 테스트가 죽는다. 따라서 1단계는 `clean_barcode_bgr` 만 완성하고, 나머지 두 개는 `synthesis.py` 가 생기는 시점에 추가한다.

**2. `import-linter` 계약 3번은 일부만 넣는다.**
설계 §4의 3번 계약은 `wemeet.sw.pipeline` 과 `wemeet.sw.server` 를 금지 대상으로 지정한다. 두 파일은 2단계 산출물이라 지금 존재하지 않는다. import-linter가 존재하지 않는 모듈을 금지 목록에 넣었을 때 오류를 내는지 아닌지를 Task 4에서 **실제로 돌려서 확인하고** 결정한다. 오류가 나면 `wemeet.ai` 만 남기고, 나머지는 2단계 할 일로 넘긴다.

**3. 설계 §19 검증 기준 중 2개는 1단계에서 확인할 수 없다.**

| 검증 항목 | 1단계 | 이유 |
|---|---|---|
| `wemeet/data/` 에서 `import wemeet.sw.pipeline` → 실패해야 함 | **연기** | `pipeline.py` 가 없다 |
| `wemeet/data/` 에서 `import wemeet.sw.decoding` → 통과해야 함 | **연기** | `decoding.py` 가 없다 |

두 번째 항목은 설계 문서가 "특히 중요하다"고 강조한 것이다 — 허용해야 하는 예외를 실수로 막아놓은 것을 잡는 검사다. **2단계 계획서 첫 줄에 이 두 항목을 넣어야 한다.** 지금 잊으면 9월에 데이터파트가 막힌다.

---

## Task 1: 프로젝트 부팅 — pyproject.toml + 빈 패키지

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `wemeet/__init__.py`
- Create: `wemeet/ai/__init__.py`
- Create: `wemeet/sw/__init__.py`
- Create: `wemeet/data/__init__.py`
- Create: `uv.lock` (명령이 생성한다. 직접 쓰지 않는다)

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `wemeet` 패키지가 import 가능해진다. `uv run` 으로 `pytest` / `ruff` / `lint-imports` 를 부를 수 있다. Task 4가 `[tool.importlinter]` 절을 이 파일에 추가한다

- [ ] **Step 1: `.python-version` 을 만든다**

```
3.12
```

- [ ] **Step 2: `pyproject.toml` 을 만든다**

`[tool.importlinter]` 절은 아직 넣지 않는다. Task 4에서 추가한다.

```toml
[project]
name = "wemeet"
version = "0.1.0"
description = "난인식 바코드 복원 파이프라인 (WE-Meet 2026, METRO팀)"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    # 4.8 부터 cv2.barcode 가 main 패키지에 포함된다. 하한을 내리지 말 것
    "opencv-python>=4.8",
    "pillow>=10.0",
    # 디코더 후보 3종 중 2개. docs/decisions/0006 확정 후 하나만 남긴다
    "pyzbar>=0.1.9",
    "zxing-cpp>=2.2",
    # 테스트 픽스처와 합성 원본 렌더링에 쓴다
    "python-barcode>=0.15",
    "fastapi>=0.115",
    "uvicorn>=0.30",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
    "import-linter>=2.0",
    # 실험 기록. 런타임 의존성이 아니다 — AI파트가 학습할 때만 쓴다.
    # uv 는 dev 그룹을 기본으로 설치하므로 `uv sync` 한 줄은 그대로 유지된다.
    "wandb>=0.17",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["wemeet"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: 빈 패키지 파일 4개를 만든다**

`wemeet/__init__.py` 만 한 줄 넣고, 나머지 3개는 **완전히 빈 파일**로 둔다.

```python
"""난인식 바코드 복원 파이프라인."""
```

```bash
touch wemeet/ai/__init__.py wemeet/sw/__init__.py wemeet/data/__init__.py
```

- [ ] **Step 4: 환경을 만든다**

Run: `uv sync`
Expected: 파이썬 3.12를 내려받고 의존성을 설치한 뒤 `uv.lock` 이 생긴다.

막히면: `pyzbar` 는 리눅스에서 시스템 라이브러리 `libzbar0` 를 필요로 하지만, **설치 시점에는 필요 없고 `import pyzbar` 하는 시점에 필요하다.** 1단계에는 pyzbar를 import하는 코드가 없으므로 여기서 막히지 않는다. 2단계에서 `decoding.py` 가 생기면 CI에 `sudo apt-get install -y libzbar0` 단계를 추가해야 한다 — 2단계 계획서에 적을 것.

- [ ] **Step 5: 패키지가 import되는지 확인한다**

Run: `uv run python -c "import wemeet; import wemeet.ai; import wemeet.sw; import wemeet.data; print('ok')"`
Expected: `ok`

- [ ] **Step 6: 커밋한다**

```bash
git add pyproject.toml .python-version uv.lock wemeet/
git commit -m "[SW] uv 프로젝트 설정과 빈 패키지 골격 추가"
```

---

## Task 2: 계약 모듈 `schemas.py` (TDD)

**Files:**
- Create: `tests/test_schemas.py`
- Create: `wemeet/schemas.py`

**Interfaces:**
- Consumes: Task 1의 `wemeet` 패키지
- Produces: 네 개의 dataclass. **이 시그니처가 AI·SW 두 파트의 계약이므로 이름 한 글자도 바꾸지 않는다.**
  - `DetectedBarcode(crop_bgr_uint8: np.ndarray, angle_deg_ccw: float, confidence: float)`
  - `RestoredBarcode(image_gray_uint8: np.ndarray, source: DetectedBarcode)`
  - `DecodeResult(text: str | None, symbology: str | None, retry_count: int, stage_ms: dict[str, float] = {}, failure_reason: str | None = None, candidate_count: int = 0, degraded: bool = False)` — `.total_ms` property
  - `PipelineResult(original_bgr_uint8: np.ndarray, detected: DetectedBarcode | None, restored: RestoredBarcode | None, decode: DecodeResult)` — `.ok` property

- [ ] **Step 1: 실패하는 테스트를 먼저 쓴다**

`tests/test_schemas.py`:

```python
import numpy as np
import pytest

from wemeet.schemas import (
    DecodeResult,
    DetectedBarcode,
    PipelineResult,
    RestoredBarcode,
)


def _crop(h: int = 20, w: int = 60) -> DetectedBarcode:
    """정상적인 DetectedBarcode 하나. 다른 테스트의 재료로 쓴다."""
    return DetectedBarcode(
        crop_bgr_uint8=np.zeros((h, w, 3), dtype=np.uint8),
        angle_deg_ccw=0.0,
        confidence=0.9,
    )


def test_잘못된_dtype은_거부된다():
    with pytest.raises(AssertionError):
        DetectedBarcode(
            crop_bgr_uint8=np.zeros((10, 10, 3), dtype=np.float32),
            angle_deg_ccw=0.0,
            confidence=0.5,
        )


def test_confidence가_1을_넘으면_거부된다():
    with pytest.raises(AssertionError):
        DetectedBarcode(
            crop_bgr_uint8=np.zeros((10, 10, 3), dtype=np.uint8),
            angle_deg_ccw=0.0,
            confidence=1.5,
        )


def test_복원_이미지가_흑백_단일채널이_아니면_거부된다():
    with pytest.raises(AssertionError):
        RestoredBarcode(
            image_gray_uint8=np.zeros((10, 10, 3), dtype=np.uint8),
            source=_crop(),
        )


def test_판독_실패시_symbology도_None이어야_한다():
    with pytest.raises(AssertionError):
        DecodeResult(
            text=None,
            symbology="CODE128",
            retry_count=0,
            failure_reason="decode_failed",
        )


def test_실패에는_반드시_이유가_있어야_한다():
    with pytest.raises(AssertionError):
        DecodeResult(text=None, symbology=None, retry_count=0, failure_reason=None)


def test_성공했는데_실패_이유가_있으면_거부된다():
    with pytest.raises(AssertionError):
        DecodeResult(
            text="123",
            symbology="CODE128",
            retry_count=0,
            failure_reason="decode_failed",
        )


def test_재시도는_3회를_넘을_수_없다():
    with pytest.raises(AssertionError):
        DecodeResult(text="123", symbology="CODE128", retry_count=4)


def test_total_ms는_단계별_시간의_합이다():
    r = DecodeResult(
        text="123",
        symbology="CODE128",
        retry_count=0,
        stage_ms={"detect": 40.0, "restore": 200.0, "decode": 20.0},
    )
    assert r.total_ms == 260.0


def test_판독_성공이면_ok는_True다():
    original = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = _crop()
    result = PipelineResult(
        original_bgr_uint8=original,
        detected=crop,
        restored=RestoredBarcode(
            image_gray_uint8=np.zeros((20, 60), dtype=np.uint8),
            source=crop,
        ),
        decode=DecodeResult(text="123", symbology="CODE128", retry_count=0),
    )
    assert result.ok is True


def test_탐지_실패여도_PipelineResult는_만들어진다():
    """파이프라인은 예외를 던지지 않는다. 실패는 값으로 표현한다 (설계 §5)."""
    result = PipelineResult(
        original_bgr_uint8=np.zeros((100, 200, 3), dtype=np.uint8),
        detected=None,
        restored=None,
        decode=DecodeResult(
            text=None,
            symbology=None,
            retry_count=0,
            failure_reason="not_detected",
        ),
    )
    assert result.ok is False
    assert result.decode.failure_reason == "not_detected"
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 수집 단계에서 `ModuleNotFoundError: No module named 'wemeet.schemas'` 로 전부 실패한다.

- [ ] **Step 3: `wemeet/schemas.py` 를 만든다**

```python
"""AI파트와 SW파트가 주고받는 데이터 형식.

이 파일은 wemeet 안의 어떤 모듈도 import 하지 않는다.
변경하려면 AI·SW 양쪽과 먼저 상의한다 (CONTRIBUTING.md).

이름에 단위를 박아둔 것은 의도한 것이다. angle 이라고만 쓰면 라디안을
넣는 사람이 나오고, crop 이라고만 쓰면 RGB를 넣는 사람이 나온다.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectedBarcode:
    """Stage 1 (탐지) → Stage 2 (복원)"""

    crop_bgr_uint8: np.ndarray  # BGR 순서, 0~255, shape (H, W, 3)
    angle_deg_ccw: float  # 원본에서 회전 보정한 각도. 도(degree), 반시계
    confidence: float  # 0.0 ~ 1.0

    def __post_init__(self) -> None:
        assert self.crop_bgr_uint8.dtype == np.uint8
        assert self.crop_bgr_uint8.ndim == 3
        assert self.crop_bgr_uint8.shape[2] == 3
        assert 0.0 <= self.confidence <= 1.0


@dataclass
class RestoredBarcode:
    """Stage 2 (복원) → Stage 3 (디코딩)"""

    image_gray_uint8: np.ndarray  # 흑백 단일 채널, 0~255, shape (H, W)
    source: DetectedBarcode

    def __post_init__(self) -> None:
        assert self.image_gray_uint8.dtype == np.uint8
        assert self.image_gray_uint8.ndim == 2


@dataclass
class DecodeResult:
    """Stage 3 (디코딩) → 최종 출력"""

    text: str | None  # 판독 실패 시 None
    symbology: str | None  # 예: "CODE128". 실패 시 None
    retry_count: int  # 0 ~ 3
    stage_ms: dict[str, float] = field(default_factory=dict)
    # 예: {"detect": 42.1, "restore": 210.3, "decode": 18.7, "retry": 95.0}

    failure_reason: str | None = None  # invalid_input / not_detected / decode_failed
    candidate_count: int = 0  # 탐지된 바코드 총 개수
    degraded: bool = False  # 복원 실패로 원본 크롭을 쓴 경우

    def __post_init__(self) -> None:
        assert 0 <= self.retry_count <= 3
        if self.text is None:
            assert self.symbology is None
            assert self.failure_reason is not None  # 실패에는 항상 이유가 있다
        else:
            assert self.failure_reason is None

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

    original_bgr_uint8: np.ndarray  # 입력 원본 (대조 표시용)
    detected: DetectedBarcode | None  # 탐지 실패 시 None
    restored: RestoredBarcode | None  # 탐지 실패 시 None
    decode: DecodeResult  # 실패해도 항상 존재 (text=None)

    @property
    def ok(self) -> bool:
        return self.decode.text is not None
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 10 passed

- [ ] **Step 5: 스타일을 맞춘다**

Run: `uv run ruff format . && uv run ruff check --fix .`
Expected: `All checks passed!`

`ruff format` 은 import 순서를 고치지 않는다. import 정렬(`I001`)은 `ruff check --fix` 가 고친다. 두 명령이 필요한 이유다.

- [ ] **Step 6: 커밋한다**

```bash
git add wemeet/schemas.py tests/test_schemas.py
git commit -m "[SW] 파트 간 계약 schemas.py 와 검증 테스트 추가"
```

---

## Task 3: 바코드 이미지 픽스처

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_schemas.py` (픽스처를 쓰는 테스트 1개 추가)

**Interfaces:**
- Consumes: `python-barcode`, `pillow`, `numpy` (Task 1이 설치)
- Produces: 픽스처 2개
  - `render_code128` → `Callable[[str], np.ndarray]`. 번호를 주면 BGR uint8 배열을 준다
  - `clean_barcode_bgr` → `np.ndarray`. 번호 `WEMEET0001` 이 찍힌 정상 Code128 이미지

**왜 이렇게 하나:** `downloads/` 는 git 제외 대상이라 GitHub 러너에 없다. 테스트가 이미지 파일을 읽으면 로컬만 통과하고 CI에서 전부 깨진다 — 원인 찾기 가장 짜증나는 형태다. 그리고 **렌더링할 때 넣은 번호가 곧 정답**이므로, 나중에 디코딩 성공/실패를 CI에서 자동 판정할 수 있다.

- [ ] **Step 1: 실패하는 테스트를 먼저 쓴다**

`tests/test_schemas.py` 맨 아래에 추가한다.

```python
def test_렌더링한_바코드는_DetectedBarcode에_그대로_들어간다(clean_barcode_bgr):
    """픽스처가 계약이 요구하는 형식(BGR, uint8, 3채널)으로 이미지를 준다."""
    detected = DetectedBarcode(
        crop_bgr_uint8=clean_barcode_bgr,
        angle_deg_ccw=0.0,
        confidence=1.0,
    )
    assert detected.crop_bgr_uint8.ndim == 3
    assert detected.crop_bgr_uint8.shape[2] == 3
    assert detected.crop_bgr_uint8.dtype == np.uint8
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `uv run pytest tests/test_schemas.py -k 렌더링 -v`
Expected: `fixture 'clean_barcode_bgr' not found` 로 실패(error)한다.

- [ ] **Step 3: `tests/conftest.py` 를 만든다**

```python
"""테스트용 바코드 이미지를 코드로 만든다.

downloads/ 는 git 제외 대상이라 GitHub 러너에 존재하지 않는다.
테스트가 이미지 파일을 읽으면 CI에서만 깨지므로, 전부 코드로 생성한다.

렌더링할 때 넣은 번호가 곧 정답이다. 디코더가 붙는 2단계부터
판독 성공/실패를 CI에서 자동 판정할 수 있다.
"""

import io
from collections.abc import Callable

import numpy as np
import pytest
from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image

# import 순서 주의: ruff의 isort 규칙은 같은 그룹 안에서 `import x` 를 전부 먼저 놓고
# 그다음 `from x import y` 를 알파벳순으로 놓는다. 이 순서를 바꾸면 I001 이 뜬다.


@pytest.fixture
def render_code128() -> Callable[[str], np.ndarray]:
    """번호를 주면 그 번호가 찍힌 Code128 이미지(BGR uint8)를 만드는 함수."""

    def _render(text: str) -> np.ndarray:
        buffer = io.BytesIO()
        Code128(text, writer=ImageWriter()).write(buffer)
        buffer.seek(0)
        rgb = np.asarray(Image.open(buffer).convert("RGB"))
        # OpenCV 계열은 BGR 이다. 계약(crop_bgr_uint8)도 BGR 이므로 여기서 뒤집는다
        return np.ascontiguousarray(rgb[:, :, ::-1])

    return _render


@pytest.fixture
def clean_barcode_bgr(render_code128: Callable[[str], np.ndarray]) -> np.ndarray:
    """훼손 없는 정상 Code128 이미지. 정답 번호는 WEMEET0001 이다."""
    return render_code128("WEMEET0001")


# 아직 만들지 않은 픽스처 2개 (설계 문서 §11):
#
#   warped_barcode_bgr   곡률 왜곡을 준 바코드
#   glared_barcode_bgr   반사 효과를 준 바코드
#
# 둘 다 wemeet.data.synthesis 를 쓴다. 그 파일은 2단계 산출물이므로
# synthesis.py 가 생기는 시점에 여기에 추가한다. 지금 만들면 ImportError 로
# 전체 테스트가 죽는다.
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `uv run pytest -v`
Expected: 11 passed

막히면: `python-barcode` 의 `ImageWriter` 는 Pillow를 필요로 한다. `ModuleNotFoundError: No module named 'PIL'` 이 나오면 Task 1의 `pillow>=10.0` 이 빠진 것이다.

- [ ] **Step 5: 이미지 파일을 읽지 않는다는 것을 확인한다**

Run: `uv run pytest -v -p no:cacheprovider` 를 `downloads/` 를 비운 상태에서 돌린다. (`downloads/` 에는 `.gitkeep` 만 있어야 한다)
Expected: 11 passed. 설계 §19 검증 기준 5번이 여기서 충족된다.

- [ ] **Step 6: 커밋한다**

```bash
git add tests/conftest.py tests/test_schemas.py
git commit -m "[SW] 바코드 이미지를 코드로 만드는 테스트 픽스처 추가"
```

---

## Task 4: import-linter 계약 — 규칙이 실제로 막는지 확인한다

**Files:**
- Modify: `pyproject.toml` (`[tool.importlinter]` 절 추가)

**Interfaces:**
- Consumes: Task 1의 패키지 4개, Task 2의 `schemas.py`
- Produces: `uv run lint-imports` 가 의존 규칙을 강제한다

**이 태스크의 핵심은 계약을 적는 게 아니라 "정말 막히는지" 확인하는 것이다.** 적어만 두고 검증하지 않으면 규칙이 작동하지 않는 채로 9월까지 간다.

- [ ] **Step 1: `pyproject.toml` 맨 아래에 계약을 추가한다**

3번 계약의 금지 목록에서 `wemeet.sw.pipeline` / `wemeet.sw.server` 를 **일단 뺀다.** 두 파일이 아직 없다. Step 3에서 되돌릴지 결정한다.

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
forbidden_modules = ["wemeet.ai"]

[[tool.importlinter.contracts]]
name = "schemas는 아무것도 import하지 않는다"
type = "forbidden"
source_modules = ["wemeet.schemas"]
forbidden_modules = ["wemeet.ai", "wemeet.sw", "wemeet.data"]
```

- [ ] **Step 2: 통과하는지 확인한다**

Run: `uv run lint-imports`
Expected: `Contracts: 4 kept, 0 broken.`

- [ ] **Step 3: 없는 모듈을 금지 목록에 넣어도 되는지 실험한다**

3번 계약의 `forbidden_modules` 를 설계 문서 원안대로 바꾼다.

```toml
forbidden_modules = ["wemeet.ai", "wemeet.sw.pipeline", "wemeet.sw.server"]
```

Run: `uv run lint-imports`

- **통과하면** 이 상태를 유지한다. 2단계에서 손댈 필요가 없어진다
- **오류가 나면**(모듈이 없다는 메시지) Step 1의 상태로 되돌리고, 아래 주석을 그 계약 위에 붙인다

```toml
# 2단계 할 일: sw/pipeline.py 와 sw/server.py 가 생기면
# forbidden_modules 에 "wemeet.sw.pipeline", "wemeet.sw.server" 를 추가한다.
# 지금은 두 파일이 없어서 넣을 수 없다.
```

어느 쪽이든 **결과를 커밋 메시지에 한 줄로 남긴다.**

- [ ] **Step 4: 금지 규칙이 실제로 막는지 확인한다 (일부러 위반)**

`wemeet/ai/__init__.py` 에 한 줄을 넣는다.

```python
import wemeet.sw  # noqa: F401
```

Run: `uv run lint-imports`
Expected: **실패한다.** `Contracts: 3 kept, 1 broken.` 과 함께 `wemeet.ai -> wemeet.sw` 가 표시된다.

- [ ] **Step 5: 위반을 되돌린다**

```bash
git checkout wemeet/ai/__init__.py
```

Run: `uv run lint-imports`
Expected: `4 kept, 0 broken.`

- [ ] **Step 6: 바닥 계약도 막는지 확인한다 (일부러 위반)**

`wemeet/schemas.py` 맨 위에 한 줄을 넣는다.

```python
import wemeet.ai  # noqa: F401
```

Run: `uv run lint-imports`
Expected: **실패한다.** `schemas는 아무것도 import하지 않는다` 계약이 broken으로 뜬다.

되돌린다: `git checkout wemeet/schemas.py` → `uv run lint-imports` 가 `4 kept, 0 broken.`

- [ ] **Step 7: 커밋한다**

```bash
git add pyproject.toml
git commit -m "[SW] import-linter 의존 규칙 계약 추가"
```

커밋 메시지 본문에 Step 3의 결과를 적는다. 예: `없는 모듈을 forbidden_modules 에 넣으면 오류가 나므로 sw.pipeline/sw.server 는 2단계로 넘긴다.`

---

## Task 5: CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: Task 1~4의 모든 명령
- Produces: PR마다 `ci` status check가 돈다. 브랜치 보호에서 이 이름을 고른다

- [ ] **Step 1: `.github/workflows/ci.yml` 을 만든다**

```yaml
name: ci

on:
  pull_request:
  push:
    branches: [main]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: uv 설치
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: 의존성 설치
        run: uv sync --frozen

      - name: 코드 스타일
        run: |
          uv run ruff format --check .
          uv run ruff check .

      - name: 파트 간 import 규칙
        run: uv run lint-imports

      - name: 테스트
        run: uv run pytest -v
```

`setup-uv@v6` 태그가 없다는 오류가 나면 <https://github.com/astral-sh/setup-uv> 의 최신 메이저 태그로 바꾼다.

`--frozen` 은 `uv.lock` 이 `pyproject.toml` 과 어긋나면 실패시킨다. 누군가 의존성을 추가하고 lock을 커밋하지 않으면 CI가 잡는다.

**프론트엔드 검사(`tsc --noEmit`, `npm run build`)는 넣지 않는다.** `web/` 이 없다. 설계 §24대로 2단계에서 추가한다.

- [ ] **Step 2: CI가 돌릴 명령을 로컬에서 그대로 돌린다**

```bash
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run lint-imports
uv run pytest -v
```

Expected: 5개 명령 모두 exit 0. 하나라도 실패하면 CI도 실패하므로 여기서 고친다.

- [ ] **Step 3: 커밋한다**

```bash
git add .github/workflows/ci.yml
git commit -m "[SW] CI 워크플로 추가 (ruff + import-linter + pytest)"
```

---

## Task 6: `docs/architecture.md` 와 README 상태 갱신

**Files:**
- Create: `docs/architecture.md`
- Create: `docs/reports/.gitkeep`
- Modify: `README.md` ("지금 상태" 절)

**Interfaces:**
- Consumes: Task 1~5의 결과
- Produces: 없음 (문서)

- [ ] **Step 1: `docs/architecture.md` 를 쓴다**

설계 문서 §3·§4·§5에서 **팀원이 매일 볼 부분만** 뽑아 옮긴다. 1500줄 설계 문서를 다시 읽게 하지 않는 것이 목적이다. 아래 5개 절을 넣는다.

1. **폴더 구조** — 설계 §3의 트리. 단, 아직 없는 파일에는 `(2단계)` 를 표시한다
2. **의존 방향** — 설계 §4의 4층 그림과 "허용되는 import 5개" 표를 그대로
3. **팀 규칙 3줄** — 설계 §4의 인용 블록 3개를 그대로
4. **스테이지 함수 시그니처** — 설계 §5의 4개 함수. 각 파트가 구현할 형태
5. **시간 예산** — 설계 §5의 표 (탐지 100 / 복원 250 / 디코딩 50 / 재시도 100 = 500ms). **0.5초는 재시도를 포함한 전체 경로 기준**이라는 문장을 반드시 넣는다

마지막에 미정 항목을 적는다.

```markdown
## 아직 정하지 않은 것

**추론 대상 하드웨어** — 0.5초 목표의 기준 장비가 기획서에 없다. 이것이 정해지지
않으면 "500ms 안에 끝났다"는 말의 의미가 확정되지 않는다. 멘토 자문 필요.
담당: 팀장 / 기한: 통합 단계 전 (설계 문서 §20)
```

- [ ] **Step 2: 보고서 자리를 만든다**

```bash
touch docs/reports/.gitkeep
```

`.gitignore` 가 `*.pdf` 를 막고 `!docs/reports/*.pdf` 로 예외를 열어뒀다. 11월 성과 보고서가 여기 들어간다.

- [ ] **Step 3: README의 "지금 상태" 절을 갱신한다**

현재 내용은 "아직 코드가 없습니다"다. 골격이 생겼으니 사실과 다르다. 아래로 바꾼다.

```markdown
## 지금 상태

**골격까지 만들었습니다.** `uv sync` → `uv run pytest` 가 통과하고, 파트 간
import 규칙이 CI로 강제됩니다.

아직 없는 것은 각 단계의 실제 코드입니다.

| 있다 | 없다 (2단계) |
|---|---|
| `wemeet/schemas.py` — 파트 간 계약 | `detection.py` / `restoration.py` / `decoding.py` |
| `tests/` — 계약 검증 + 바코드 픽스처 | `pipeline.py` / `server.py` |
| CI 3검사 | `download.py` / `synthesis.py` / `ground_truth.py` |
| | `web/` React 화면 |

위 "데이터 받기"의 `wemeet.data.download` 도 2단계에서 만듭니다. 지금은
받아올 데이터 자체가 없습니다.
```

- [ ] **Step 4: 커밋한다**

```bash
git add docs/architecture.md docs/reports/.gitkeep README.md
git commit -m "[문서] architecture.md 추가와 README 상태 갱신"
```

---

## 완료 판정

설계 문서 §19의 검증 기준을 이 계획의 범위로 옮긴 것이다. **전부 직접 돌려서 확인한다.**

- [ ] 1. 빈 상태에서 통과한다

```bash
git clone <repo> /tmp/wemeet-clean && cd /tmp/wemeet-clean
uv sync && uv run pytest -v
```
→ 11 passed

- [ ] 2. `uv run lint-imports` → `4 kept, 0 broken.`

- [ ] 3. 일부러 넣은 위반 2개가 막힌다 (Task 4 Step 4·6에서 확인함)
  - `wemeet/ai/` 에 `import wemeet.sw` → 실패
  - `wemeet/schemas.py` 에 `import wemeet.ai` → 실패

- [ ] 4. `pytest` 가 `downloads/` 없이 통과한다 (Task 3 Step 5)

- [ ] 5. `main` 에 직접 push가 거부된다 — 브랜치 보호를 켠 뒤 확인한다 (`docs/external/README.md` §1)

- [ ] 6. 승인 없는 PR은 머지 버튼이 비활성화된다

**2단계로 넘기는 검증 2개** — 2단계 계획서 첫 줄에 적는다.

- `wemeet/data/` 에서 `import wemeet.sw.pipeline` → 실패해야 한다
- `wemeet/data/` 에서 `import wemeet.sw.decoding` → **통과해야 한다** (허용된 예외를 실수로 막지 않았는지 확인. 이게 막히면 9월 GT 확보 작업이 멈춘다)

## 2단계로 넘기는 것 (잊지 말 것)

| 항목 | 왜 지금 못 하나 |
|---|---|
| `conftest.py` 의 `warped_barcode_bgr` / `glared_barcode_bgr` 픽스처 | `wemeet.data.synthesis` 가 없다 |
| import-linter 3번 계약에 `sw.pipeline` / `sw.server` 추가 | 두 파일이 없다 (Task 4 Step 3 결과에 따라 불필요할 수도 있다) |
| CI에 `libzbar0` 설치 단계 | `pyzbar` 를 import하는 코드가 생기는 시점에 필요하다 |
| CI에 프론트엔드 검사 (`tsc --noEmit`, `npm run build`) | `web/` 이 없다 |
| `set_seed()` 시드 고정 유틸 (설계 §11) | 호출할 학습 스크립트와 `synthesis.py` 가 없다. `schemas.py` 에는 넣지 않는다 — 계약 모듈에 유틸이 섞이면 안 된다 |
| 위 "2단계로 넘기는 검증 2개" | 해당 파일이 없다 |
