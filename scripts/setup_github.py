"""GitHub 라벨·마일스톤·착수 이슈를 한 번에 만든다.

실행:
    GH_TOKEN=ghp_xxxx uv run python scripts/setup_github.py

미리 볼 때 (아무것도 만들지 않는다):
    GH_TOKEN=ghp_xxxx uv run python scripts/setup_github.py --dry-run

토큰은 코드에 넣지 않는다. 환경변수로만 받는다.
필요한 권한: classic 토큰이면 `repo`, fine-grained 토큰이면 Issues=write + Metadata=read.
다 만들고 나면 토큰을 폐기(revoke)해도 된다. 이 스크립트는 한 번만 돌린다.

이미 같은 이름이 있으면 건너뛴다. 여러 번 돌려도 중복이 생기지 않는다.

무엇을 만드는지는 docs/external/README.md 에 정리돼 있다.
이 파일이 그 문서의 실행 가능한 버전이다.
"""

import json
import os
import sys
import urllib.error
import urllib.request

REPO = "kim1034/2026-Wemeet-Barcode-Restoration"
API = "https://api.github.com"

# --------------------------------------------------------------------------
# 라벨 — docs/external/README.md §1
# --------------------------------------------------------------------------
# part: 접두사를 쓰는 이유: GitHub 라벨 목록이 알파벳순이라 세 개가 붙어서 보인다.
LABELS = [
    ("part:ai", "1d76db", "AI파트 작업"),
    ("part:sw", "0e8a16", "SW파트 작업"),
    ("part:data", "fbca04", "데이터파트 작업"),
    ("blocked", "b60205", "다른 작업을 기다리는 중"),
    ("decision", "5319e7", "결정이 필요한 사항 (docs/decisions/ 와 연결)"),
    ("good-first-issue", "7057ff", "1학년이 먼저 잡을 작업"),
]

# --------------------------------------------------------------------------
# 마일스톤 — 설계 문서 §22
# --------------------------------------------------------------------------
# 2026-08 의 완료 조건에서 "더미 파이프라인"을 뺐다. 범위에서 제외했으므로
# 지금 상태(CI 통과)로 이미 충족이다.
MILESTONES = [
    ("2026-08 레포 구축", "2026-08-31T23:59:59Z", "CI가 통과한다"),
    (
        "2026-09 데이터+탐지",
        "2026-09-30T23:59:59Z",
        "데이터셋 v1, 탐지 모델 확정, 베이스라인 수치 확보",
    ),
    (
        "2026-10 기하보정+통합",
        "2026-10-31T23:59:59Z",
        "통합 파이프라인 동작, E2E 지연 측정 완료",
    ),
    (
        "2026-11 평가+보고서",
        "2026-11-30T23:59:59Z",
        "시연 프로그램, 평가 보고서, 발표 자료",
    ),
]

M09 = "2026-09 데이터+탐지"
M10 = "2026-10 기하보정+통합"

# --------------------------------------------------------------------------
# 이슈
# --------------------------------------------------------------------------
# 본문 형식은 .github/ISSUE_TEMPLATE/task.md 와 맞춘다.
# 완료 조건은 "동작하면" 같은 말이 아니라 확인할 수 있는 문장으로 쓴다.
ISSUES = [
    # ---- 9월 1주차: 이것들이 막히면 뒤가 전부 막힌다 ----
    {
        "title": "[결정] 바코드 심볼로지 범위 확정 (1D / QR)",
        "labels": ["decision"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

인천세관 통합검사장에서 실제로 쓰는 바코드 종류를 확인하고 대응 범위를 정한다.
QR을 포함할지가 핵심이다.

## 현장에서 확인할 것 — 한 번에 다 물어볼 것

**현장에 두 번 가기 어렵다.** 심볼로지만 받아 오면 나중에 다시 물어봐야 하는
항목들이 있어서 함께 적어뒀다.

- [ ] 무슨 심볼로지인가 (Code128 / GS1-128 / ITF-14 / 기타)
- [ ] QR 이 섞여 있는가
- [ ] **번호가 몇 자리인가**
- [ ] 숫자만인가, 영문이 섞이는가
- [ ] 고정 접두사가 있는가
- [ ] 송장 한 장에 바코드가 몇 개 붙어 있는가
- [ ] 인쇄 상태 — 감열지인지, 흐리게 인쇄된 것이 흔한지

**자릿수가 왜 필요한가**: 바코드 길이가 곧 난이도다. 10자면 약 68 모듈,
20자면 약 123 모듈이라 같은 폭에서 막대가 절반 두께가 된다.
사전 실험에서 "모듈 폭이 1~2px 이하로 압축되면 복원 불가" 를 측정했으므로
자릿수가 촬영 해상도 요구사항까지 결정한다.

**바코드 개수가 왜 필요한가**: "탐지 2개 이상이면 1등만 처리" 라고 정한 근거가
"물류 현장에서 흔하다" 는 추측이다. 실제 개수를 보면 그 결정을 검증할 수 있다.

## 완료 조건

`docs/decisions/0001-바코드-심볼로지-범위.md` 의 "결정과 이유"가 채워지고
맨 위 상태가 `결정` 으로 바뀐다. 위 확인 항목이 전부 체크돼 있어야 한다.

## 왜 최우선인가

이것이 틀리면 **이후 데이터 수집·라벨링 작업이 전부 무효**가 된다.
현장 확인이 필요하므로 다른 작업보다 먼저 착수한다 (설계 문서 §20).

이것이 확정돼야 `0004` 의 합성 원본 렌더링 규격을 정할 수 있고,
그게 정해져야 데이터파트가 synthesis.py 를 쓸 수 있다.

## 막히면

팀장. 멘토 문의가 필요할 수 있다.""",
    },
    {
        "title": "[결정] 디코더 선정 — pyzbar / zxing-cpp / OpenCV 실측 비교",
        "labels": ["decision", "part:sw"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

디코더 후보 3종을 같은 이미지로 돌려 판독률과 속도를 비교한다.
`cv2.barcode.BarcodeDetector` 는 opencv-python 4.8 이상에 포함되어 있어
별도 설치가 필요 없다 (`pyproject.toml` 에 하한이 걸려 있다).

## 완료 조건

`docs/decisions/0006-디코더-선정.md` 의 비교 표가 채워지고 상태가 `결정` 이 된다.
선정 후 `pyproject.toml` 에서 탈락한 패키지를 제거하는 PR까지 올린다.

## 막히면

SW파트. `pyzbar` 는 리눅스에서 시스템 라이브러리 `libzbar0` 가 필요하다
(`import` 시점에 필요하다). CI에서 쓰려면 워크플로에 설치 단계를 추가해야 한다.""",
    },
    {
        "title": "디코더 베이스라인 측정 — 보정 없이 얼마나 읽히는지",
        "labels": ["part:sw"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

기하 보정을 붙이지 않은 상태에서 디코더만으로 판독률을 측정한다 (설계 문서 §14 2단계).

조기 종료 비율도 같이 재둔다 — 기울기 보정만으로 읽히는 비율이다.
사전 실험에서 수직 방향 왜곡은 디코딩을 깨지 않는 것을 확인했으므로 예상보다 높을 수 있다.

## 완료 조건

훼손 유형별 판독률 수치가 `docs/decisions/0005-평가-지표-체계.md` 에 기록된다.

## 왜 필요한가

**목표 수치를 근거 있게 정하기 위한 기준선이다.** 이 값이 없으면
"디코딩률 90%"가 쉬운 목표인지 불가능한 목표인지 알 수 없다.
기하 보정의 기여도도 이 값과의 차이로만 말할 수 있다.

## 막히면

SW파트. 디코더 선정 이슈와 병렬로 진행할 수 있다.""",
    },
    {
        "title": "오픈소스 바코드 데이터 실태 파악",
        "labels": ["part:data"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

공개 데이터셋에 어떤 라벨이 붙어 있는지, 우리가 쓸 수 있는 형태인지 확인한다
(설계 문서 §14 1단계).

## 완료 조건

데이터셋별로 장수·라벨 형식(BBox / OBB / 없음)·라이선스를 표로 정리해
`docs/decisions/0004-데이터-수집-및-GT-프로토콜.md` 에 넣는다.

## 왜 지금 하나

**재라벨링 규모가 여기서 정해진다.** OBB가 필요한데 BBox만 있으면
1학년 2명의 라벨링 작업량이 결정되고, 그게 9월 2주차 일정을 좌우한다.

## 막히면

데이터파트.""",
    },
    {
        "title": "학습용 GPU 확보 경로 결정",
        "labels": ["part:ai"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

연구실 서버 / 학교 클러스터 / Colab 중 무엇으로 학습을 돌릴지 정한다.

## 완료 조건

`docs/external/README.md` 에 사용할 환경과 접근 방법(계정·예약 절차)이 적힌다.

## 왜 지금 하나

**GitHub 무료 러너에는 GPU가 없다.** 학습은 CI에서 돌릴 수 없으므로
어디서 돌릴지 정해두지 않으면 9월 4주차 모델 선정이 시작되지 않는다 (설계 문서 §20).

## 막히면

AI파트. 지도교수·멘토 문의가 필요할 수 있다.""",
    },
    # ---- 9월 2주차 ----
    {
        "title": "[결정] 평가 지표 체계와 목표 수치 확정",
        "labels": ["decision"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

기획서 4.6의 지표를 검토하고 목표 수치를 정한다.
설계 문서 §17에 발견된 문제와 제안이 정리돼 있다.

## 완료 조건

`docs/decisions/0005-평가-지표-체계.md` 의 지표 목록과 목표 수치가 채워지고
상태가 `결정` 이 된다.

## 주의

기획서 내용을 바꾸는 것이므로 **멘토·지도교수 확인이 필요하다.**
목표 수치는 베이스라인 측정 이슈가 끝난 뒤에 정한다 — 그 전에는 근거가 없다.

## 막히면

팀장 + 전원.""",
    },
    {
        "title": "[결정] 데이터 수집·GT 프로토콜과 라벨링 도구 선정",
        "labels": ["decision", "part:data"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

데이터 구성 비율, 정답(GT) 확보 절차, 라벨링 도구(CVAT / Roboflow)를 정한다.

## 완료 조건

`docs/decisions/0004-데이터-수집-및-GT-프로토콜.md` 상태가 `결정` 이 된다.

## 반드시 포함할 것

**실촬영은 "평탄 상태로 먼저 촬영 → 디코더로 정답 확보 → 그다음 훼손 상태 촬영"
순서를 지켜야 한다.** 순서를 바꾸면 정답 번호를 알 수 없는 이미지가 남는다
(설계 문서 §14 5단계).

촬영 시 **개인정보 영역을 물리적으로 차폐**한다 (테이프·마스킹지).
촬영 후 블러보다 촬영 전 차폐가 확실하다.

## 막히면

데이터파트.""",
    },
    {
        "title": "라벨링 가이드라인 작성 + 공동 라벨링 20장 + 교차 검수",
        "labels": ["part:data", "good-first-issue"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

라벨링 기준을 문서로 만들고, 전원이 같은 20장을 라벨링해 기준을 맞춘 뒤
본 작업에 들어간다. 이후 10%를 교차 검수한다 (설계 문서 §14 3단계).

## 완료 조건

가이드라인 문서가 있고, 공동 20장의 라벨이 서로 일치하는 것을 확인했다.

## 왜 20장을 같이 하나

**기준을 안 맞추고 나눠서 하면 사람마다 다르게 라벨링한다.**
구겨진 바코드의 경계를 어디까지 볼지가 특히 갈린다.
나중에 발견하면 전부 다시 해야 한다.

## 막히면

데이터파트 (김강민, 조아라). 1학년(이다현, 이아침)이 함께 참여한다.""",
    },
    # ---- 2단계 골격: 범위에서 뺐지만 남은 일 ----
    {
        "title": "Stage 1 detection.py 스텁 — 탐지·크롭",
        "labels": ["part:ai"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

`wemeet/ai/detection.py` 에 계약대로 함수를 만든다. 모델 없이 동작하는
가짜 구현이어도 된다 — 이미지 중앙을 잘라 반환하는 식으로.

```python
def detect(image_bgr: np.ndarray) -> list[DetectedBarcode]: ...
```

## 완료 조건

`from wemeet.ai.detection import detect` 가 되고, 반환값이 `DetectedBarcode`
리스트이며 `uv run pytest` 와 `uv run lint-imports` 가 통과한다.

## 주의

`wemeet.sw` 나 `wemeet.data` 를 import하면 CI가 막는다.
공용이 필요하면 `wemeet.schemas` 를 쓴다. 자세한 규칙은 `docs/architecture.md`.

## 막히면

AI파트.""",
    },
    {
        "title": "Stage 2 geometry.py 스텁 — 기하 추정 (이미지 안 만듦)",
        "labels": ["part:ai"],
        "milestone": M10,
        "body": """## 무엇을 해야 하나

`wemeet/ai/geometry.py` 에 계약대로 함수를 만든다.

```python
def estimate_geometry(target: DetectedBarcode) -> GeometryField: ...
```

**이미지를 반환하지 않는다.** 얼마나 휘었는지를 제어점 좌표로만 내놓는다.
`GeometryField` 에 이미지 필드가 없으므로 픽셀을 만들어 넘길 방법이 구조적으로 없다.
생성 모델이 만든 가짜 바코드는 체크섬을 우연히 통과할 수 있어서(Code128 약 1/103)
이렇게 막아뒀다.

**리스트가 아니라 하나를 받는다.** 검출 결과가 여러 개일 때 무엇을 보정할지는
AI파트가 아니라 `pipeline` 이 정할 문제다.

좌표 규칙을 지킬 것 — 크롭 기준, 0~1 정규화, (x, y) 순서,
`dst`=펴진 격자 / `src`=지금 있는 위치. **방향을 뒤집으면 왜곡이 두 번 적용되는데
약한 왜곡에서는 우연히 읽혀서 놓치기 쉽다.**

## 완료 조건

반환값이 `GeometryField` 이고 제어점이 8개 이상이다.
`uv run pytest` 와 `uv run lint-imports` 가 통과한다.

이미지를 반환하지 않는다는 것을 테스트로 확인할 것 —
`GeometryField` 에 이미지 필드가 없다.

## 막히면

AI파트.""",
    },
    {
        "title": "Stage 3 rectify.py — OpenCV 기하 보정",
        "labels": ["part:sw"],
        "milestone": M10,
        "body": """## 무엇을 해야 하나

`wemeet/sw/rectify.py` 를 만든다. AI가 준 제어점으로 이미지를 실제로 편다.

```python
def apply_field(
    target: DetectedBarcode,
    field: GeometryField,
    interpolation: int = cv2.INTER_CUBIC,
) -> RectifiedBarcode: ...
```

학습이 없는 결정론적 코드라 모델을 기다릴 필요가 없다. 지금 바로 진짜로 만들 수 있다.

## 완료 조건

- 항등 대응(src == dst)을 넣으면 `map_x[y, x] ~= x` 다. **이것부터 확인할 것**
- 35도 압축한 합성 바코드를 정답 제어점으로 보정하면 디코딩이 성공한다
- **src/dst 를 뒤집으면 디코딩이 실패한다** (방향 함정 테스트)
- 20ms 안에 끝난다

## TPS 는 직접 계산한다

`cv2.createThinPlateSplineShapeTransformer` 는 **opencv-python 에 없다.**
contrib(78MB) 에만 있어서 쓰지 않기로 했다. numpy 로 30줄이면 되고,
검증된 코드가 설계 문서 §9 에 있다. 그대로 쓰면 된다.

## 주의

제어점은 0~1 정규화 좌표다. 픽셀로 바꿔서 TPS 를 풀어야 한다.
그리고 **새 픽셀을 만들지 마라.** 인페인팅이나 생성을 붙이면 이 설계의 근거가 무너진다.

## 막히면

SW파트. 설계 문서 §9 와 docs/parts/sw.md 의 rectify.py 절을 볼 것.""",
    },
    {
        "title": "Stage 4 decoding.py 스텁 + 반사 대응",
        "labels": ["part:sw"],
        "milestone": M10,
        "body": """## 무엇을 해야 하나

`wemeet/sw/decoding.py` 에 계약대로 함수를 만들고 피드백 루프를 붙인다.

```python
def decode(image: RectifiedBarcode) -> DecodeResult: ...
```

## 완료 조건

재시도는 **최대 3회**이고 실제 시도 횟수가 `retry_count` 에 들어간다
(계약이 `0 <= retry_count <= 3` 을 검사한다).
판독 실패 시 `text=None`, `symbology=None`, `failure_reason="decode_failed"`.

## 주의

**실패했는데 이유가 없는 상태는 계약이 금지한다.** `text=None` 이면
`failure_reason` 이 반드시 있어야 `DecodeResult` 생성이 통과한다.

## 막히면

SW파트.""",
    },
    {
        "title": "pipeline.py — 4단계 연결 + 조기 종료",
        "labels": ["part:sw"],
        "milestone": M10,
        "body": """## 무엇을 해야 하나

`wemeet/sw/pipeline.py` 에서 검출 → (1차 디코딩) → 기하 추정 → 기하 보정 → 디코딩을
이어 붙인다. 1차 디코딩에서 읽히면 뒤 단계를 건너뛴다.

```python
def run(image_bgr: np.ndarray) -> PipelineResult: ...
```

## 완료 조건

설계 문서 §5의 실패 정책 7가지가 전부 구현되고 테스트로 확인된다.

- 이미지 형식 오류 → `failure_reason="invalid_input"`
- 검출 0개 → `failure_reason="not_detected"`, 이후 단계 생략
- 탐지 2개 이상 → confidence 최상위 1개만 처리, 총 개수를 `candidate_count` 에
- 기하 추정 실패 또는 신뢰도 0.3 미만 → 보정 없이 진행, `degraded=True`
- 디코딩 실패 → 보정 파라미터를 바꿔 최대 3회 재시도 (추정은 다시 안 돌린다)
- 500ms 초과 → **중단하지 않고 완료**, 시간만 기록

## 원칙

**파이프라인은 예외를 던지지 않는다.** 어떤 상황에서도 `PipelineResult` 를 반환한다.
현장 컨베이어에서 예외가 올라오면 서비스가 멈추고, 그게 곧 노리드 존 정체다.

## 막히면

SW파트.""",
    },
    {
        "title": "server.py — FastAPI 엔드포인트",
        "labels": ["part:sw"],
        "milestone": M10,
        "body": """## 무엇을 해야 하나

`wemeet/sw/server.py` 에 `POST /api/decode` 를 만든다.
응답 형식은 설계 문서 §23에 전부 정해져 있다.

## 완료 조건

이미지를 올리면 원본·펴진 이미지(base64 data URI)와 판독 결과가 함께 온다.
**판독 실패도 HTTP 200** 으로 반환하고 `ok: false` 와 `failure_reason` 을 담는다.

## 왜 실패에 200인가

판독 실패는 서버 오류가 아니라 **정상적인 처리 결과**다. 4xx/5xx로 주면
프론트엔드가 예외 처리 경로에서 펴진 이미지를 꺼내야 해서 화면 코드가 꼬인다.

## 막히면

SW파트.""",
    },
    {
        "title": "web/ Vite + TypeScript 스캐폴드",
        "labels": ["part:sw"],
        "milestone": M10,
        "body": """## 무엇을 해야 하나

`web/` 에 Vite + TypeScript 프로젝트를 만든다. UI 라이브러리와 상태 관리는
쓰지 않는다 — 화면이 하나이므로 CSS와 `useState` 로 충분하다 (설계 문서 §24).

## 완료 조건

`cd web && npm install && npm run dev` 로 화면이 뜨고, 원본·펴진 이미지와
판독 텍스트가 함께 보인다. `web/src/api.ts` 에 응답 타입이 선언돼 있다.

CI에 `npx tsc --noEmit` 과 `npm run build` 단계를 추가한다.

## 왜 지금 만드나

**AI 모델 없이도 화면이 뜨는 상태를 먼저 만들면 프론트엔드 통합 위험이
미리 소진된다.** 가짜 응답으로라도 화면이 붙어 있으면, 나중에 진짜 모델이
나왔을 때 "통합"이라는 힘든 작업 자체가 없어진다.

## 막히면

SW파트. Node 22 LTS 를 쓰고 `web/.nvmrc` 에 고정한다.""",
    },
    {
        "title": "E2E 스모크 테스트 + conftest 픽스처 2개 추가",
        "labels": ["part:sw"],
        "milestone": M10,
        "body": """## 무엇을 해야 하나

`pipeline.run()` 에 이미지를 넣어 `PipelineResult` 가 나오는지 확인하는
테스트를 만든다. 그리고 `tests/conftest.py` 에 아직 없는 픽스처 2개를 추가한다.

- `warped_barcode_bgr` — 곡률 왜곡
- `glared_barcode_bgr` — 반사 효과

둘 다 `wemeet.data.synthesis` 를 쓰므로 그 파일이 생긴 뒤에 만든다.

## 완료 조건

설계 문서 §5의 실패 정책(탐지 0개, 2개 이상, 예산 초과)이 테스트로 확인된다.

## 같이 확인할 것 (중요)

import-linter 계약 중 **아직 검증하지 못한 2개**가 있다. 해당 파일이 없어서
1단계에서 확인할 수 없었다.

- `wemeet/data/` 에서 `import wemeet.sw.pipeline` → **막혀야 한다**
- `wemeet/data/` 에서 `import wemeet.sw.decoding` → **통과해야 한다**

두 번째가 특히 중요하다. **허용해야 하는 예외를 실수로 막아놓은 것을 잡는
검사다.** 이게 막히면 실촬영 정답 확보 작업이 멈춘다.
import-linter 는 존재하지 않는 모듈을 금지 목록에 넣어도 오류를 내지 않으므로
(오타도 조용히 통과한다) 직접 넣어보고 확인해야 한다.

## 막히면

SW파트 + 데이터파트.""",
    },
    {
        "title": "데이터파트 스텁 3개 — download / synthesis / ground_truth",
        "labels": ["part:data"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

`wemeet/data/` 에 파일 3개를 만든다.

| 파일 | 하는 일 |
|---|---|
| `download.py` | HF Hub에서 `downloads/` 로 받아온다 |
| `synthesis.py` | 정상 바코드에 곡률·반사 훼손을 입힌다 (역방향 합성) |
| `ground_truth.py` | 실촬영 이미지의 정답 번호를 디코더로 확보한다 |

## 완료 조건

`uv run python -m wemeet.data.download` 가 동작한다.
README "시작하기" 3번이 이 명령을 안내하고 있으므로 그 문장이 사실이 된다.

## 주의 2가지

**`ground_truth.py` 만 `wemeet.sw.decoding` 을 import할 수 있다.**
`pipeline` 과 `server` 는 못 쓴다 — CI가 막는다.

**합성에 시드를 고정한다.** 데이터가 매번 달라지면 모델 비교가 성립하지 않는다.
`set_seed()` 유틸을 만들어 쓴다 (설계 문서 §11). `schemas.py` 에는 넣지 않는다.

## 막히면

데이터파트.""",
    },
    # ---- 외부 설정 ----
    {
        "title": "Hugging Face 조직 metro-wemeet + 저장소 3개 생성",
        "labels": ["part:data"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

HF 조직을 만들고 저장소 3개를 만든다 (`docs/external/huggingface.md`).

| 저장소 | 종류 | 공개 |
|---|---|---|
| `metro-wemeet/barcode-datasets` | Dataset | public |
| `metro-wemeet/barcode-field` | Dataset | **private** |
| `metro-wemeet/barcode-weights` | Model | public |

## 완료 조건

저장소 3개가 존재하고, `barcode-field` 가 **private** 이다.

## 절대 틀리면 안 되는 것

**`barcode-field` 는 반드시 private.** 실촬영 송장에는 수취인 이름·주소·연락처가
인쇄돼 있다. public으로 만들었다가 나중에 바꿔도, 그 사이에 올라간 이미지는
이미 노출된 것이다. 저장소를 분리해두는 이유가 실수로 섞이지 않게 하는 것이다.

접근 권한은 촬영 담당자와 최종 평가 담당자로 제한한다. 8명 전원에게 줄 필요가 없다.

## 왜 지금 하나

저장소 이름이 `wemeet/data/download.py` 코드에 그대로 박힌다.
이름이 먼저 확정돼 있으면 나중에 코드와 문서를 양쪽 고치는 일이 없다.

## 막히면

데이터파트 + 팀장(조직 생성).""",
    },
    {
        "title": "Weights & Biases 학생 계정 + 팀 + 프로젝트 생성",
        "labels": ["part:ai"],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

W&B 학생 계정을 신청하고 팀과 프로젝트를 만든다 (`docs/external/wandb.md`).

- 팀 이름: `metro-wemeet`
- 프로젝트: `wemeet-barcode` **하나만** 쓴다

## 완료 조건

AI파트 전원이 팀에 들어가 있고, 각자 API 키를 로컬 `.env` 에 넣었다.

## 왜 프로젝트를 하나만 쓰나

Stage별로 나누면 검출과 기하 추정의 추론 시간을 합쳐서 500ms 예산을 확인할 때
두 곳을 오가야 한다.

## 언제까지

**학습을 시작하는 9월 4주차 전까지면 된다.** 지금 급하지 않다.
다만 학생 계정 인증(학교 이메일)에 시간이 걸릴 수 있으니 9월 첫 주에 신청한다.

## 주의

API 키를 커밋하지 않는다. `.env` 는 `.gitignore` 에 있지만
`.env.example` 은 커밋되므로 거기에 실제 키를 쓰지 않는다.

## 막히면

AI파트 + 팀장(팀 생성).""",
    },
    {
        "title": "팀원 8명 Collaborator 초대 + GitHub 아이디 수집",
        "labels": [],
        "milestone": M09,
        "body": """## 무엇을 해야 하나

Settings → Collaborators 에서 팀원 8명을 초대한다.
초대에 GitHub 아이디가 필요하므로 먼저 수집한다.

## 완료 조건

8명이 초대를 수락해 저장소에 push할 수 있다.

## 참고

리뷰어 자동 배정(CODEOWNERS)은 쓰지 않기로 했으므로, 아이디는 초대와
리뷰어 수동 지정에만 쓴다. 파트별 담당은 README 표에 이름으로 적혀 있다.

## 막히면

팀장.""",
    },
]


def api(method: str, path: str, payload: dict | None = None) -> dict | list:
    """GitHub API 호출. 실패하면 상태 코드와 응답 본문을 그대로 보여준다."""
    token = os.environ["GH_TOKEN"]
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read() or b"{}")


def main() -> int:
    if not os.environ.get("GH_TOKEN"):
        print("GH_TOKEN 환경변수가 없습니다.", file=sys.stderr)
        print("사용법: GH_TOKEN=ghp_xxxx uv run python scripts/setup_github.py", file=sys.stderr)
        return 1

    dry = "--dry-run" in sys.argv
    if dry:
        print("[미리보기] 아무것도 만들지 않습니다.\n")

    # 토큰이 유효한지, 쓰기 권한이 있는지 먼저 확인한다.
    # 이슈를 20개 만들다가 중간에 401이 나는 것보다 지금 실패하는 게 낫다.
    try:
        repo = api("GET", f"/repos/{REPO}")
    except urllib.error.HTTPError as e:
        print(f"저장소를 읽을 수 없습니다: HTTP {e.code}", file=sys.stderr)
        print(e.read().decode(errors="replace"), file=sys.stderr)
        return 1
    if not repo.get("permissions", {}).get("push"):
        print("이 토큰에는 쓰기 권한이 없습니다. classic 이면 repo 스코프가 필요합니다.")
        return 1
    print(f"저장소: {repo['full_name']}\n")

    # ---- 라벨 ----
    existing = {label["name"] for label in api("GET", f"/repos/{REPO}/labels?per_page=100")}
    print("라벨")
    for name, color, description in LABELS:
        if name in existing:
            print(f"  = {name} (이미 있음)")
            continue
        if not dry:
            api(
                "POST",
                f"/repos/{REPO}/labels",
                {"name": name, "color": color, "description": description},
            )
        print(f"  + {name}")

    # ---- 마일스톤 ----
    found = api("GET", f"/repos/{REPO}/milestones?state=all&per_page=100")
    numbers = {m["title"]: m["number"] for m in found}
    print("\n마일스톤")
    for title, due_on, description in MILESTONES:
        if title in numbers:
            print(f"  = {title} (이미 있음)")
            continue
        if dry:
            print(f"  + {title}")
            continue
        created = api(
            "POST",
            f"/repos/{REPO}/milestones",
            {"title": title, "due_on": due_on, "description": description},
        )
        numbers[title] = created["number"]
        print(f"  + {title}")

    # ---- 이슈 ----
    # 제목으로 중복을 판단한다. 닫힌 이슈도 포함해서 본다 — 이미 처리한 일을
    # 다시 만들지 않기 위해서다.
    seen = {
        issue["title"]
        for issue in api("GET", f"/repos/{REPO}/issues?state=all&per_page=100")
        if "pull_request" not in issue
    }
    print("\n이슈")
    made = 0
    for issue in ISSUES:
        if issue["title"] in seen:
            print(f"  = {issue['title']} (이미 있음)")
            continue
        payload = {
            "title": issue["title"],
            "body": issue["body"],
            "labels": issue["labels"],
        }
        number = numbers.get(issue["milestone"])
        if number is not None:
            payload["milestone"] = number
        if not dry:
            api("POST", f"/repos/{REPO}/issues", payload)
        made += 1
        print(f"  + {issue['title']}")

    print(f"\n{'만들 이슈' if dry else '만든 이슈'} {made}개")
    if not dry:
        print(f"확인: https://github.com/{REPO}/issues")
        print("\n토큰은 이제 폐기(revoke)해도 됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
