# 기하 추정·보정 파이프라인 설계

- 프로젝트: AI 기반 난인식 바코드 품질 인식 고도화 (WE-Meet, METRO팀)
- 작성일: 2026-08-04
- 상태: **검토 대기**
- 대체 대상: [메인 브랜치 구조 설계](2026-07-30-main-branch-structure-design.md) §5의 Stage 2 정의

---

## 1. 개요

### 무엇을 바꾸나

Stage 2를 **이미지 복원(생성)** 에서 **기하 추정(예측) + 기하 보정(변환)** 으로 바꾼다.

| | 기존 | 변경 후 |
|---|---|---|
| AI의 출력 | 복원된 **이미지** | 얼마나 휘었는지 나타내는 **숫자** |
| 픽셀을 만드나 | **만든다** | 만들지 않는다. 있는 픽셀을 옮긴다 |
| 펴는 주체 | AI 모델 | OpenCV (`remap`) — SW파트 |
| 단계 수 | 3 | 4 |

### 범위

| 포함 | 제외 |
|---|---|
| 새 파이프라인 4단계 정의 | 모델 구조 선정 (`decisions/0003`) |
| `schemas.py` 계약 변경 전문 | 학습 코드 |
| 조기 종료·재시도 루프 설계 | 실제 구현 (2단계 계획서) |
| 반사 대응 방침 | 디코더 선정 (`decisions/0006`, 3종 비교 유지) |
| TPS 구현 방침 | 데이터 수집 절차 (기존 설계 §14 유효) |
| 사전 검증 실험 결과와 측정치 | |
| 시간 예산 재배분 | |

### 안 바뀌는 것

혼란을 줄이기 위해 먼저 못 박는다.

- 의존 방향 `data → sw → ai → schemas` — 그대로. AI 책임이 줄어 경계가 더 깔끔해진다
- `DetectedBarcode` · `DecodeResult` · `PipelineResult` — 그대로
- 파이프라인은 예외를 던지지 않고 실패를 값으로 표현한다는 원칙 — 그대로
- `failure_reason` 세 값(`invalid_input` / `not_detected` / `decode_failed`) — 그대로
- CI · 테스트 · 협업 규칙 · 폴더 구조 — 무관

바뀌는 것은 **`RestoredBarcode` 하나가 `GeometryField` + `RectifiedBarcode` 둘로 갈라지는 것**과 단계 수가 3 → 4가 되는 것이다.

---

## 2. 왜 바꾸나

### 2.1 생성 모델은 틀린 번호를 만들 수 있다

가장 중요한 이유다. 생성 모델이 만든 가짜 바코드는 **체크섬을 우연히 통과할 수 있다.**

| 심볼로지 | 검증 | 무작위로 통과할 확률 |
|---|---|---|
| Code128 | mod-103 체크문자 | 약 **1/103** |
| EAN-13 | mod-10 체크digit | **1/10** |

세관에서 하루 수천 건을 처리하면 **틀린 번호가 조용히 통과하는 사고가 실제로 발생한다.** 이건 못 읽는 것보다 훨씬 나쁘다 — 못 읽으면 사람이 처리하지만, 틀리게 읽으면 화물이 엉뚱한 곳으로 간다. 그리고 사고가 났을 때 원인을 추적할 수단이 없다.

기하 보정은 있는 픽셀만 옮기므로 이 실패 모드가 **구조적으로 존재하지 않는다.**

### 2.2 학습 데이터를 무한정 만들 수 있다

기하 추정은 **정답을 합성으로 정확히 만들 수 있다.** 깨끗한 바코드에 알고 있는 왜곡을 입히면, 정답은 그 왜곡의 역함수 그 자체다. 라벨링이 필요 없다.

생성 복원은 이게 안 된다. "구겨진 이미지 ↔ 펴진 이미지" 쌍이 필요하고, 실촬영으로 만들려면 같은 송장을 구긴 상태와 편 상태로 각각 찍어야 하는데 **픽셀이 정렬되지 않는다.** 9월에 데이터파트가 막힐 위험이 이 변경으로 크게 줄어든다.

### 2.3 3개월에 실현 가능한 규모가 된다

제어점 회귀는 작은 CNN으로 된다. 생성 모델(GAN·diffusion)은 학습이 불안정하고 GPU 시간이 훨씬 많이 들며, GPU 확보 경로도 아직 미정이다(기존 설계 §20).

### 2.4 설명할 수 있다

제어점과 flow map을 화살표로 그리면 "AI가 무엇을 했는지"가 그림 하나로 설명된다. 기획서 7.2가 내세우는 "내부가 블랙박스가 아니다"라는 차별점이 **더 강해진다.**

### 2.5 못 읽을 때 정직하게 실패한다

정보가 소실될 정도로 구겨진 경우, 생성 모델은 그럴듯한 거짓을 만든다. 기하 보정은 "못 읽음"으로 끝나고 노리드 존으로 보낸다. 이게 맞는 동작이다.

---

## 3. 새 파이프라인

```
                          사진 입력
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │ Stage 1 · AI          Detection        │
        │  · 바코드 영역 검출 (OBB)                │
        │  · 기울기 보정 후 크롭                    │
        │  출력: DetectedBarcode                  │
        │  ※ 디코더를 부르지 않는다                 │
        └────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ 파이프라인 판단   │  ← SW. AI 아님
                    │ 1차 디코딩 시도   │
                    └────────┬────────┘
                     읽힘 ───┴──▶ 종료 (약 150ms)
                             │ 안 읽힘
                             ▼
        ┌────────────────────────────────────────┐
        │ Stage 2 · AI    Geometry Estimation    │
        │  · 곡률 / 휘어짐 / 주름 추정              │
        │  출력: GeometryField (숫자만)            │
        │  ※ 이미지를 반환하지 않는다               │
        └────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │ Stage 3 · SW    Geometry Correction    │
        │  · TPS 계수 계산 → flow map → remap     │
        │  · 기존 픽셀만 이동                      │
        │  출력: RectifiedBarcode                 │
        └────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │ Stage 4 · SW           Decode          │
        │  4a 반사 대응 — 바 방향 행 선택·집계       │
        │  4b 디코딩 (0006에서 선정)                │
        │  4c 실패 → 보정 파라미터 바꿔 최대 3회     │
        └────────────────────────────────────────┘
                             │
                  성공 ──────┴────── 실패
                   │                  │
                   ▼                  ▼
              번호 + 이미지        failure_reason
```

재시도는 **Stage 4 → Stage 3으로 되돌아간다.** 보정 파라미터를 바꿔 다시 펴고 다시 읽는다. 루프의 주인은 `pipeline.py`다.

### 단계별 담당과 입출력

| 단계 | 담당 | 입력 | 출력 |
|---|---|---|---|
| 1 검출 | AI | 원본 BGR | `DetectedBarcode` |
| 1.5 1차 디코딩 | **SW (파이프라인)** | `RectifiedBarcode(field=None)` | `DecodeResult` |
| 2 기하 추정 | AI | `DetectedBarcode` | `GeometryField` |
| 3 기하 보정 | SW | `DetectedBarcode` + `GeometryField` | `RectifiedBarcode` |
| 4 디코딩 | SW | `RectifiedBarcode` | `DecodeResult` |

### 1차 디코딩을 Stage 1에 넣지 않는 이유

원안은 Stage 1(AI)이 초기 decode를 시도했다. 그러면 `wemeet/ai/detection.py`가 `wemeet.sw.decoding`을 import해야 하고, **의존 규칙 위반으로 CI가 막는다.**

```
wemeet.ai is not allowed to import wemeet.sw
```

"먼저 읽어보고 안 되면 보정한다"는 **조율 판단**이므로 지휘자(파이프라인)의 일이다. 기존 설계가 정한 "판단은 파이프라인에 모은다"는 원칙 그대로다. 이렇게 하면 AI는 "휘어짐만 예측"하는 순수한 역할로 남고 단독 실험도 계속 가능하다.

---

## 4. 계약 변경 — `wemeet/schemas.py`

### 추가

```python
@dataclass
class GeometryField:
    """Stage 2 (기하 추정) → Stage 3 (기하 보정)

    얼마나 휘었는지만 담는다. 이미지 필드가 없으므로
    AI 가 픽셀을 만들어 반환할 방법이 구조적으로 없다.

    좌표계 (반드시 지킬 것):
      · 기준은 DetectedBarcode.crop_bgr_uint8 이다. 원본 이미지가 아니다.
      · 0.0~1.0 정규화 좌표다. 픽셀이 아니다. (x, y) 순서다.
      · dst = 펴진 뒤의 격자 위치, src = 그 내용이 지금 있는 위치.
        이 방향을 뒤집으면 왜곡이 두 번 적용된다 — §8.5 참고.
    """

    control_points_dst_norm: np.ndarray   # (N, 2) 펴진 격자. 보통 규칙적인 격자
    control_points_src_norm: np.ndarray   # (N, 2) 휜 이미지에서의 위치
    method: str                            # "tps" | "perspective"
    confidence: float                      # 0.0 ~ 1.0

    def __post_init__(self) -> None:
        assert self.control_points_dst_norm.shape == self.control_points_src_norm.shape
        assert self.control_points_dst_norm.ndim == 2
        assert self.control_points_dst_norm.shape[1] == 2
        assert len(self.control_points_dst_norm) >= 4      # §8.2: 8개 권장, 최소 4개
        assert self.method in ("tps", "perspective")
        assert 0.0 <= self.confidence <= 1.0


@dataclass
class RectifiedBarcode:
    """Stage 3 (기하 보정) → Stage 4 (디코딩)

    OpenCV 가 픽셀을 이동시킨 결과. 새 픽셀은 없다.
    """

    image_gray_uint8: np.ndarray           # 흑백 단일 채널, shape (H, W)
    source: DetectedBarcode
    field: GeometryField | None            # None = 보정하지 않고 통과 (1차 시도)

    def __post_init__(self) -> None:
        assert self.image_gray_uint8.dtype == np.uint8
        assert self.image_gray_uint8.ndim == 2
```

### 폐기

```python
# @dataclass
# class RestoredBarcode:      ← 폐기 (2026-08-04)
#     image_gray_uint8: np.ndarray
#     source: DetectedBarcode
```

`RestoredBarcode`는 "AI가 만든 복원 이미지"를 뜻했다. 이제 AI는 이미지를 만들지 않는다. 이름이 남아 있으면 누군가 다시 생성 모델을 붙인다.

`RectifiedBarcode`가 자리를 대신하지만 **의미가 다르다** — `field`를 통해 "무엇을 근거로 어떻게 옮겼는지"를 함께 들고 다닌다. `field=None`은 "옮기지 않았다"는 뜻이다.

### `DecodeResult` · `PipelineResult` 변경

필드는 그대로 두고 해석만 조정한다.

| 필드 | 기존 의미 | 새 의미 |
|---|---|---|
| `degraded` | 복원 모델이 터져서 원본 크롭을 씀 | **보정하지 못하고** 원본 크롭으로 디코딩함 |
| `stage_ms` | `detect` / `restore` / `decode` / `retry` | `detect` / `decode_first` / `estimate` / `warp` / `decode` / `retry` |

`PipelineResult.restored: RestoredBarcode | None` 은 `rectified: RectifiedBarcode | None` 으로 이름을 바꾼다. 이 필드는 화면에 "펴진 모습"을 보여주는 데 쓰이므로 반드시 유지한다.

---

## 5. 스테이지 함수 시그니처

```python
# wemeet/ai/detection.py            @AI파트
def detect(image_bgr: np.ndarray) -> list[DetectedBarcode]: ...

# wemeet/ai/geometry.py             @AI파트   ← restoration.py 를 대체
def estimate_geometry(target: DetectedBarcode) -> GeometryField: ...
#   이미지를 반환하지 않는다. 반환형에 이미지 필드가 없다.

# wemeet/sw/rectify.py              @SW파트   ← 새 파일
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

파일 이름 변경: `ai/restoration.py` → `ai/geometry.py`. `sw/rectify.py`가 새로 생긴다.

`interpolation`을 인자로 노출하는 이유는 재시도에서 이 값을 바꾸기 때문이다(§6).

---

## 6. 조기 종료와 재시도 루프

```python
# wemeet/sw/pipeline.py
def run(image_bgr):
    ms = {}

    found = _timed(ms, "detect", detect, image_bgr)
    if not found:
        return _fail(image_bgr, "not_detected", ms)
    best = max(found, key=lambda b: b.confidence)

    # ── 1차 시도: 기울기 보정만으로 읽히는지 본다 ──────────────
    plain = RectifiedBarcode(
        image_gray_uint8=cv2.cvtColor(best.crop_bgr_uint8, cv2.COLOR_BGR2GRAY),
        source=best,
        field=None,                      # None = 보정하지 않았다
    )
    result = _timed(ms, "decode_first", decode, plain)
    if result.text is not None:
        return _ok(image_bgr, best, plain, result, ms, len(found))

    # ── 안 읽혔다. 기하 추정으로 넘어간다 ─────────────────────
    field = _timed(ms, "estimate", estimate_geometry, best)

    if field.confidence < 0.3:           # 추정을 신뢰할 수 없다
        result.degraded = True           # 보정 없이 실패로 종료
        return _fail_with(image_bgr, best, plain, result, ms, len(found))

    # ── 보정 + 디코딩, 실패하면 파라미터를 바꿔 최대 3회 ────────
    RETRIES = [
        dict(interpolation=cv2.INTER_CUBIC),     # 1회: 기본
        dict(interpolation=cv2.INTER_LANCZOS4),  # 2회: 더 선명한 보간
        dict(interpolation=cv2.INTER_LINEAR),    # 3회: 부드럽게 (노이즈가 심할 때)
    ]
    rectified = None
    for i, opts in enumerate(RETRIES):
        rectified = _timed(ms, "warp", apply_field, best, field, **opts)
        result = _timed(ms, "decode" if i == 0 else "retry", decode, rectified)
        result.retry_count = i
        if result.text is not None:
            return _ok(image_bgr, best, rectified, result, ms, len(found))

    result.failure_reason = "decode_failed"
    return _fail_with(image_bgr, best, rectified, result, ms, len(found))
```

### 재시도가 의미를 갖게 된다

기존 설계의 재시도는 디코더 파라미터만 바꿨고 실제로 바꿀 것이 별로 없었다. 이제는 **보정 자체를 다르게 해서 다시 시도**한다.

| 회차 | 바꾸는 것 | 근거 |
|---|---|---|
| 1 | `INTER_CUBIC` | 기본. 대부분 여기서 끝난다 |
| 2 | `INTER_LANCZOS4` | 경계가 더 선명해진다. 얇은 바에 유리 |
| 3 | `INTER_LINEAR` | 부드럽다. 노이즈가 심할 때 유리 |

보간법 외에 **제어점 스케일 조정**(예: 변위를 0.9배·1.1배로 축소·확대)도 후보다. 실측 후 `decisions/0006`에 순서를 확정한다.

---

## 7. 반사 대응 — Stage 4a

### 왜 필요한가

기하 보정은 픽셀을 **이동**시킨다. 반사로 하얗게 포화된 영역은 정보가 없으므로 옮겨도 여전히 하얗다. 기획서와 README가 문제를 "구겨짐 **그리고** 반사"로 정의하므로, 기하 보정만으로는 **문제의 절반만 다루게 된다.**

### 생성 없이 처리하는 방법

```
기하 보정 후 (바가 수직으로 정렬됨)
   → 바 높이 방향으로 여러 행을 읽는다
   → 반사로 포화된 행(예: 평균 밝기가 임계 이상, 분산이 임계 이하)은 버린다
   → 남은 행으로 열마다 값을 정한다 (중앙값 또는 대비 최대 행 선택)
   → 1D 신호 하나를 만들어 디코딩
```

**1D 바코드는 바 높이 방향으로 정보가 중복된다.** 반사는 보통 바의 일부 높이만 덮으므로, 살아 있는 행만 골라 쓰면 읽힌다. **있는 픽셀 중에서 고르는 것**이므로 hallucination이 없다.

기하 보정이 **선행 조건**이다 — 바가 휘어 있으면 "같은 열"이 정의되지 않아 행을 합칠 수 없다. 즉 이번 변경이 이 기법을 가능하게 만든다.

### 한계

반사가 **전체 높이를 덮은 열**은 정보가 진짜로 없다. 어떤 방법으로도 정직하게 복구할 수 없으므로 실패로 보고한다. 이 경계를 문서에 남겨두면 11월 평가에서 "왜 이건 못 읽나"에 답할 수 있다.

### 2D 코드(QR)는 이 기법을 쓸 수 없다

QR은 높이 방향 중복이 없다. `decisions/0001`에서 QR을 범위에 넣는다면 반사 대응은 별도 설계가 필요하다. 현재 0001은 미결정이다.

---

## 8. 사전 검증 실험

설계를 확정하기 전에 **실제로 되는지 측정했다.** 아래는 모두 실측값이다.

### 8.1 무엇이 1D 바코드를 깨뜨리는가

`python-barcode`로 Code128 `WEMEET0001`(376×280px, 모듈 폭 약 5.5px)을 렌더링하고 `zxing-cpp`로 판독했다.

| 왜곡 | 결과 |
|---|---|
| 없음 | 읽힘 (기준선) |
| **수직 방향 사인파 휘어짐, 진폭 4~32px** | **전부 읽힘 — 깨지지 않는다** |
| 가로 방향 원통 압축 20° | 읽힘 |
| 가로 방향 원통 압축 **35° 이상** | **실패** |

**수직 변형은 무해하다.** 바가 수직이라 위아래로 밀려도 가로 방향 바 패턴이 변하지 않는다. 정보를 파괴하는 것은 **가로 방향(바 폭 방향) 비선형 압축**이다.

> **설계에 미치는 영향**: 추정해야 할 기하는 주로 **가로 방향**이다. 2D flow 전체를 정확히 맞출 필요가 없다. 문제의 차원이 예상보다 낮다.
>
> 그리고 **조기 종료가 자주 성공할 것**으로 예상된다. 수직으로만 휜 건은 기울기 보정만으로 읽힌다.

### 8.2 제어점 몇 개가 필요한가

35° 압축된 이미지를 알고 있는 제어점 대응으로 TPS 보정한 뒤 판독했다.

| 감긴 각도 | 3×2=6 | 4×2=8 | 6×3=18 | 8×3=24 | 12×3=36 |
|---|---|---|---|---|---|
| 35° | 실패 | **읽힘** | 읽힘 | 읽힘 | 읽힘 |
| 50° | 실패 | **읽힘** | 읽힘 | 읽힘 | 읽힘 |
| 70° | 실패 | **읽힘** | 읽힘 | 읽힘 | 읽힘 |
| 84° | 실패 | 실패 | 읽힘 | 실패 | 실패 |

**제어점 8개(4×2)면 35~70°가 전부 복원된다.** 모델 출력이 16개 숫자(8점 × 2좌표)면 충분하다는 뜻이다. 매우 작은 출력이다.

기본안을 **4×3 = 12점**으로 잡는다. 8점으로 충분하지만 세로 방향 주름에 여유를 둔다.

### 8.3 제어점이 얼마나 정확해야 하는가

6×3=18점에 가우시안 오차를 넣고 각 조건 15회 시도했다.

| 제어점 오차 σ | 35° | 50° | 70° |
|---|---|---|---|
| 0 px | 100% | 100% | 100% |
| 0.5 px | 100% | 100% | 100% |
| 1 px | 100% | 100% | 100% |
| 2 px | 100% | 100% | 100% |
| 3 px | 100% | 100% | 100% |
| 5 px | 100% | 100% | 100% |
| 8 px | 100% | 100% | 87% |

**모듈 폭이 5.5px인데 제어점 오차 8px까지 읽힌다.** 왜곡이 저주파(매끄러움)라 TPS가 노이즈에도 전체 추세를 잡고, 남는 잔차는 디코더가 감당하는 완만한 변형이기 때문이다.

> **설계에 미치는 영향**: 기하 추정의 **정확도는 병목이 아니다.** 모델이 픽셀 단위로 정밀할 필요가 없다. 작은 CNN으로 충분할 근거가 된다.
>
> **다만 이 수치는 낙관적 상한이다.** 합성 이미지, 노이즈 없음, 반사 없음, 검출·기울기 보정이 완벽하다는 조건이다. 실제로는 블러·노이즈·반사가 겹치므로 목표는 더 엄격하게 잡아야 한다. `decisions/0005`의 목표 수치는 실촬영 베이스라인 측정 후 정한다.

### 8.4 복원 가능한 한계

84°는 제어점을 늘려도 실패한다. **추정 오차 문제가 아니라 정보 소실이다.**

가장 압축된 지점(가장자리)의 압축률은 대략 `cos θ`다.

| 각도 | 압축률 | 가장자리 모듈 폭 | 판정 |
|---|---|---|---|
| 35° | 0.82 | 4.5 px | 복원됨 |
| 50° | 0.64 | 3.5 px | 복원됨 |
| 70° | 0.34 | 1.9 px | 복원됨 |
| 84° | 0.10 | **0.6 px** | **복원 불가** |

**복원 가능 조건: 가장 압축된 지점에서 모듈 폭이 약 1~2px 이상 남아 있을 것.** 1px 이하로 압축되면 샘플링 한계 아래이므로 어떤 방법으로도 되살릴 수 없다.

이 기준은 **촬영 조건과 카메라 해상도 요구사항으로 이어진다.** 화물이 얼마나 심하게 감겨 있을 수 있는지를 현장에서 확인하고, 그 각도에서 모듈이 2px 이상 남도록 해상도를 정해야 한다. `decisions/0004`에 반영한다.

### 8.5 실험 중 발견한 함정

**src / dst 방향을 뒤집으면 왜곡이 두 번 적용되는데, 약한 왜곡에서는 우연히 읽힌다.** 실험 1차 시도에서 이 실수를 했고, 35°에서는 100% 읽혀서 정상으로 보였다. 50°에서 0%, 2px 오차에서 93%처럼 **결과가 비단조적으로 나오는 것**을 보고 발견했다.

```python
# 틀림 — col 을 그대로 쓰면 "펴진 -> 휜" 방향이다
src_x = np.interp(dst_x, np.arange(w), col)

# 맞음 — 역함수를 써야 "휜 -> 펴진" 이 된다
src_x = np.interp(dst_x, col, np.arange(w))
```

> **설계에 미치는 영향**: 계약 docstring에 좌표계와 방향을 못 박았다(§4). 그리고 **검증 기준에 "왜곡을 두 번 적용하면 실패하는지" 테스트를 넣는다**(§15-6). 결과가 비단조적이면 방향을 먼저 의심할 것.

### 재현 절차

`python-barcode`로 Code128 `WEMEET0001` 렌더링 → `cv2.remap`으로 원통 압축(각도 θ, 호 길이가 원본 폭과 같게 `r = (w-1)/2/θ`) → 제어점 격자에서 역함수 보간 → §9의 `tps_flow` → `cv2.remap(INTER_CUBIC, BORDER_REPLICATE)` → `zxingcpp.read_barcode`. 난수 시드 42.

---

## 9. TPS 구현 방침 — 직접 구현한다

### OpenCV contrib를 쓰지 않는다

`cv2.createThinPlateSplineShapeTransformer`는 **`opencv-python`(main)에 없다.** 실측 확인했다.

| 패키지 | TPS | 비고 |
|---|---|---|
| `opencv-python` 5.0.0 | **없음** | 현재 `pyproject.toml`에 있는 것 |
| `opencv-contrib-python` 5.0.0 | 있음 | 다운로드 78 MB |

contrib로 바꾸지 않는 이유:

1. **78MB가 늘어난다.** 8명이 각자 받는다
2. `cv2.barcode`를 쓰려고 main 패키지를 선택한 기존 결정(설계 §8)을 뒤집게 된다
3. contrib API가 `DMatch` 리스트를 요구하고 **이미지를 직접 warp해서 flow map을 돌려주지 않는다.** 우리는 flow map을 화면에 보여주고 싶다(§2.4)

### numpy로 계산한다

TPS는 선형 시스템 하나를 푸는 것이다. 30줄이면 된다. **아래 코드는 실험에서 검증된 것이다** — SW파트는 이것을 그대로 쓰면 된다.

```python
def tps_flow(
    src: np.ndarray,           # (N, 2) 휜 이미지에서의 제어점 위치 (픽셀)
    dst: np.ndarray,           # (N, 2) 펴진 격자에서의 제어점 위치 (픽셀)
    shape: tuple[int, int],    # 출력 (H, W)
    reg: float = 0.0,          # 정규화. 추정이 불안하면 0.1~1.0
) -> tuple[np.ndarray, np.ndarray]:
    """제어점 대응으로 dense flow map 을 만든다.

    반환값을 cv2.remap 에 넣으면 펴진 격자의 각 픽셀이
    휜 이미지의 어디서 값을 가져올지가 정해진다.
    """
    n = len(dst)
    d = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(d > 0, d**2 * np.log(d**2), 0.0)
    k += reg * np.eye(n)

    p = np.hstack([np.ones((n, 1)), dst])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T

    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    dg = np.linalg.norm(grid[:, None, :] - dst[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ug = np.where(dg > 0, dg**2 * np.log(dg**2), 0.0)
    pg = np.hstack([np.ones((len(grid), 1)), grid])

    out = []
    for axis in (0, 1):
        b = np.concatenate([src[:, axis], np.zeros(3)])
        coef = np.linalg.solve(a, b)
        out.append((ug @ coef[:n] + pg @ coef[n:]).reshape(h, w).astype(np.float32))
    return out[0], out[1]      # map_x, map_y
```

적용:

```python
map_x, map_y = tps_flow(src_px, dst_px, (h, w))
rectified = cv2.remap(
    crop_gray, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
)
```

`method == "perspective"` 인 경우는 `cv2.getPerspectiveTransform` + `cv2.warpPerspective`를 쓴다. 제어점 4개일 때만 유효하다.

### 성능

380×280 이미지, 제어점 12개에서 격자 계산이 지배적이다. 실험에서 한 번에 수십 ms 수준이었다. **시간 예산 20ms를 맞추려면 최적화가 필요할 수 있다** — 격자를 1/4로 계산하고 `cv2.resize`로 늘리는 방법이 있다. 2단계 구현 시 실측하고, 넘으면 이 방법을 쓴다.

---

## 10. 기하 추정 모델 후보 — `decisions/0003` 갱신

`0003-복원-모델-선정.md` → `0003-기하-추정-모델-선정.md`로 바꾼다. 후보가 완전히 달라진다.

| 후보 | 출력 | 장점 | 위험 |
|---|---|---|---|
| **제어점 회귀 + TPS** (기본안) | 4×3=12점 × 2좌표 = 24개 숫자 | 매끄러움이 구조적으로 보장 → **픽셀이 찢어지지 않음.** 출력이 작아 학습이 쉽다 | 급격한 주름 표현에 한계 |
| Dense flow map | (H, W, 2) | 주름까지 표현 | smoothness 정규화 필요. 노이즈 시 찢어짐 |
| 4점 호모그래피 | 8개 숫자 | 가장 단순. **베이스라인으로 반드시 측정할 것** | 평면 왜곡만. 원통은 못 다룸 |

§8.2가 8점으로 35~70°가 복원됨을 보였으므로 **제어점 회귀로 시작한다.** 부족하면 dense flow로 올리고, 그 비교가 0003의 내용이 된다.

백본은 미정이다. 출력이 24개 숫자뿐이므로 무거운 모델이 필요 없다 — ResNet18·MobileNet 수준에서 시작한다.

---

## 11. 시간 예산

| 단계 | 기존 | 새 배분 |
|---|---|---|
| 검출 + 기울기 보정 | 100 | 110 |
| **1차 디코딩** (조기 종료 지점) | — | 40 |
| 기하 추정 (AI) | 250 | 150 |
| 기하 보정 (`tps_flow` + `remap`) | — | 20 |
| 디코딩 + 재시도 최대 3회 | 50 + 100 | 150 |
| 여유 | — | 30 |
| **합계** | **500** | **500** |

**조기 종료되면 150ms에 끝난다.** §8.1에서 수직 변형이 무해함을 확인했으므로 이 경로를 타는 비율이 상당할 것으로 예상된다. 평균 지연이 기존안보다 개선된다 — 보고서에 쓸 수 있는 실질적 결과다.

0.5초는 재시도를 포함한 전체 경로 기준이고, 초과해도 중단하지 않고 기록만 한다는 원칙은 그대로다.

---

## 12. 파트별 책임 변경

| 파트 | 전 | 후 |
|---|---|---|
| **AI** | 검출 + **이미지 복원** | 검출 + **기하 추정**. 이미지를 만들지 않는다 |
| **SW** | 디코딩 + 연결 + 서버 + 화면 | 여기에 **기하 보정 + 반사 전처리** 추가 |
| **데이터** | 훼손↔정상 이미지 **쌍** | **왜곡 + 그 정답 파라미터** — 합성으로 무한 생성 |

**데이터파트가 가장 편해진다.** 깨끗한 바코드에 알고 있는 왜곡을 입히면 정답이 곧 그 역함수다. 라벨링이 필요 없다. 다만 §8.4에 따라 **어느 각도까지 합성할지**를 정해야 한다 — 모듈 폭이 1px 이하가 되는 왜곡은 학습에 넣어도 배울 수 없다.

**SW파트 부담이 늘어난다.** `rectify.py`와 반사 전처리가 추가된다. 다만 둘 다 학습이 필요 없는 결정론적 코드이고, TPS 구현은 §9에 검증된 코드가 있으므로 난이도는 낮다.

---

## 13. 합성 데이터 생성

`wemeet/data/synthesis.py`가 만들 것이 달라진다.

```
깨끗한 바코드 렌더링 (python-barcode)
   → 왜곡 파라미터를 뽑는다 (감긴 각도, 주름 위치·세기, 기울기)
   → cv2.remap 으로 왜곡을 적용한다
   → 저장: (왜곡된 이미지, 제어점 정답, 정답 번호)
```

정답 제어점은 왜곡을 만들 때 쓴 매핑의 **역함수를 제어점 격자에서 평가**한 것이다. §8.5의 방향 함정을 여기서도 주의해야 한다.

같은 시드로 같은 데이터가 나오도록 시드를 고정한다(기존 설계 §11).

**렌더링할 때 넣은 번호가 곧 정답이므로** 디코딩 성공/실패를 자동 판정할 수 있다. 이 성질은 기존 설계와 동일하게 유지된다.

---

## 14. 영향받는 문서와 코드

| 대상 | 무엇을 |
|---|---|
| `wemeet/schemas.py` | `RestoredBarcode` 폐기, `GeometryField`·`RectifiedBarcode` 추가. **코드가 있는 유일한 파일** |
| `tests/test_schemas.py` | 새 계약 검증 추가 (§16) |
| `docs/decisions/0003` | 「복원 모델」 → 「기하 추정 모델」. 후보 교체 |
| `docs/decisions/0004` | §8.4의 해상도 요구사항 추가 |
| `docs/decisions/0005` | 목표 수치 산정 시 §8.3의 상한 성격 명시 |
| `docs/architecture.md` | 4단계, 시간 예산, 코드 흐름 절 |
| `docs/parts/ai.md` | 「복원」 → 「기하 추정」. 픽셀을 만들지 않음을 명시 |
| `docs/parts/sw.md` | `rectify.py`, 반사 전처리 추가 |
| `README.md` | 3단계 그림 → 4단계 |
| `docs/onboarding.html` | 02 · 06 · 08장 |
| 기존 설계 문서 §5 | 폐기 표시 + 이 문서로 연결 |
| `pyproject.toml` | 변경 없음 — contrib를 쓰지 않기로 했으므로 |

**아직 스텁 코드가 하나도 없어서 지금이 가장 싼 시점이다.** 10월에 바꾸면 세 파트 코드를 다 고쳐야 한다.

---

## 15. 검증 기준

1. `GeometryField`에 잘못된 값을 넣으면 거부된다 — 제어점 개수 3개, `method="restore"`, `confidence=1.5`
2. `RectifiedBarcode`에 3채널 이미지를 넣으면 거부된다
3. `field=None`인 `RectifiedBarcode`가 만들어진다 (1차 디코딩 경로)
4. §9의 `tps_flow`가 **항등 대응**(src == dst)에서 항등 매핑을 만든다 — `map_x[y, x] ≈ x`
5. 합성으로 35° 압축한 바코드를 정답 제어점으로 보정하면 **디코딩이 성공한다** (§8.2 재현)
6. **src/dst를 뒤집으면 디코딩이 실패한다** — 방향 함정이 테스트로 잡힌다 (§8.5)
7. `uv run lint-imports`가 통과한다. `ai/geometry.py`가 `wemeet.sw`를 부르지 않는다
8. 일부러 `ai/geometry.py`에 `import wemeet.sw.rectify`를 넣으면 CI가 막는다

5·6번이 이 설계의 핵심을 지키는 테스트다. 4번은 TPS 구현이 맞는지 가장 싸게 확인하는 방법이다.

---

## 16. 위험과 미결정

| 항목 | 내용 | 담당 / 기한 |
|---|---|---|
| **반사 대응의 실효** | §7은 원리는 맞지만 실촬영에서 얼마나 통할지 미검증이다. 반사가 전체 높이를 덮으면 실패한다 | SW파트 / 9월 2주차 실측 |
| **주름(fold)** | §8은 원통(매끄러운 곡률)만 시험했다. 접힌 주름은 불연속이라 TPS가 표현하지 못할 수 있다 | AI파트 / 0003 실험 |
| **보정 20ms** | §9 성능 참고 — 현재 구현은 더 걸릴 수 있다. 격자 축소 최적화가 필요할 수 있다 | SW파트 / 2단계 |
| **QR 범위** | 2D 코드는 §7의 반사 대응을 쓸 수 없다. `decisions/0001` 미결정 | 팀장 / 9월 1주차 |
| **촬영 해상도** | §8.4의 "모듈 2px 이상" 기준을 현장 조건에 대입해야 한다 | 데이터파트 / 9월 1주차 |
| **AI 추정 실패 시 임계값** | §6의 `confidence < 0.3`은 근거 없는 초기값이다 | AI파트 / 0003 확정 후 |

---

## 17. 기존 설계 문서 처리

[2026-07-30 설계 문서](2026-07-30-main-branch-structure-design.md)의 §5(스테이지 간 계약) 중 `RestoredBarcode`와 Stage 2 정의를 **폐기 표시하고 이 문서로 연결한다. 지우지 않는다.**

`docs/decisions/README.md`가 세운 원칙을 따르는 것이다.

> 나중에 뒤집히면 지우지 말고 아래에 추가한다 — **왜 바뀌었는지가 정보다**

11월 보고서에서 "왜 생성 복원을 쓰지 않았나"는 이 프로젝트의 핵심 설명이 된다. 그때 근거가 되는 것이 §2와 §8이다.

CODEOWNERS 폐기(§6)를 같은 방식으로 처리한 선례가 있다.
