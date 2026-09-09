# 합성 데이터 파이프라인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 레시피(시드 + 파라미터)에서 구겨진 Code128 이미지와 정답 제어점을 재현 가능하게 생성하고, 실제 디코딩으로 구간을 라벨링해 §7 비중(목표 50 / 불가·관측가능 35 / 1차성공 15)으로 학습·평가 세트를 채운다.

**Architecture:** 높이장의 **기울기**(`z_x`, `z_y`)를 단일 통화로 쓴다. 기하(압축률 `m`)와 광학(법선)이 같은 기울기에서 나오므로 "반사 위치 ↔ 곡률" 상관이 공짜로 물리적으로 일관된다. 이미지는 굽지 않고 레시피만 JSONL 로 저장하고, 학습 때 워커가 즉석 생성한다. 제어점은 변환식으로 옮기지 않고 **밀집 대응장 `G`(펴진→관측)** 를 이미지와 나란히 들고 다니다 맨 끝에서 한 번 샘플링한다.

**Tech Stack:** Python 3.12, numpy≥2.0, opencv-python≥4.8, python-barcode≥0.15, zxing-cpp≥2.2, pytest, uv

**Spec:** `docs/superpowers/specs/2026-08-28-synthesis-design.md` — **§6·§7 은 2026-09-08 개정판**을 볼 것. 실측치 원문은 `docs/experiments/2026-08-28-synthesis-spike/RESULTS.md`.

**이식 원본:** `docs/experiments/2026-08-28-synthesis-spike/code/` — `common.py`(기하·렌더·tps), `13_augment_G.py`(대응장 증강), `17_eval_set.py`(광학·레시피·라벨링 루프). **검증된 코드지만 실험 스크립트다.** 편차 두 곳은 이식하면서 고친다 (아래 Global Constraints).

---

## Global Constraints

이 절의 규칙은 **모든 태스크의 요구사항에 암묵적으로 포함된다.**

### 좌표 규약 — 틀리면 거꾸로 펴진다

| 항목 | 규칙 |
|---|---|
| 단위 | **0.0 ~ 1.0 정규화.** 픽셀 아님 |
| 원점 | `0.0` = 첫 픽셀 중심, `1.0` = 마지막 픽셀 중심. 픽셀 환산은 **`w−1`, `h−1` 을 곱한다** |
| 순서 | `(x, y)`. `y` 는 **아래로** 증가 |
| 기준 | 최종 관측 크롭의 **원본 크기** |
| 배열 순서 | 행 우선 (위쪽 줄부터, 각 줄은 왼쪽부터) |
| 방향 | `dst` = 펴진 격자, `src` = 지금 있는 위치 |
| `dst` 범위 | `0.0 ~ 1.0` |
| `src` 범위 | `−0.5 ~ 1.5` — 회전 잔차를 모델이 같이 펴므로 크롭 밖을 가리킬 수 있다 |

### 절대 틀리면 안 되는 네 줄

```python
s = np.cumsum(1.0 / m, axis=1)          # cumsum(m) 이 아니다. 실측 오차 0.24px vs 35.11px
I = base * (ka + kd * ndl) + ks * spec  # 정반사는 더하기. 곱하면 검은 막대가 안 지워진다
z_x *= min(1.0, S_max / abs(z_x).max()) # clip(m) 금지. clip 은 법선각을 11.32° 어긋나게 한다
G[v, u] = (x_obs, y_obs)                # 방향은 펴진→관측. F 로 들면 회전이 2D 역문제가 된다
```

### 고정 상수 (샘플마다 안 뽑는다)

| | 값 | 근거 |
|---|---|---|
| `k_a` (밑기본 밝기) | **0.35** | §4 |
| `k_d` (그늘 세기) | **0.50** | §4 |
| `c_min` (판독 하한) | **1.0 px** | §3. 합성 실측 하한 0.8px 에 맞춘 값 |
| `persistence` (옥타브 감쇠) | **0.7** | §3. 목표 구간 10% 로 최대 |
| 제어점 격자 | **`n_x=6`, `n_y=3`** | §6 |
| `G` 해상도 | **`n_v=33`, `n_u=513`** | §5. 크롭 크기와 무관하게 고정 |
| 관측 높이 `H_OBS` | **220 px** | 스파이크 전체가 이 값으로 측정됐다 |
| 여백 범위 | **−2% ~ +15%** | §5. −6% 는 start/stop 이 잘려 못 읽는다 |

### 스파이크 코드의 편차 두 곳 — 이식하면서 고친다

| 스파이크 | 고칠 것 | 왜 |
|---|---|---|
| `budget()` 이 `sqrt((d_m0/2)**2 - 1)` — `c_min` 이 2.0 으로 하드코딩 | `c_min` 을 인자로, 기본값 **1.0** | §3. `c_min=2.0` 이면 `d_m0 < 2.0` 에서 정의되지 않아 저해상도 절반이 표현 불가가 된다 |
| `grad_crumple` 이 `per = slope / octaves` — 옥타브마다 균등(persistence 1.0) | 가중치 **`0.7**j`** 를 쓰고 합이 `slope` 가 되게 정규화 | §3 실측: persistence 1.0 은 목표 구간 5%, 0.7 은 10% |

### 의존 규칙

```
wemeet/data/ 는 wemeet.sw.rectify 를 import 하지 않는다
```

라벨링에 필요한 "펴기" 는 **`scripts/label_recipes.py` 안에 라벨링 도구로** 둔다.
`scripts/` 와 `tests/` 는 의존 규칙 밖이다 (§1). SW파트가 `wemeet/sw/rectify.py` 를 내면
`import` 로 갈아끼운다 — Task 8 에 그 자리를 만들어 둔다.

### 산출물

이미지를 굽지 않는다. `data/recipes/*.jsonl` 이 데이터의 정본이고 커밋한다.
평가 세트만 이미지로 굽고 `data/eval/` 은 `.gitignore` 다.

---

## File Structure

```
wemeet/data/
  render.py      깨끗한 Code128 렌더링 (python-barcode 경계를 여기 가둔다)
  surface.py     레시피 파라미터 → 높이장 기울기 (z_x, z_y).  §3 수식의 집
  warp.py        기울기 → 매핑 → 관측 이미지 + 대응장 G       ← 방향 함정이 사는 곳
  optics.py      Blinn-Phong 음영·정반사 + 포화 화소 비율
  augment.py     geometric(여백·회전, G 동반) / photometric(블러·노이즈·JPEG)
  synthesis.py   Recipe 정의 · 버킷별 샘플링 · 조립 진입점 · JSONL 입출력

scripts/
  label_recipes.py   구간 라벨링 · 비중 채우기 · 평가 세트 추출 (+ 라벨링용 펴기)

tests/
  test_render.py  test_surface.py  test_warp.py  test_optics.py
  test_augment.py  test_synthesis.py  test_labeling.py
```

`warp.py` 를 따로 떼는 이유가 크다. 이 프로젝트에서 가장 비싼 버그가 정확히 거기 한 줄이었고,
**약한 왜곡에서는 우연히 성공해서 안 잡힌다.** 한 파일에 가둬야 §10 검증 셋 하나로 지킬 수 있다.

`surface.py` 는 왜곡 종류가 늘어도 인터페이스가 안 변한다 — 전부 기울기를 만드는 함수다.

---

## Task 1: `render.py` — 모듈 폭이 정확한 깨끗한 바코드

**Files:**
- Create: `wemeet/data/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `render_clean(text: str, module_px: float, height_px: int = 220) -> np.ndarray` — uint8 2D 그레이스케일. 가장 좁은 막대의 폭이 `module_px` 와 일치한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_render.py`:

```python
import numpy as np
from wemeet.data.render import render_clean


def _narrowest_black_run(img: np.ndarray) -> int:
    """중간 행에서 검은 픽셀이 연속하는 가장 짧은 구간의 길이."""
    row = img[img.shape[0] // 2] < 128
    runs, current = [], 0
    for pixel in row:
        if pixel:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return min(runs)


def test_render_clean_shape_and_dtype():
    img = render_clean("WEMEET0001", module_px=4.0, height_px=220)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape[0] == 220


def test_module_width_matches_request():
    img = render_clean("WEMEET0001", module_px=4.0)
    assert abs(_narrowest_black_run(img) - 4.0) <= 0.5


def test_module_width_scales():
    narrow = render_clean("WEMEET0001", module_px=2.0)
    wide = render_clean("WEMEET0001", module_px=6.0)
    assert wide.shape[1] > narrow.shape[1] * 2.5
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wemeet.data.render'`

- [ ] **Step 3: 최소 구현**

`wemeet/data/render.py`:

```python
"""깨끗한 바코드 렌더링. python-barcode 의존을 이 파일에 가둔다."""

import io

import barcode
import cv2
import numpy as np
from barcode.writer import ImageWriter

_DPI = 300


def render_clean(text: str, module_px: float, height_px: int = 220) -> np.ndarray:
    """모듈 폭이 정확히 module_px 가 되는 Code128 을 그린다.

    python-barcode 는 mm 로 받으므로 dpi 를 고정하고 환산한다.
    """
    module_mm = module_px * 25.4 / _DPI
    writer = ImageWriter()
    obj = barcode.get("code128", text, writer=writer)
    buf = io.BytesIO()
    obj.write(buf, options={
        "module_width": module_mm,
        "module_height": 12.0,
        "quiet_zone": 2.0,
        "write_text": False,
        "dpi": _DPI,
    })
    buf.seek(0)
    arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = arr.shape
    return cv2.resize(arr, (w, height_px), interpolation=cv2.INTER_AREA)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_render.py -v`
Expected: PASS (3개)

- [ ] **Step 5: 커밋**

메시지: `[합성] 모듈 폭이 정확한 Code128 렌더링`
대상: `wemeet/data/render.py`, `tests/test_render.py`

---

## Task 2: `surface.py` — 기울기 생성과 예산

**Files:**
- Create: `wemeet/data/surface.py`
- Test: `tests/test_surface.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `grad_cylinder(w: int, h: int, theta_deg: float) -> tuple[np.ndarray, np.ndarray]` — `(z_x, z_y)`, 각각 `(h, w)` float64
  - `grad_sine(w, h, slope: float, lam: float, psi_deg: float = 0.0, phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]`
  - `grad_crease(w, h, slope: float, w_c: float, psi_deg: float = 0.0, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]`
  - `grad_crumple(w, h, slope: float, lam0: float, rng: np.random.Generator, octaves: int = 3, persistence: float = 0.7, psi_spread: float = 25.0) -> tuple[np.ndarray, np.ndarray]`
  - `octave_weights(octaves: int, persistence: float) -> np.ndarray` — 합이 1
  - `slope_budget(d_m0: float, c_min: float = 1.0) -> float`
  - `limit_slope(zx, zy, s_max: float) -> tuple[np.ndarray, np.ndarray, float]` — 세 번째는 적용된 `scale`

**주의:** `slope` 는 전부 **"이 성분의 최대 |z_x|"** 다. 진폭이 아니다. 그래야 예산을 성분끼리 선형으로 나눠 쓸 수 있다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_surface.py`:

```python
import math

import numpy as np
import pytest

from wemeet.data.surface import (grad_crease, grad_crumple, grad_cylinder,
                                 grad_sine, limit_slope, octave_weights,
                                 slope_budget)


def test_cylinder_edge_slope_is_tan_theta():
    zx, zy = grad_cylinder(401, 8, 45.0)
    assert abs(abs(zx).max() - math.tan(math.radians(45.0))) < 0.02
    assert np.allclose(zy, 0.0)


def test_cylinder_zero_angle_is_flat():
    zx, zy = grad_cylinder(101, 8, 0.0)
    assert np.allclose(zx, 0.0) and np.allclose(zy, 0.0)


def test_sine_peak_slope_equals_requested():
    zx, _ = grad_sine(400, 8, slope=0.8, lam=200.0)
    assert abs(abs(zx).max() - 0.8) < 0.01


def test_crease_saturates_to_plus_minus_slope():
    zx, _ = grad_crease(400, 8, slope=1.2, w_c=3.0)
    assert abs(zx.max() - 1.2) < 0.01
    assert abs(zx.min() + 1.2) < 0.01


def test_octave_weights_decay_by_persistence():
    w = octave_weights(3, 0.7)
    assert abs(w.sum() - 1.0) < 1e-12
    assert abs(w[1] / w[0] - 0.7) < 1e-12
    assert abs(w[2] / w[1] - 0.7) < 1e-12


def test_crumple_respects_total_slope():
    rng = np.random.default_rng(0)
    zx, _ = grad_crumple(400, 8, slope=1.0, lam0=300.0, rng=rng)
    assert abs(zx).max() <= 1.0 + 1e-9


def test_slope_budget_uses_c_min_one_by_default():
    assert abs(slope_budget(2.0) - math.sqrt(3.0)) < 1e-12
    assert abs(slope_budget(2.0, c_min=2.0) - 0.0) < 1e-12


def test_slope_budget_is_defined_below_two_px():
    """c_min=1.0 이라야 저해상도(d_m0 < 2.0)가 표현된다."""
    assert slope_budget(1.6) > 0.0
    with pytest.raises(ValueError):
        slope_budget(1.6, c_min=2.0)


def test_limit_slope_scales_both_axes_together():
    zx = np.array([[0.0, 4.0]])
    zy = np.array([[0.0, 2.0]])
    out_x, out_y, scale = limit_slope(zx, zy, s_max=2.0)
    assert abs(scale - 0.5) < 1e-12
    assert abs(out_x.max() - 2.0) < 1e-12
    assert abs(out_y.max() - 1.0) < 1e-12


def test_limit_slope_does_not_amplify():
    zx = np.array([[0.0, 1.0]])
    zy = np.zeros_like(zx)
    _, _, scale = limit_slope(zx, zy, s_max=5.0)
    assert scale == 1.0
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_surface.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 최소 구현**

`wemeet/data/surface.py`:

```python
"""레시피 파라미터 → 높이장 기울기. 설계 §3 수식의 집.

전부 (z_x, z_y) 를 돌려준다. slope 인자는 언제나 "이 성분의 최대 |z_x|" 다 —
진폭이 아니다. 그래야 기울기 예산을 성분끼리 선형으로 나눠 쓸 수 있다.
"""

import math

import numpy as np


def grad_cylinder(w: int, h: int, theta_deg: float):
    """전역 원통 감김. 관측 폭이 w 일 때 가장자리 기울기가 tan(theta)."""
    if theta_deg <= 0:
        return np.zeros((h, w)), np.zeros((h, w))
    th = math.radians(theta_deg)
    c = (w - 1) / 2.0
    r = (w - 1) / (2.0 * math.sin(th))
    x = np.arange(w, dtype=np.float64) - c
    zx = -x / np.sqrt(np.maximum(r * r - x * x, 1e-9))
    return np.tile(zx, (h, 1)), np.zeros((h, w))


def _axis(w: int, h: int, psi_deg: float):
    """능선 축 t = (x-cx)cos(psi) + (y-cy)sin(psi) 와 cos/sin."""
    psi = math.radians(psi_deg)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    t = (gx - (w - 1) / 2) * math.cos(psi) + (gy - (h - 1) / 2) * math.sin(psi)
    return t, math.cos(psi), math.sin(psi)


def grad_sine(w, h, slope, lam, psi_deg=0.0, phase=0.0):
    """물결 주름. slope = 이 성분의 최대 |z_x|."""
    t, cp, sp = _axis(w, h, psi_deg)
    d = slope * np.cos(2 * np.pi * t / lam + phase)
    return d, d * (sp / cp if cp else 0.0)


def grad_crease(w, h, slope, w_c, psi_deg=0.0, offset=0.0):
    """접힌 능선. 기울기가 -slope -> +slope 로 tanh 전이."""
    t, cp, sp = _axis(w, h, psi_deg)
    d = slope * np.tanh((t - offset) / max(w_c, 1e-6))
    return d, d * (sp / cp if cp else 0.0)


def octave_weights(octaves: int, persistence: float) -> np.ndarray:
    """옥타브별 기울기 가중치. 합이 1 이 되게 정규화한다.

    스파이크는 균등(persistence=1.0)이었다. 실측에서 균등은 목표 구간 5%,
    0.7 은 10% 다 (설계 §3).
    """
    w = np.array([persistence ** j for j in range(octaves)], dtype=np.float64)
    return w / w.sum()


def grad_crumple(w, h, slope, lam0, rng, octaves=3, persistence=0.7,
                 psi_spread=25.0):
    """여러 스케일 능선의 합 = 구김. 옥타브마다 파장이 절반, 기울기가 persistence 배."""
    weights = octave_weights(octaves, persistence)
    zx = np.zeros((h, w))
    zy = np.zeros((h, w))
    for j, weight in enumerate(weights):
        a, b = grad_sine(w, h, slope * weight, lam0 / (2 ** j),
                         psi_deg=float(rng.uniform(-psi_spread, psi_spread)),
                         phase=float(rng.uniform(0, 2 * np.pi)))
        zx += a
        zy += b
    return zx, zy


def slope_budget(d_m0: float, c_min: float = 1.0) -> float:
    """판독 하한 d_m0 * m_min >= c_min 에서 나오는 최대 허용 기울기 S_max.

    v1 은 c_min = 1.0 이다. 2.0 을 쓰면 d_m0 < 2.0 에서 정의되지 않아
    목표 구간이 열리는 저해상도 절반이 표현 불가가 된다 (설계 §3).
    """
    ratio = d_m0 / c_min
    if ratio < 1.0:
        raise ValueError(
            f"d_m0={d_m0} 가 c_min={c_min} 보다 작아 S_max 가 정의되지 않는다"
        )
    return math.sqrt(ratio ** 2 - 1.0)


def limit_slope(zx, zy, s_max: float):
    """초과분을 clip 하지 않고 기울기를 통째로 줄인다.

    np.clip(m, ...) 은 안 된다 — m 만 자르면 대응하는 z 가 없어져서
    음영·반사가 보여주는 곡률과 기하가 어긋난다 (실측 법선각 최대 11.32°).
    """
    peak = float(np.abs(zx).max())
    scale = min(1.0, s_max / peak) if peak > 0 else 1.0
    return zx * scale, zy * scale, scale
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_surface.py -v`
Expected: PASS (10개)

- [ ] **Step 5: 커밋**

메시지: `[합성] 기울기 생성기 4종과 예산 — c_min 1.0, persistence 0.7`
대상: `wemeet/data/surface.py`, `tests/test_surface.py`

---

## Task 3: `warp.py` (1/2) — 매핑. 방향 함정이 사는 곳

**Files:**
- Create: `wemeet/data/warp.py`
- Test: `tests/test_warp.py`

**Interfaces:**
- Consumes: `wemeet.data.surface.grad_cylinder`, `grad_crease` (테스트에서만)
- Produces:
  - `flat_coord(zx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]` — `(m, s_hat, s_total)`. `m` 은 압축률 `(h,w)`, `s_hat` 은 행마다 0~1 로 정규화한 펴진 좌표 `(h,w)`, `s_total` 은 펴진 총 길이(px, 행 평균)
  - `fit_obs_width(make_grad, w_flat: int, h: int, iters: int = 4) -> tuple[int, np.ndarray, np.ndarray]` — `(w_obs, zx, zy)`. `make_grad(w, h) -> (zx, zy)` 콜백
  - `apply_warp(clean: np.ndarray, s_hat: np.ndarray) -> np.ndarray` — uint8 관측 이미지

**이 태스크가 지키는 설계 §10 항목:** 1(해석해 대조), 2(오류판 실패), 6(m 과 기울기 일치), 7(m ≤ 1)

> **왜 이 테스트들이 중요한가.** `cumsum(m)` 과 `cumsum(1/m)` 은 **약한 왜곡에서 둘 다 읽힌다.**
> 디코딩으로는 절대 안 잡힌다. 해석해와 대조하는 것만이 잡는다 — 실측 오차 0.24px vs 35.11px.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_warp.py`:

```python
import math

import numpy as np

from wemeet.data.render import render_clean
from wemeet.data.surface import grad_crease, grad_cylinder
from wemeet.data.warp import apply_warp, fit_obs_width, flat_coord


def _cylinder_geometry(w, theta_deg):
    th = math.radians(theta_deg)
    return (w - 1) / 2.0, (w - 1) / (2.0 * math.sin(th))


def test_cumsum_of_inverse_m_matches_arcsin_within_half_pixel():
    """설계 §10-1. J = 1/m 를 적분하면 r*arcsin((x-c)/r) 이다."""
    w, theta = 401, 45.0
    zx, _ = grad_cylinder(w, 8, theta)
    _, s_hat, s_total = flat_coord(zx)

    c, r = _cylinder_geometry(w, theta)
    x = np.arange(w, dtype=np.float64) - c
    analytic = r * (np.arcsin(x / r) - np.arcsin(-c / r))
    numeric = s_hat[0] * s_total

    assert np.abs(numeric - analytic).max() < 0.5


def test_cumsum_of_m_fails_the_same_comparison():
    """설계 §10-2. 오류판(cumsum(m))은 위 대조에서 크게 벗어난다."""
    w, theta = 401, 45.0
    zx, _ = grad_cylinder(w, 8, theta)
    m = 1.0 / np.sqrt(1.0 + zx ** 2)

    wrong = np.cumsum(m, axis=1) - m[:, :1]
    c, r = _cylinder_geometry(w, theta)
    x = np.arange(w, dtype=np.float64) - c
    analytic = r * (np.arcsin(x / r) - np.arcsin(-c / r))

    assert np.abs(wrong[0] - analytic).max() > 5.0


def test_m_never_exceeds_one():
    """설계 §10-7."""
    for theta in (10.0, 30.0, 60.0, 75.0):
        zx, _ = grad_cylinder(301, 8, theta)
        m, _, _ = flat_coord(zx)
        assert m.max() <= 1.0 + 1e-12
        assert m.min() > 0.0


def test_m_and_slope_stay_consistent():
    """설계 §10-6. sqrt(1/m^2 - 1) 이 |z_x| 와 같아야 한다."""
    zx, _ = grad_crease(301, 8, slope=1.5, w_c=4.0)
    m, _, _ = flat_coord(zx)
    recovered = np.sqrt(1.0 / m ** 2 - 1.0)
    assert np.abs(recovered - np.abs(zx)).max() < 1e-9


def test_s_hat_is_normalised_per_row():
    zx, _ = grad_crease(201, 8, slope=1.0, w_c=3.0, psi_deg=20.0)
    _, s_hat, _ = flat_coord(zx)
    assert np.allclose(s_hat[:, 0], 0.0)
    assert np.allclose(s_hat[:, -1], 1.0)


def test_fit_obs_width_recovers_flat_length():
    """감기면 좁아 보인다. 펴진 길이가 w_flat 이 되도록 관측 폭을 맞춘다."""
    w_flat, h = 400, 8
    w_obs, zx, _ = fit_obs_width(
        lambda w, hh: grad_cylinder(w, hh, 50.0), w_flat, h)
    _, _, s_total = flat_coord(zx)
    assert w_obs < w_flat
    assert abs(s_total - w_flat) / w_flat < 0.05


def test_apply_warp_shape_and_dtype():
    clean = render_clean("WEMEET0001", module_px=4.0, height_px=220)
    zx, _ = grad_crease(301, 220, slope=1.0, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat)
    assert obs.shape == s_hat.shape
    assert obs.dtype == np.uint8
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_warp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wemeet.data.warp'`

- [ ] **Step 3: 최소 구현**

`wemeet/data/warp.py`:

```python
"""기울기 → 매핑 → 관측 이미지. 방향 함정을 이 파일에 가둔다.

규약 (설계 §2):
    x = 관측 좌표, s = 펴진 좌표
    J = ds/dx = sqrt(1 + z_x^2) >= 1     신장률
    m = dx/ds = 1/J <= 1                 압축률
    s(x) = cumsum(J) = cumsum(1/m)       <- cumsum(m) 이 아니다
"""

import cv2
import numpy as np


def flat_coord(zx: np.ndarray):
    """관측 -> 펴진 매핑. 행마다 가로로만 적분한다 (v = y 근사).

    돌려주는 것은 (m, s_hat, s_total):
        m       압축률 (h, w).  0 < m <= 1
        s_hat   행마다 0~1 로 정규화한 펴진 좌표 (h, w)
        s_total 펴진 총 길이(px). 행 평균
    """
    m = 1.0 / np.sqrt(1.0 + zx ** 2)
    integrand = 1.0 / m
    s = np.cumsum(integrand, axis=1) - integrand[:, :1]
    s_hat = s / s[:, -1:]
    return m, s_hat, float(s[:, -1].mean())


def fit_obs_width(make_grad, w_flat: int, h: int, iters: int = 4):
    """펴진 길이가 w_flat 이 되도록 관측 폭을 맞춘다.

    감기면 좁아 보이므로 관측 폭은 펴진 폭보다 작다. 고정점 반복으로 찾는다.
    """
    w = w_flat
    for _ in range(iters):
        zx, zy = make_grad(w, h)
        _, _, s_total = flat_coord(zx)
        w_new = max(32, int(round(w * w_flat / s_total)))
        if w_new == w:
            break
        w = w_new
    zx, zy = make_grad(w, h)
    return w, zx, zy


def apply_warp(clean: np.ndarray, s_hat: np.ndarray) -> np.ndarray:
    """깨끗한 이미지를 관측 이미지로 찌그러뜨린다."""
    h_obs, w_obs = s_hat.shape
    h0, w0 = clean.shape
    map_x = (s_hat * (w0 - 1)).astype(np.float32)
    gy = np.linspace(0, h0 - 1, h_obs, dtype=np.float32)
    map_y = np.tile(gy[:, None], (1, w_obs))
    return cv2.remap(clean, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_warp.py -v`
Expected: PASS (7개)

- [ ] **Step 5: 커밋**

메시지: `[합성] 관측 매핑 — cumsum(1/m) 을 해석해로 잠근다`
대상: `wemeet/data/warp.py`, `tests/test_warp.py`

---

## Task 4: `warp.py` (2/2) — 대응장 `G` 와 제어점

**Files:**
- Modify: `wemeet/data/warp.py` (함수 3개 추가)
- Test: `tests/test_warp.py` (테스트 4개 추가)

**Interfaces:**
- Consumes: Task 3 의 `flat_coord`
- Produces:
  - `build_G(s_hat: np.ndarray, n_v: int = 33, n_u: int = 513) -> np.ndarray` — `(n_v, n_u, 2)` float64. `G[i, j] = (x_obs, y_obs)` **픽셀 단위**. 펴진 정규화 좌표 `(u, v)` 의 내용이 있는 관측 픽셀 위치
  - `sample_G(G: np.ndarray, u: float, v: float) -> np.ndarray` — `(2,)` 이중선형 샘플
  - `control_points(G, n_x: int, n_y: int, shape: tuple[int, int], u_lo: float = 0.0, u_hi: float = 1.0) -> tuple[np.ndarray, np.ndarray]` — `(dst_norm, src_norm)`, 각각 `(n_x*n_y, 2)`. 행 우선. `shape` 는 **현재 이미지의 (h, w)** 이고 이걸로 `src` 를 정규화한다

**이 태스크가 지키는 설계 §10 항목:** 3(단조성), 그리고 §5 의 "`G` 에서 뽑은 제어점이 직접 역보간과 차이 0.000000"

> **방향이 `G`(펴진→관측)여야 한다.** `F`(관측→펴진)로 들면 회전 시 벡터장을 리샘플해야
> 하고 제어점 추출이 2D 역문제가 된다. `G` 로 들면 회전이 저장된 좌표에 아핀을 곱하는 것으로 끝난다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_warp.py` 에 추가:

```python
from wemeet.data.warp import build_G, control_points, sample_G


def _direct_control_points(s_hat, n_x, n_y):
    """G 를 안 거치고 행별 역보간으로 직접 뽑는 참조 구현 (테스트 전용)."""
    h, w = s_hat.shape
    xs = np.arange(w, dtype=np.float64)
    dst, src = [], []
    for v in np.linspace(0, 1, n_y):
        row = min(h - 1, int(round(v * (h - 1))))
        for u in np.linspace(0, 1, n_x):
            dst.append([u, v])
            src.append([np.interp(u, s_hat[row], xs) / (w - 1), v])
    return np.array(dst), np.array(src)


def test_build_G_shape_and_endpoints():
    zx, _ = grad_crease(301, 220, slope=1.0, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    g = build_G(s_hat)
    assert g.shape == (33, 513, 2)
    assert np.allclose(g[:, 0, 0], 0.0, atol=1e-6)
    assert np.allclose(g[:, -1, 0], 300.0, atol=1e-6)


def test_sample_G_at_nodes_returns_the_node():
    zx, _ = grad_crease(301, 220, slope=1.0, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    g = build_G(s_hat)
    assert np.allclose(sample_G(g, 0.0, 0.0), g[0, 0])
    assert np.allclose(sample_G(g, 1.0, 1.0), g[-1, -1])


def test_control_points_from_G_match_direct_interpolation():
    """설계 §5. 두 경로가 일치해야 한다 (실측 차이 0.000000)."""
    zx, _ = grad_crease(301, 220, slope=1.2, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    g = build_G(s_hat)
    dst_g, src_g = control_points(g, 6, 3, s_hat.shape)
    dst_d, src_d = _direct_control_points(s_hat, 6, 3)
    assert np.abs(dst_g - dst_d).max() < 1e-9
    assert np.abs(src_g - src_d).max() < 2e-3


def test_src_x_is_monotonic_per_row():
    """설계 §10-3. 깨졌다면 역보간을 빼먹은 것이다."""
    zx, _ = grad_crease(301, 220, slope=1.5, w_c=3.0, psi_deg=20.0)
    _, s_hat, _ = flat_coord(zx)
    g = build_G(s_hat)
    _, src = control_points(g, 6, 3, s_hat.shape)
    for row in src.reshape(3, 6, 2):
        assert np.all(np.diff(row[:, 0]) > 0)
```

> `src` 허용 오차가 `2e-3` 인 것은 `G` 가 `513` 열로 이산화돼 있어서다. 정규화 좌표
> 기준이므로 `301px` 크롭에서 0.6px 이다. `dst` 는 이산화와 무관하므로 `1e-9` 다.

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_warp.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_G'`

- [ ] **Step 3: 최소 구현**

`wemeet/data/warp.py` 에 추가:

```python
def build_G(s_hat: np.ndarray, n_v: int = 33, n_u: int = 513) -> np.ndarray:
    """밀집 대응장. G[i, j] = 펴진 (u, v) 의 내용이 있는 관측 픽셀 (x, y).

    크롭 크기와 무관하게 해상도가 고정이라 float64 두 장 270KB 다.
    """
    h, w = s_hat.shape
    xs = np.arange(w, dtype=np.float64)
    us = np.linspace(0.0, 1.0, n_u)
    g = np.empty((n_v, n_u, 2), dtype=np.float64)
    for i, v in enumerate(np.linspace(0.0, 1.0, n_v)):
        row = min(h - 1, int(round(v * (h - 1))))
        g[i, :, 0] = np.interp(us, s_hat[row], xs)
        g[i, :, 1] = v * (h - 1)
    return g


def sample_G(g: np.ndarray, u: float, v: float) -> np.ndarray:
    """대응장을 이중선형으로 샘플한다."""
    n_v, n_u = g.shape[:2]
    fu, fv = u * (n_u - 1), v * (n_v - 1)
    i0, j0 = int(np.floor(fv)), int(np.floor(fu))
    i1, j1 = min(i0 + 1, n_v - 1), min(j0 + 1, n_u - 1)
    a, b = fv - i0, fu - j0
    return ((1 - a) * (1 - b) * g[i0, j0] + (1 - a) * b * g[i0, j1]
            + a * (1 - b) * g[i1, j0] + a * b * g[i1, j1])


def control_points(g: np.ndarray, n_x: int, n_y: int, shape,
                   u_lo: float = 0.0, u_hi: float = 1.0):
    """정답 제어점. dst = 펴진 격자, src = 지금 있는 위치. 행 우선.

    u_lo/u_hi 는 크롭으로 남은 펴진 범위다. dst 는 그 범위를 다시 0~1 로
    정규화한다 — 크롭 뒤 제어점은 변환으로 얻을 수 없기 때문이다 (설계 §5).
    """
    h, w = shape
    dst, src = [], []
    span = max(u_hi - u_lo, 1e-9)
    for v in np.linspace(0.0, 1.0, n_y):
        for u in np.linspace(u_lo, u_hi, n_x):
            p = sample_G(g, u, v)
            dst.append([(u - u_lo) / span, v])
            src.append([p[0] / (w - 1), p[1] / (h - 1)])
    return np.array(dst), np.array(src)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_warp.py -v`
Expected: PASS (11개)

- [ ] **Step 5: 커밋**

메시지: `[합성] 대응장 G 와 제어점 추출 — 직접 역보간과 일치를 잠근다`
대상: `wemeet/data/warp.py`, `tests/test_warp.py`

---

## Task 5: `optics.py` — Blinn-Phong 과 포화 화소 비율

**Files:**
- Create: `wemeet/data/optics.py`
- Test: `tests/test_optics.py`

**Interfaces:**
- Consumes: `wemeet.data.surface.grad_crease` (테스트에서만)
- Produces: `shade(base: np.ndarray, zx, zy, light, ka: float = 0.35, kd: float = 0.50, ks: float = 0.0, p: float = 100.0) -> tuple[np.ndarray, float]` — `(uint8 이미지, 포화 화소 비율)`. `base` 는 uint8 또는 float64 `(h, w)`, `light` 는 길이 3 시퀀스

**포화 화소 비율이 §7 의 `τ` 다.** 라벨링에서 "기하가 소실된 불가" 를 가르는 유일한 신호이므로
여기서 반드시 같이 돌려줘야 한다.

**이 태스크가 지키는 설계 §10 항목:** 8(하이라이트가 곡률에 붙는다), 그리고 §4 의 "정반사는 더하기"

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_optics.py`:

```python
import numpy as np

from wemeet.data.optics import shade
from wemeet.data.surface import grad_crease


def test_specular_is_added_not_multiplied():
    """설계 §4. 곱하면 0 * 무엇 = 0 이라 검은 막대가 절대 안 지워진다."""
    h, w = 8, 301
    zx, zy = grad_crease(w, h, slope=1.5, w_c=4.0)
    black = np.zeros((h, w), dtype=np.uint8)
    img, _ = shade(black, zx, zy, light=(0.3, -0.2, 1.0), ks=0.9, p=40.0)
    assert img.max() > 60, "검은 바탕이 정반사로 밝아져야 한다 (더하기)"


def test_no_specular_leaves_black_black():
    h, w = 8, 301
    zx, zy = grad_crease(w, h, slope=1.5, w_c=4.0)
    black = np.zeros((h, w), dtype=np.uint8)
    img, _ = shade(black, zx, zy, light=(0.3, -0.2, 1.0), ks=0.0)
    assert img.max() == 0


def test_highlight_sits_on_the_crease():
    """설계 §10-8. 하이라이트가 곡률 최대점에서 5px 안. 무작위 마스크는 86.8px."""
    h, w = 8, 401
    zx, zy = grad_crease(w, h, slope=1.5, w_c=4.0, offset=0.0)
    black = np.zeros((h, w), dtype=np.uint8)
    img, _ = shade(black, zx, zy, light=(0.0, 0.0, 1.0), ks=0.9, p=60.0)

    highlight_col = int(np.argmax(img[h // 2]))
    curvature_col = int(np.argmax(np.abs(np.gradient(zx[h // 2]))))
    assert abs(highlight_col - curvature_col) <= 5


def test_saturation_ratio_is_zero_without_specular():
    h, w = 8, 101
    zx, zy = np.zeros((h, w)), np.zeros((h, w))
    white = np.full((h, w), 255, dtype=np.uint8)
    _, sat = shade(white, zx, zy, light=(0.0, 0.0, 1.0), ks=0.0)
    assert sat == 0.0, "ka + kd = 0.85 이므로 포화가 없어야 한다"


def test_saturation_ratio_rises_with_specular():
    h, w = 8, 101
    zx, zy = np.zeros((h, w)), np.zeros((h, w))
    white = np.full((h, w), 255, dtype=np.uint8)
    _, sat = shade(white, zx, zy, light=(0.0, 0.0, 1.0), ks=1.0, p=1.0)
    assert sat > 0.9


def test_flat_surface_uses_ambient_plus_full_diffuse():
    h, w = 4, 16
    zx, zy = np.zeros((h, w)), np.zeros((h, w))
    white = np.full((h, w), 255, dtype=np.uint8)
    img, _ = shade(white, zx, zy, light=(0.0, 0.0, 1.0), ks=0.0)
    assert abs(float(img.mean()) - 0.85 * 255) < 1.0
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_optics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wemeet.data.optics'`

- [ ] **Step 3: 최소 구현**

`wemeet/data/optics.py`:

```python
"""Blinn-Phong 음영과 정반사. 법선은 기하에 쓴 바로 그 기울기에서 나온다.

    I = B * (k_a + k_d * max(0, n.l))  +  k_s * (n.h)^p,   h = (l + v)/|l + v|

정사영 가정이므로 v = (0, 0, 1) 이다.
정반사는 **더한다**. 곱하면 검은 막대가 절대 지워지지 않아 정보를 죽이는
메커니즘 자체가 사라진다 (설계 §4: 더하기 0.275 vs 곱하기 0.005).
"""

import numpy as np

_SATURATION = 0.99


def shade(base, zx, zy, light, ka: float = 0.35, kd: float = 0.50,
          ks: float = 0.0, p: float = 100.0):
    """음영·정반사를 얹고 (이미지, 포화 화소 비율) 을 돌려준다.

    포화 화소 비율은 설계 §7 의 tau 다 — 기하가 소실된 불가 샘플을 가르는
    유일한 신호이므로 여기서 같이 낸다. 렌더 직후라 추가 비용이 없다.
    """
    n = np.stack([-zx, -zy, np.ones_like(zx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    l = np.asarray(light, dtype=np.float64)
    l = l / np.linalg.norm(l)
    hv = l + np.array([0.0, 0.0, 1.0])
    hv /= np.linalg.norm(hv)

    diffuse = np.clip(n @ l, 0.0, None)
    specular = np.clip(n @ hv, 0.0, None) ** p

    img = np.clip(np.asarray(base, dtype=np.float64) / 255.0
                  * (ka + kd * diffuse) + ks * specular, 0.0, 1.0)
    sat = float((img >= _SATURATION).mean())
    return (img * 255).astype(np.uint8), sat
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_optics.py -v`
Expected: PASS (6개)

- [ ] **Step 5: 커밋**

메시지: `[합성] Blinn-Phong 음영·정반사와 포화 화소 비율`
대상: `wemeet/data/optics.py`, `tests/test_optics.py`

---

## Task 6: `augment.py` — 여백·회전(`G` 동반)과 광도 증강

**Files:**
- Create: `wemeet/data/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Consumes: Task 4 의 `build_G`, `control_points`
- Produces:
  - `geometric_rotate(img: np.ndarray, g: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]`
  - `geometric_margin(img, g, left: float, right: float, top: float, bottom: float) -> tuple[np.ndarray, np.ndarray, float, float]` — `(img, G, u_lo, u_hi)`. 각 비율은 **양수 = 여백 추가, 음수 = 잘라먹기**. `[-0.02, 0.15]` 로 clamp 한다
  - `photometric(img: np.ndarray, rng, sigma: float = 0.0, noise: float = 0.0, jpeg: int | None = None) -> np.ndarray`

**시그니처로 두 종류를 가른다** (설계 §5). `geometric*` 는 좌표를 **반드시 함께 받고 함께 돌려준다.**
`photometric` 은 좌표를 **받지도 않는다.** 실수로 좌표를 안 옮기는 사고를 타입 수준에서 막는다.

**−2% 하한의 근거:** 잘라먹기 −6%/−4% 는 Code128 의 start/stop 이 사라져 **어떤 제어점으로도 못 읽는다**
(실측). −2%/−2% 는 읽힌다. 검출기가 바코드를 잘라먹으면 그건 검출 실패이므로 학습 데이터에 넣을 이유가 없다.

> **설계 §10-4(펴진 이미지 픽셀 비교)는 여기서 못 한다** — `rectify` 가 아직 없기 때문이다.
> **Task 8 에서 추가한다.** 이 태스크는 좌표 수준에서 잡을 수 있는 것만 잡는다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_augment.py`:

```python
import inspect

import numpy as np

from wemeet.data.augment import geometric_margin, geometric_rotate, photometric
from wemeet.data.surface import grad_crease
from wemeet.data.warp import build_G, control_points, flat_coord


def _fixture():
    zx, _ = grad_crease(301, 220, slope=1.2, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    img = np.full(s_hat.shape, 200, dtype=np.uint8)
    return img, build_G(s_hat)


def test_rotate_round_trip_restores_G():
    img, g = _fixture()
    img2, g2 = geometric_rotate(img, g, 7.0)
    _, g3 = geometric_rotate(img2, g2, -7.0)
    assert np.abs(g3 - g).max() < 1e-6


def test_rotate_keeps_image_shape():
    img, g = _fixture()
    img2, g2 = geometric_rotate(img, g, 12.0)
    assert img2.shape == img.shape
    assert g2.shape == g.shape


def test_padding_keeps_full_u_range():
    img, g = _fixture()
    img2, _, u_lo, u_hi = geometric_margin(img, g, 0.08, 0.05, 0.05, 0.05)
    assert (u_lo, u_hi) == (0.0, 1.0)
    assert img2.shape[1] > img.shape[1]


def test_cutting_shrinks_u_range():
    img, g = _fixture()
    img2, _, u_lo, u_hi = geometric_margin(img, g, -0.02, -0.02, 0.0, 0.0)
    assert img2.shape[1] < img.shape[1]
    assert u_lo > 0.0 and u_hi < 1.0


def test_margin_is_clamped_to_spec_range():
    img, g = _fixture()
    _, _, lo_a, hi_a = geometric_margin(img, g, -0.30, 0.0, 0.0, 0.0)
    _, _, lo_b, hi_b = geometric_margin(img, g, -0.02, 0.0, 0.0, 0.0)
    assert abs(lo_a - lo_b) < 1e-9 and abs(hi_a - hi_b) < 1e-9


def test_control_points_stay_inside_contract_range_after_augmentation():
    img, g = _fixture()
    img, g = geometric_rotate(img, g, 5.0)
    img, g, u_lo, u_hi = geometric_margin(img, g, 0.06, 0.04, 0.05, 0.05)
    dst, src = control_points(g, 6, 3, img.shape, u_lo, u_hi)
    assert dst.min() >= 0.0 and dst.max() <= 1.0
    assert src.min() >= -0.5 and src.max() <= 1.5


def test_photometric_never_receives_coordinates():
    """설계 §5. 시그니처로 사고를 막는다."""
    params = inspect.signature(photometric).parameters
    assert "g" not in params and "G" not in params


def test_photometric_leaves_control_points_bit_identical():
    """설계 §10-5."""
    img, g = _fixture()
    before = control_points(g, 6, 3, img.shape)
    photometric(img, np.random.default_rng(0), sigma=0.6, noise=3.0, jpeg=80)
    after = control_points(g, 6, 3, img.shape)
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])


def test_photometric_is_identity_when_all_off():
    img, _ = _fixture()
    out = photometric(img, np.random.default_rng(0))
    assert np.array_equal(out, img)


def test_photometric_changes_pixels_when_on():
    img, _ = _fixture()
    out = photometric(img, np.random.default_rng(0), sigma=0.7, noise=4.0, jpeg=70)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_augment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wemeet.data.augment'`

- [ ] **Step 3: 최소 구현**

`wemeet/data/augment.py`:

```python
"""검출기 출력 모사. 두 종류를 시그니처로 가른다 (설계 §5).

    geometric*  좌표(G)를 반드시 함께 받고 함께 돌려준다
    photometric 좌표를 받지도 않는다

여백은 -2% ~ +15% 다. -6% 는 Code128 의 start/stop 이 잘려 어떤 제어점으로도
못 읽는다 (실측). 검출기가 바코드를 잘라먹으면 그건 검출 실패다.
"""

import cv2
import numpy as np

MARGIN_MIN = -0.02
MARGIN_MAX = 0.15


def geometric_rotate(img: np.ndarray, g: np.ndarray, deg: float):
    """이미지를 회전하고 대응장에 같은 아핀을 곱한다.

    G 가 펴진->관측 방향이라 저장된 좌표에 행렬을 곱하는 것으로 끝난다.
    """
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), deg, 1.0)
    out = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    homogeneous = np.concatenate([g, np.ones(g.shape[:2] + (1,))], axis=-1)
    return out, homogeneous @ matrix.T


def geometric_margin(img: np.ndarray, g: np.ndarray,
                     left: float, right: float, top: float, bottom: float):
    """여백을 더하거나(양수) 잘라먹는다(음수). 대응장을 같이 옮긴다.

    돌려주는 u_lo/u_hi 는 남은 크롭이 대응하는 펴진 범위다. 잘라먹으면
    좁아진다 — 크롭 뒤 제어점은 변환으로 얻을 수 없으므로 이 범위를
    control_points 에 넘겨 다시 정규화해야 한다.
    """
    h, w = img.shape
    fracs = [float(np.clip(f, MARGIN_MIN, MARGIN_MAX))
             for f in (left, right, top, bottom)]
    dl, dr = int(round(fracs[0] * w)), int(round(fracs[1] * w))
    dt, db = int(round(fracs[2] * h)), int(round(fracs[3] * h))

    pad = [max(0, dl), max(0, dr), max(0, dt), max(0, db)]
    out = cv2.copyMakeBorder(img, pad[2], pad[3], pad[0], pad[1],
                             cv2.BORDER_REPLICATE)
    moved = g + np.array([pad[0], pad[2]], dtype=np.float64)

    x0, y0 = max(0, -dl), max(0, -dt)
    x1 = out.shape[1] - max(0, -dr)
    y1 = out.shape[0] - max(0, -db)
    out = out[y0:y1, x0:x1].copy()
    moved = moved - np.array([x0, y0], dtype=np.float64)

    hh, ww = out.shape
    inside = ((moved[..., 0] >= 0) & (moved[..., 0] <= ww - 1)).all(axis=0)
    us = np.linspace(0.0, 1.0, g.shape[1])
    if not inside.any():
        return out, moved, 0.0, 1.0
    return out, moved, float(us[inside][0]), float(us[inside][-1])


def photometric(img: np.ndarray, rng, sigma: float = 0.0,
                noise: float = 0.0, jpeg: int | None = None) -> np.ndarray:
    """초점 블러 -> 센서 노이즈 -> JPEG. 좌표는 건드리지 않는다."""
    out = img
    if sigma > 0.05:
        out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigma)
        out = np.clip(out, 0, 255).astype(np.uint8)
    if noise > 0.0:
        out = np.clip(out.astype(np.float64)
                      + rng.normal(0.0, noise, out.shape), 0, 255).astype(np.uint8)
    if jpeg is not None:
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_augment.py -v`
Expected: PASS (10개)

- [ ] **Step 5: 커밋**

메시지: `[합성] 증강 — geometric 은 G 동반, photometric 은 좌표를 안 받는다`
대상: `wemeet/data/augment.py`, `tests/test_augment.py`

---

## Task 7: `synthesis.py` — 레시피와 조립

**Files:**
- Create: `wemeet/data/synthesis.py`
- Test: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: Task 1~6 전부
- Produces:
  - `BUCKETS: dict[str, tuple[float, float]]` — `{"L": (1.6, 2.5), "M": (2.5, 3.7), "H": (3.7, 6.0)}`
  - `Recipe` — `@dataclass(frozen=True)`. 필드는 아래 구현 참조
  - `draw_recipe(rng: np.random.Generator, bucket: str, index: int) -> Recipe`
  - `build(recipe: Recipe) -> Sample` — `Sample` 은 `@dataclass` 로 `obs`(uint8), `dst_norm`, `src_norm`, `m_min`, `sat_ratio`, `scale`, `w_flat`
  - `recipe_to_dict(r: Recipe) -> dict` / `recipe_from_dict(d: dict) -> Recipe`

**조립 순서는 설계 §5 그대로다. 바꾸지 말 것.**

```
① 깨끗한 바코드                       render.py
② 기하 왜곡 → 관측 이미지 + G          warp.py
③ 음영·정반사                         optics.py
④ 기하 증강 (여백·회전) — 이미지와 G   augment.py
⑤ 제어점 = 최종 G 의 격자 샘플         warp.py
⑥ 광도 증강 (블러·노이즈·JPEG)         augment.py
```

**분포는 설계 §6 v1 그대로다** (preset 은 2026-09-08 개정판의 30/30/30/10).

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_synthesis.py`:

```python
import hashlib

import numpy as np

from wemeet.data.synthesis import (BUCKETS, build, draw_recipe,
                                   recipe_from_dict, recipe_to_dict)


def test_bucket_ranges_cover_the_spec_span():
    assert BUCKETS["L"][0] == 1.6
    assert BUCKETS["H"][1] == 6.0
    assert BUCKETS["L"][1] == BUCKETS["M"][0]
    assert BUCKETS["M"][1] == BUCKETS["H"][0]


def test_draw_recipe_respects_bucket():
    rng = np.random.default_rng(1)
    for name, (lo, hi) in BUCKETS.items():
        for i in range(50):
            r = draw_recipe(rng, name, i)
            assert lo <= r.d_m0 <= hi
            assert 1.3 <= r.d_t <= 2.6
            assert r.d_t < 0.97 * r.d_m0


def test_draw_recipe_is_deterministic_for_same_seed():
    a = draw_recipe(np.random.default_rng(7), "L", 3)
    b = draw_recipe(np.random.default_rng(7), "L", 3)
    assert recipe_to_dict(a) == recipe_to_dict(b)


def test_recipe_round_trips_through_dict():
    r = draw_recipe(np.random.default_rng(2), "M", 0)
    assert recipe_from_dict(recipe_to_dict(r)) == r


def test_build_is_bit_reproducible():
    """설계 §10-9. 같은 레시피 두 번 -> 이미지 바이트가 동일하다."""
    r = draw_recipe(np.random.default_rng(11), "L", 0)
    a, b = build(r), build(r)
    assert hashlib.sha256(a.obs.tobytes()).hexdigest() == \
           hashlib.sha256(b.obs.tobytes()).hexdigest()
    assert np.array_equal(a.src_norm, b.src_norm)


def test_build_honours_the_contract_ranges():
    rng = np.random.default_rng(5)
    for i in range(8):
        s = build(draw_recipe(rng, "L", i))
        assert s.dst_norm.shape == (18, 2)
        assert s.src_norm.shape == (18, 2)
        assert s.dst_norm.min() >= 0.0 and s.dst_norm.max() <= 1.0
        assert s.src_norm.min() >= -0.5 and s.src_norm.max() <= 1.5


def test_build_reports_m_min_and_saturation():
    s = build(draw_recipe(np.random.default_rng(3), "L", 0))
    assert 0.0 < s.m_min <= 1.0
    assert 0.0 <= s.sat_ratio <= 1.0
    assert 0.0 < s.scale <= 1.0


def test_slope_target_is_hit_not_just_bounded():
    """S_t 는 상한이 아니라 목표다. 최소 모듈 폭이 d_t 근처여야 한다."""
    rng = np.random.default_rng(9)
    for i in range(6):
        r = draw_recipe(rng, "L", i)
        s = build(r)
        assert abs(s.m_min * r.d_m0 - r.d_t) < 0.25 * r.d_t
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_synthesis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wemeet.data.synthesis'`

- [ ] **Step 3: 최소 구현**

`wemeet/data/synthesis.py`:

```python
"""레시피 샘플링과 조립. 이미지는 저장하지 않고 레시피만 남긴다.

조립 순서는 설계 §5 다:
    렌더 -> 기하 왜곡(+G) -> 음영 -> 기하 증강(+G) -> 제어점 -> 광도 증강
"""

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from wemeet.data.augment import geometric_margin, geometric_rotate, photometric
from wemeet.data.optics import shade
from wemeet.data.render import render_clean
from wemeet.data.surface import (grad_crease, grad_crumple, grad_cylinder,
                                 grad_sine, limit_slope, slope_budget)
from wemeet.data.warp import (apply_warp, build_G, control_points,
                              fit_obs_width, flat_coord)

BUCKETS = {"L": (1.6, 2.5), "M": (2.5, 3.7), "H": (3.7, 6.0)}
PRESETS = ("crease", "sine", "octave", "cylinder")
PRESET_P = (0.30, 0.30, 0.30, 0.10)

H_OBS = 220
K_A, K_D = 0.35, 0.50
C_MIN = 1.0
PERSISTENCE = 0.7
N_X, N_Y = 6, 3


@dataclass(frozen=True)
class Recipe:
    seed: int
    bucket: str
    text: str
    d_m0: float
    d_t: float
    preset: str
    cyl_share: float
    psi: float
    w_c: float
    lam_f: float
    phase: float
    offset: float
    light: tuple
    ks: float
    p: float
    sigma: float
    noise: float
    jpeg: int
    rot_deg: float
    margin: tuple
    symbology: str = "code128"
    n_x: int = N_X
    n_y: int = N_Y


@dataclass
class Sample:
    obs: np.ndarray
    dst_norm: np.ndarray
    src_norm: np.ndarray
    m_min: float
    sat_ratio: float
    scale: float
    w_flat: int


def draw_recipe(rng: np.random.Generator, bucket: str, index: int) -> Recipe:
    """설계 §6 v1 분포. d_m0 만 버킷 범위에서 뽑는다."""
    lo, hi = BUCKETS[bucket]
    d_m0 = float(rng.uniform(lo, hi))
    d_t = float(rng.uniform(1.3, min(2.6, 0.97 * d_m0)))
    return Recipe(
        seed=int(rng.integers(0, 2 ** 31 - 1)),
        bucket=bucket,
        text=f"WEMEET{index:04d}",
        d_m0=d_m0,
        d_t=d_t,
        preset=str(rng.choice(PRESETS, p=PRESET_P)),
        cyl_share=float(rng.uniform(0.15, 0.45)),
        psi=float(rng.uniform(-45.0, 45.0)),
        w_c=float(rng.uniform(1.0, 9.0)),
        lam_f=float(rng.uniform(0.5, 1.3)),
        phase=float(rng.uniform(0.0, 2 * np.pi)),
        offset=float(rng.uniform(-0.3, 0.3)),
        light=(float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)), 1.0),
        ks=float(rng.uniform(0.0, 0.9)),
        p=float(rng.uniform(40.0, 300.0)),
        sigma=float(rng.uniform(0.0, 0.8)),
        noise=float(rng.uniform(1.0, 5.0)),
        jpeg=int(rng.integers(70, 96)),
        rot_deg=float(rng.uniform(-5.0, 5.0)),
        margin=tuple(float(rng.uniform(-0.02, 0.15)) for _ in range(4)),
    )


def _make_grad(recipe: Recipe, s_t: float, rng: np.random.Generator):
    """레시피 -> (w, h) 를 받아 기울기를 만드는 콜백."""
    share = recipe.cyl_share if recipe.preset != "cylinder" else 1.0

    def make(w: int, h: int):
        theta = math.degrees(math.atan(s_t * share))
        zx, zy = grad_cylinder(w, h, theta)
        rest = s_t * (1.0 - share)
        if rest <= 0.0:
            return zx, zy
        if recipe.preset == "crease":
            a, b = grad_crease(w, h, rest, recipe.w_c,
                               psi_deg=recipe.psi, offset=recipe.offset * w)
        elif recipe.preset == "sine":
            a, b = grad_sine(w, h, rest, recipe.lam_f * w,
                             psi_deg=recipe.psi, phase=recipe.phase)
        else:
            a, b = grad_crumple(w, h, rest, recipe.lam_f * w,
                                rng=np.random.default_rng(recipe.seed),
                                persistence=PERSISTENCE)
        return zx + a, zy + b

    return make


def build(recipe: Recipe) -> Sample:
    """레시피 하나를 이미지와 정답 제어점으로 조립한다."""
    rng = np.random.default_rng(recipe.seed)
    clean = render_clean(recipe.text, recipe.d_m0, H_OBS)
    _, w_flat = clean.shape

    s_t = math.sqrt(max((recipe.d_m0 / recipe.d_t) ** 2 - 1.0, 1e-9))
    s_max = slope_budget(recipe.d_m0, C_MIN)

    w_obs, zx, zy = fit_obs_width(_make_grad(recipe, s_t, rng), w_flat, H_OBS)

    # S_t 는 상한이 아니라 목표다 — 정확히 맞춘다.
    peak = float(np.abs(zx).max())
    if peak > 1e-9:
        zx, zy = zx * (s_t / peak), zy * (s_t / peak)
    zx, zy, scale = limit_slope(zx, zy, s_max)

    m, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat)
    g = build_G(s_hat)

    obs, sat = shade(obs, zx, zy, recipe.light, K_A, K_D, recipe.ks, recipe.p)

    obs, g = geometric_rotate(obs, g, recipe.rot_deg)
    obs, g, u_lo, u_hi = geometric_margin(obs, g, *recipe.margin)

    dst, src = control_points(g, recipe.n_x, recipe.n_y, obs.shape, u_lo, u_hi)

    obs = photometric(obs, rng, recipe.sigma, recipe.noise, recipe.jpeg)
    return Sample(obs, dst, src, float(m.min()), sat, scale, w_flat)


def recipe_to_dict(r: Recipe) -> dict:
    d = asdict(r)
    d["light"] = list(r.light)
    d["margin"] = list(r.margin)
    return d


def recipe_from_dict(d: dict) -> Recipe:
    d = dict(d)
    d["light"] = tuple(d["light"])
    d["margin"] = tuple(d["margin"])
    return Recipe(**d)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_synthesis.py -v`
Expected: PASS (8개)

> `test_slope_target_is_hit_not_just_bounded` 가 실패하면 `limit_slope` 가 자주 발동한다는
> 뜻이다. `scale < 1` 이 된 비율을 세어 보라 — 높으면 분포 설계가 틀렸다는 신호다 (설계 §3).

- [ ] **Step 5: 커밋**

메시지: `[합성] 레시피 정의와 조립 — 설계 §5 순서, §6 v1 분포`
대상: `wemeet/data/synthesis.py`, `tests/test_synthesis.py`

---

## Task 8: `scripts/label_recipes.py` — 구간 라벨링과 비중 채우기

**Files:**
- Create: `scripts/label_recipes.py`
- Test: `tests/test_labeling.py`
- Modify: `tests/test_augment.py` (설계 §10-4 픽셀 비교 테스트 추가)

**Interfaces:**
- Consumes: Task 7 의 `build`, `draw_recipe`, `BUCKETS`, `Sample`
- Produces:
  - `tps_flow(src_px, dst_px, shape) -> tuple[np.ndarray, np.ndarray]`
  - `rectify(obs, dst_norm, src_norm, out_shape) -> np.ndarray`
  - `decode(img) -> str | None`
  - `label_sample(sample, out_shape, tau: float) -> str` — `"first_ok" | "target" | "hard" | "burned"`
  - `fill_bucket(bucket, per_band: dict, render_cap: int, seed: int, tau: float) -> tuple[list, dict]` — `(레시피 목록, 통계)`

**여기가 "펴는 코드" 가 사는 곳이다.** `wemeet/data/` 는 이걸 import 하지 않는다 — `scripts/` 는
의존 규칙 밖이다 (설계 §1). SW파트가 `wemeet/sw/rectify.py` 를 내면 아래 `_USE_SW_RECTIFY`
자리에서 갈아끼운다.

**구간 판정과 단락:**

```
1차 디코딩 성공  ->  "first_ok"          (보정·재디코딩을 건너뛴다 = 단락)
1차 실패 & 보정 후 성공  ->  "target"
1차 실패 & 보정 후 실패 & 포화 <= tau  ->  "hard"
1차 실패 & 보정 후 실패 & 포화 >  tau  ->  "burned"   (버린다)
```

**단락은 선택이 아니라 필수다.** 1차 성공률이 높은 고해상도에서 이득이 크고,
없으면 `H` 버킷이 20 h 에서 33 h 가 된다 (설계 §7).

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_labeling.py`:

```python
import numpy as np

from scripts.label_recipes import (BANDS, fill_bucket, label_sample, rectify,
                                   tps_flow)
from wemeet.data.synthesis import Sample, build, draw_recipe


def test_tps_flow_is_identity_for_identical_control_points():
    """설계 §10-11."""
    grid = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 60.0], [100.0, 60.0]])
    mx, my = tps_flow(grid, grid, (61, 101))
    gy, gx = np.mgrid[0:61, 0:101]
    assert np.abs(mx - gx).max() < 1e-3
    assert np.abs(my - gy).max() < 1e-3


def test_rectify_returns_requested_shape():
    s = build(draw_recipe(np.random.default_rng(4), "L", 0))
    out = rectify(s.obs, s.dst_norm, s.src_norm, (220, s.w_flat))
    assert out.shape == (220, s.w_flat)
    assert out.dtype == np.uint8


def test_bands_are_exactly_the_four_in_the_spec():
    assert BANDS == ("target", "hard", "first_ok", "burned")


def test_burned_requires_saturation_above_tau(monkeypatch):
    """포화가 tau 를 넘는 불가만 burned 다. 나머지 불가는 hard 로 남긴다."""
    import scripts.label_recipes as mod
    monkeypatch.setattr(mod, "decode", lambda img: None)
    obs = np.zeros((20, 40), dtype=np.uint8)
    dst = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    hot = Sample(obs, dst, dst, 0.5, sat_ratio=0.20, scale=1.0, w_flat=40)
    cool = Sample(obs, dst, dst, 0.5, sat_ratio=0.01, scale=1.0, w_flat=40)
    assert label_sample(hot, (20, 40), tau=0.04) == "burned"
    assert label_sample(cool, (20, 40), tau=0.04) == "hard"


def test_first_ok_short_circuits_the_second_decode(monkeypatch):
    """단락이 없으면 H 버킷이 20h -> 33h 가 된다."""
    import scripts.label_recipes as mod
    calls = []

    def counting_decode(img):
        calls.append(1)
        return "WEMEET0000"

    monkeypatch.setattr(mod, "decode", counting_decode)
    obs = np.zeros((20, 40), dtype=np.uint8)
    dst = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    s = Sample(obs, dst, dst, 0.5, 0.0, 1.0, 40)
    assert label_sample(s, (20, 40), tau=0.04) == "first_ok"
    assert len(calls) == 1


def test_fill_bucket_stops_at_the_render_cap():
    kept, stats = fill_bucket("L", {"target": 2, "hard": 2, "first_ok": 1},
                              render_cap=40, seed=1, tau=0.04)
    assert stats["rendered"] <= 40
    assert len(kept) <= 5
    for band, quota in {"target": 2, "hard": 2, "first_ok": 1}.items():
        assert stats["kept"][band] <= quota


def test_fill_bucket_never_backfills_a_shortfall():
    """H 에서 불가가 모자라도 1차 성공으로 메우지 않는다 (설계 §7)."""
    kept, stats = fill_bucket("L", {"target": 1, "hard": 1, "first_ok": 1},
                              render_cap=30, seed=2, tau=0.04)
    for band, quota in {"target": 1, "hard": 1, "first_ok": 1}.items():
        assert stats["kept"][band] <= quota
```

`tests/test_augment.py` 에 추가 (설계 §10-4):

```python
def test_rotation_with_G_beats_stale_G_on_rectified_pixels():
    """설계 §10-4. 증강 검증에 디코딩을 쓰면 안 된다 — 20도까지 틀려도 읽힌다.

    기준: G 를 같이 변환한 쪽의 평균 차이가 미변환 쪽의 절반 이하.
    """
    from scripts.label_recipes import rectify
    from wemeet.data.render import render_clean
    from wemeet.data.warp import apply_warp

    clean = render_clean("WEMEET0001", module_px=5.5, height_px=220)
    zx, _ = grad_crease(clean.shape[1], 220, slope=1.4, w_c=4.0)
    _, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat)
    g = build_G(s_hat)
    out_shape = (220, clean.shape[1])

    dst0, src0 = control_points(g, 6, 3, obs.shape)
    base = rectify(obs, dst0, src0, out_shape).astype(np.float64)

    rotated, g_rot = geometric_rotate(obs, g, 2.0)
    dst_ok, src_ok = control_points(g_rot, 6, 3, rotated.shape)
    with_g = rectify(rotated, dst_ok, src_ok, out_shape).astype(np.float64)
    stale = rectify(rotated, dst0, src0, out_shape).astype(np.float64)

    err_ok = np.abs(with_g - base).mean()
    err_stale = np.abs(stale - base).mean()
    assert err_ok < err_stale / 2.0
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_labeling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

먼저 `scripts/__init__.py` 를 빈 파일로 만들어야 import 가 된다.

- [ ] **Step 3: 최소 구현**

`scripts/__init__.py` (빈 파일) 과 `scripts/label_recipes.py`:

```python
"""구간 라벨링과 비중 채우기. 오프라인 도구다.

wemeet/data/ 는 이 파일을 import 하지 않는다 — scripts/ 는 의존 규칙 밖이다 (설계 §1).
여기 있는 rectify 는 라벨링 도구다. SW파트가 wemeet/sw/rectify.py 를 내면
_USE_SW_RECTIFY 자리에서 갈아끼운다.
"""

import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np
import zxingcpp

from wemeet.data.synthesis import (BUCKETS, Sample, build, draw_recipe,
                                   recipe_to_dict)

BANDS = ("target", "hard", "first_ok", "burned")
H_OBS = 220

# SW파트가 wemeet/sw/rectify.py 를 내면 여기를 True 로 바꾸고 아래 import 를 쓴다.
# from wemeet.sw.rectify import apply_field
_USE_SW_RECTIFY = False


def tps_flow(src_px, dst_px, shape, reg: float = 0.0):
    """Thin Plate Spline. 제어점에서 픽셀별 "어디서 가져올지" 지도를 만든다."""
    n = len(dst_px)
    d = np.linalg.norm(dst_px[:, None, :] - dst_px[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(d > 0, d ** 2 * np.log(d ** 2), 0.0)
    k += reg * np.eye(n)
    p = np.hstack([np.ones((n, 1)), dst_px])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T

    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    dg = np.linalg.norm(grid[:, None, :] - dst_px[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ug = np.where(dg > 0, dg ** 2 * np.log(dg ** 2), 0.0)
    pg = np.hstack([np.ones((len(grid), 1)), grid])

    out = []
    for axis in (0, 1):
        b = np.concatenate([src_px[:, axis], np.zeros(3)])
        coef = np.linalg.solve(a, b)
        out.append((ug @ coef[:n] + pg @ coef[n:]).reshape(h, w).astype(np.float32))
    return out[0], out[1]


def rectify(obs, dst_norm, src_norm, out_shape):
    """정규화 제어점으로 편다. 픽셀 환산은 w-1, h-1 을 곱한다."""
    ho, wo = out_shape
    hi, wi = obs.shape
    dst_px = dst_norm * np.array([wo - 1, ho - 1])
    src_px = src_norm * np.array([wi - 1, hi - 1])
    mx, my = tps_flow(src_px, dst_px, out_shape)
    return cv2.remap(obs, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def decode(img):
    try:
        res = zxingcpp.read_barcode(img)
        return res.text if res and res.valid else None
    except Exception:
        return None


def label_sample(sample: Sample, out_shape, tau: float) -> str:
    """구간 판정. 1차 성공이면 보정·재디코딩을 건너뛴다 (단락)."""
    if decode(sample.obs) is not None:
        return "first_ok"
    fixed = rectify(sample.obs, sample.dst_norm, sample.src_norm, out_shape)
    if decode(fixed) is not None:
        return "target"
    return "burned" if sample.sat_ratio > tau else "hard"


def fill_bucket(bucket: str, per_band: dict, render_cap: int, seed: int,
                tau: float):
    """버킷 하나를 비중대로 채운다. 모자라면 메우지 않고 부족분으로 남긴다."""
    rng = np.random.default_rng(seed)
    kept, kept_count, seen = [], Counter(), Counter()
    rendered = 0
    sat_of_target = []

    while rendered < render_cap and any(
            kept_count[b] < per_band.get(b, 0) for b in per_band):
        recipe = draw_recipe(rng, bucket, rendered)
        sample = build(recipe)
        rendered += 1
        band = label_sample(sample, (H_OBS, sample.w_flat), tau)
        seen[band] += 1
        if band == "target":
            sat_of_target.append(sample.sat_ratio)
        if band == "burned":
            continue
        if kept_count[band] < per_band.get(band, 0):
            kept_count[band] += 1
            kept.append((recipe, band, sample.sat_ratio, sample.m_min))

    stats = {
        "bucket": bucket,
        "rendered": rendered,
        "hit_cap": rendered >= render_cap,
        "seen": dict(seen),
        "kept": dict(kept_count),
        "shortfall": {b: per_band[b] - kept_count[b] for b in per_band
                      if kept_count[b] < per_band[b]},
        "tau_suggestion": (float(np.percentile(sat_of_target, 95))
                           if sat_of_target else None),
    }
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, choices=sorted(BUCKETS))
    ap.add_argument("--n", type=int, required=True, help="렌더 상한")
    ap.add_argument("--per-band", default="5000/3500/1500",
                    help="target/hard/first_ok")
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bake", default=None,
                    help="평가 세트용. 이 디렉터리에 이미지를 굽는다")
    args = ap.parse_args()

    t, h, f = (int(v) for v in args.per_band.split("/"))
    kept, stats = fill_bucket(args.bucket, {"target": t, "hard": h,
                                            "first_ok": f},
                              args.n, args.seed, args.tau)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_stats": stats}, ensure_ascii=False) + "\n")
        for recipe, band, sat, m_min in kept:
            row = recipe_to_dict(recipe)
            row.update(band=band, sat_ratio=sat, m_min=m_min)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.bake:
        # 평가 세트는 이미지로 굽는다 — 코드가 바뀌어도 같은 사진으로 비교해야
        # 성능 변화가 코드 탓인지 데이터 탓인지 갈린다 (설계 §1).
        os.makedirs(args.bake, exist_ok=True)
        for i, (recipe, band, _, _) in enumerate(kept):
            if band != "target":
                continue
            sample = build(recipe)
            cv2.imwrite(f"{args.bake}/{i:05d}.png", sample.obs)
            np.savez(f"{args.bake}/{i:05d}.npz",
                     dst_norm=sample.dst_norm, src_norm=sample.src_norm)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_labeling.py tests/test_augment.py -v`
Expected: PASS (labeling 7개 + augment 11개)

- [ ] **Step 5: 커밋**

메시지: `[합성] 구간 라벨링과 비중 채우기 — 단락 필수, 부족분은 메우지 않는다`
대상: `scripts/__init__.py`, `scripts/label_recipes.py`, `tests/test_labeling.py`, `tests/test_augment.py`

---

## Task 9: 의존 규칙 확인과 파일럿

**Files:**
- Test: `tests/test_labeling.py` (없음 — 이 태스크는 실행과 문서 갱신이 산출물이다)
- Modify: `docs/superpowers/specs/2026-08-28-synthesis-design.md` §7 의 추정치 표

**Interfaces:**
- Consumes: Task 8 의 `fill_bucket`
- Produces: 측정된 `τ` 와 버킷별 목표 출현율. 이게 본 생성의 시간 예산을 정한다

- [ ] **Step 1: 의존 규칙이 실제로 막는지 확인한다**

`pyproject.toml` 의 계약은 이미 있다 (`data는 sw.decoding까지만 쓴다`). **문제는 import-linter 가
존재하지 않는 모듈을 `forbidden_modules` 에 넣어도 조용히 통과시킨다는 것이다** — 오타를 내면
계약이 무력화된다. 그래서 일부러 어겨서 막히는지 본다.

`wemeet/data/synthesis.py` 맨 위에 임시로 한 줄 넣는다:

```python
from wemeet.sw import rectify  # noqa  ← 일부러 어긴다. 확인 후 지운다
```

Run: `uv run lint-imports`
Expected: **FAIL** — `data는 sw.decoding까지만 쓴다` 계약 위반이 보고된다.

막히지 않으면 계약이 무력한 것이다. `forbidden_modules` 의 문자열을 확인하라.

- [ ] **Step 2: 그 줄을 지우고 통과를 확인한다**

Run: `uv run lint-imports && uv run pytest -q`
Expected: 계약 4개 전부 통과, 테스트 전체 통과

- [ ] **Step 3: 커밋**

메시지: `[합성] 의존 규칙이 실제로 막히는지 확인`
대상: (코드 변경 없음 — 확인만. 커밋할 것이 없으면 건너뛴다)

- [ ] **Step 4: 버킷별 파일럿을 돌린다**

버킷마다 렌더 상한 3000 으로 작게 돌려 **목표 출현율**과 **`τ` 후보**를 잰다.
`per-band` 를 크게 잡아 상한까지 다 돌게 한다.

```
uv run python -m scripts.label_recipes --bucket L --n 3000 --per-band 9999/9999/9999 --out data/pilot/L.jsonl
uv run python -m scripts.label_recipes --bucket M --n 3000 --per-band 9999/9999/9999 --out data/pilot/M.jsonl
uv run python -m scripts.label_recipes --bucket H --n 3000 --per-band 9999/9999/9999 --out data/pilot/H.jsonl
```

각 실행이 `_stats` 를 찍는다. 받아 적을 것:

| 항목 | 어디서 | 왜 중요한가 |
|---|---|---|
| `seen["target"] / rendered` | 버킷별 목표 출현율 | **`H` 의 3% 가 외삽값이다.** 1.5% 면 본 생성이 20 h -> 40 h 가 된다 |
| `tau_suggestion` | 목표 구간 포화율의 95 백분위수 | `τ` 시작값. 지금 문서의 0.04 는 §9 평균에서 찍은 눈대중이다 |
| `seen["burned"] / seen` | 포화로 버려지는 비율 | 너무 크면 `ks`·`p` 분포가 과하다는 신호 |
| 벽시계 시간 | 실행 시간 | 버킷별 라벨링 속도. 문서의 7.2/4.4/2.3 장/초가 맞는지 |

- [ ] **Step 5: 잰 값을 설계 문서에 반영한다**

`docs/superpowers/specs/2026-08-28-synthesis-design.md` §7 의 「초과 생성 비용」 표에서
**추정치를 측정치로 바꾸고, 어느 것이 실측인지 표시한다.** 지금은 `L` 12% / `M` 7.5% /
`H` 3% 가 전부 추정이고 각주로 그렇게 적혀 있다.

`τ` 를 `scripts/label_recipes.py` 의 `--tau` 기본값에도 반영한다.

메시지: `[합성] 파일럿 측정 — 버킷별 목표 출현율과 tau 확정`

---

## 파일럿 다음 — 본 생성

계획의 범위는 여기까지다. 아래는 **코드가 아니라 실행**이므로 태스크로 두지 않는다.

```
평가 세트 1500장 (3 벌 × 500)      8코어 약 4분
학습 세트 30000장 (3 버킷 × 10000)  8코어 약 3.2 h  (파일럿 결과에 따라 최대 5 h)
```

렌더 상한은 파일럿에서 잰 출현율로 다시 잡는다. 문서의 `L` 62,000 / `M` 100,000 /
`H` 250,000 은 추정 출현율에서 나온 값이다.

---

## Self-Review

**1. 스펙 커버리지**

| 설계 절 | 어느 태스크가 |
|---|---|
| §1 파일 분해 | File Structure |
| §1 실행 인터페이스 | Task 8 (CLI), Task 9 (파일럿) |
| §2 좌표 규약 | Global Constraints + Task 3·4 |
| §3 기울기 예산 · `c_min` · persistence · 스케일 다운 | Task 2 |
| §4 광학 (더하기, `k_a`/`k_d` 고정, 포화율) | Task 5 |
| §5 대응장 `G` · 조립 순서 · 두 시그니처 · 여백 | Task 4, 6, 7 |
| §6 v1 분포 · preset 30/30/30/10 | Task 7 |
| §7 구간 넷 · 버킷 층화 · 50/35/15 · 단락 · 렌더 상한 · 부족분 | Task 8 |
| §8 모델 입력 규격 | **범위 밖** — 학습 코드는 `0003` 이다 |
| §9 confidence | **범위 밖** — v1 은 학습하지 않는다 |
| §10 검증 기준 1~9, 11, 13 | Task 3(1·2·6·7), 4(3), 5(8), 6(5), 7(9), 8(4·11), 9(13) |
| §10-10 평가 세트 목표 구간 95% | **파일럿 다음** — 평가 세트를 실제로 뽑을 때 |
| §10-12 계약 | 기존 `tests/test_schemas.py` 22개가 이미 덮는다 |

**2. 빠졌다가 채운 것 — 평가 세트 굽기**

설계 §1 의 `--extract-target` 경로(목표 구간만 골라 **이미지로 굽기**)가 처음 초안에 없었다.
평가 세트는 반드시 이미지로 구워야 한다 — 설계 §1 이 "코드가 바뀌어도 같은 사진으로 비교해야
성능 변화가 코드 탓인지 데이터 탓인지 갈린다" 고 정했다.

**Task 8 Step 3 의 `main()` 에 `--bake <dir>` 로 채워 넣었다.** 별도 태스크로 쪼갤 만큼
크지 않고, `fill_bucket` 이 이미 레시피를 주므로 굽는 루프 열 줄이면 된다.
제어점은 `.npz` 로 이미지 옆에 같이 저장한다.

**3. 타입 정합성** — `Sample` 필드명이 Task 7 정의(`obs`, `dst_norm`, `src_norm`, `m_min`,
`sat_ratio`, `scale`, `w_flat`)와 Task 8 사용처에서 일치. `control_points` 인자 순서
`(g, n_x, n_y, shape, u_lo, u_hi)` 가 Task 4 정의와 Task 6·7 호출에서 일치.

**4. 알려진 취약점**

| | |
|---|---|
| `test_module_width_matches_request` 의 `±0.5px` | `python-barcode` 의 mm 반올림 때문에 정확히 안 맞을 수 있다. 실패하면 허용 오차가 아니라 **`_DPI` 를 올려** 해결하라 |
| `test_slope_target_is_hit_not_just_bounded` 의 `25%` | `limit_slope` 발동 빈도에 달렸다. 자주 실패하면 분포가 예산을 넘는다는 신호이므로 테스트가 아니라 §6 을 고쳐야 한다 |
| Task 6 의 `%` 단위 여백 | `geometric_margin` 이 pad 와 crop 을 한 함수에서 처리한다. 네 방향이 섞이면 인덱스 실수가 나기 쉬우니 테스트 5개를 반드시 먼저 통과시킬 것 |

---

## Execution Handoff

이 계획은 태스크 9개다. 앞의 7개는 서로 독립적이지 않다 — Task 3·4 가 `warp.py` 를
나눠 쓰고, Task 7 이 1~6 전부를 부른다. **순서대로 가야 한다.**
