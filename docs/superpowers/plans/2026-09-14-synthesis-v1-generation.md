# 합성 v1 본 생성 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 물리 치수와 어긋난 상수 셋(`H_OBS`·`w_c`·렌더러 여백)을 고치고, 샤딩으로 16코어를 써서 학습 레시피 30,000장과 구운 평가 세트 1,500장을 만들어 `123metro/barcode-datasets` 에 발행한다.

**Architecture:** 라벨의 물리 치수(50×30 / 60×40 mm)가 처음으로 주어졌으므로, **표면 기하 파라미터는 전부 라벨 폭 상대로 바꾸고 카메라·센서 파라미터만 픽셀 절대로 남긴다.** 높이는 상수가 아니라 `h_flat = round(w_flat / aspect)` 로 유도한다. 생성은 버킷마다 `--shard i/N` 로 쪼개 프로세스 15개가 나눠 돌리고, 샤드 난수는 `default_rng([seed, i])` 로 독립·재현된다. 이미지는 평가 세트만 굽고 학습은 레시피(JSONL)만 남긴다.

**Tech Stack:** Python 3.12, numpy≥2.0, opencv-python≥4.8, python-barcode≥0.15, zxing-cpp≥2.2, huggingface-hub≥0.25, pytest, uv

**Spec:** `docs/superpowers/specs/2026-09-14-synthesis-v1-generation-design.md`
본체 설계는 `docs/superpowers/specs/2026-08-28-synthesis-design.md` 가 계속 정본이다 — 구간 정의·비중·광학·기울기 예산은 그 문서를 본다.

---

## Global Constraints

이 절의 규칙은 **모든 태스크의 요구사항에 암묵적으로 포함된다.**

### 좌표 규약 — 틀리면 거꾸로 펴진다

| 항목 | 규칙 |
|---|---|
| 단위 | **0.0 ~ 1.0 정규화.** 픽셀 아님 |
| 원점 | `0.0` = 첫 픽셀 중심, `1.0` = 마지막 픽셀 중심. 픽셀 환산은 **`w−1`, `h−1` 을 곱한다** |
| 순서 | `(x, y)`. `y` 는 **아래로** 증가 |
| 방향 | `dst` = 펴진 격자, `src` = 지금 있는 위치 |
| `src` 범위 | `−0.5 ~ 1.5` — 회전 잔차 때문에 크롭 밖을 가리킬 수 있다 |

### 절대 틀리면 안 되는 네 줄

```python
s = np.cumsum(1.0 / m, axis=1)          # cumsum(m) 이 아니다
I = base * (ka + kd * ndl) + ks * spec  # 정반사는 더하기
z_x *= min(1.0, S_max / abs(z_x).max()) # clip(m) 금지
G[v, u] = (x_obs, y_obs)                # 방향은 펴진→관측
```

### 상대냐 절대냐 — 새 파라미터를 추가할 때 이것부터 정한다 (설계 §1.4)

> **표면 기하(라벨에 붙어 있는 것)는 라벨 폭 상대여야 하고,
> 카메라·센서·디코더의 성질은 픽셀·계조 절대여야 한다.**

`lam_f·w` / `offset·w` / **`w_c_f·w`** / `aspect` 는 상대. `sigma`(초점 블러 px) /
`noise`(계조) / `jpeg` / `c_min`(판독 하한 px) / `d_m0` / `d_t` 는 **절대가 맞다.**

### 고정 상수 (샘플마다 안 뽑는다)

| | 값 | 근거 |
|---|---|---|
| `k_a` / `k_d` | **0.35** / **0.50** | 설계 §4 |
| `c_min` | **1.0 px** | 설계 §3 |
| `persistence` | **0.7** | 설계 §3 |
| 제어점 격자 | **`n_x=16`, `n_y=3`** | `0003` (2026-09-14 확정) |
| `G` 해상도 | **`n_v=33`, `n_u=513`** | 설계 §5 |
| 종횡비 범위 | **`U(2.0, 2.3)`** | 설계 §1.1 |
| 주름 폭 범위 | **`w_c_f ~ U(0.003, 0.029)`** | 설계 §1.2 |
| 텍스트 wrap | **`index % 10000`** | 설계 §2 |
| 샤드 수 | **`L` 2 / `M` 6 / `H` 7 = 15** | 설계 §4 |

**`H_OBS` 는 삭제한다. 220 을 「값으로 쓰는」 코드는 어디에도 없어야 한다** — 기본 인자,
상수, 리터럴 전부. 높이는 언제나 `w_flat / aspect` 에서 나온다.

> **산문은 예외다.** 왜 지웠는지 설명하는 주석·독스트링·문서에는 숫자를 적어도 된다 —
> 오히려 적어야 한다. "예전의 고정 높이 상수" 라고만 쓰면 읽는 사람이 어느 상수였는지
> 모른다. 검사는 `grep 220` 이 아니라 **"기본값 없는 시그니처 + 유도식"** 으로 한다.

### 의존 규칙

```
wemeet/data/ 는 wemeet.sw.rectify 를 import 하지 않는다
```

`scripts/` 와 `tests/` 는 규칙 밖이다. 라벨링용 `rectify` 는 `scripts/label_recipes.py` 안에 둔다.

### 커밋

각 태스크 끝에서 커밋한다. 메시지는 `[합성] ` 또는 `[데이터] ` 로 시작하고, 끝에
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` 를 붙인다.

---

## File Structure

| 파일 | 책임 | 태스크 |
|---|---|---|
| `wemeet/data/render.py` | `python-barcode` 의존을 가둔다. 모듈 개수 계산과 여백 없는 렌더 | 1 |
| `wemeet/data/synthesis.py` | 레시피 샘플링과 조립. 버킷 정의 | 2 |
| `scripts/label_recipes.py` | 구간 라벨링·비중 채우기·굽기. 오프라인 도구 | 3, 4 |
| `scripts/generate_v1.py` | **신규.** 샤드 병렬 실행과 `_stats` 병합 | 5 |
| `data/hf/README.md` | **신규.** 데이터셋 카드 | 9 |
| `.gitignore` | 생성물 제외, 통계는 포함 | 5 |

`surface.py` · `warp.py` · `optics.py` · `augment.py` 는 **건드리지 않는다.** 상대/절대
문제는 전부 호출부(`synthesis.py`)에서 값을 어떻게 넘기느냐의 문제였다.

---

## Task 1: `render.py` — 여백 제거와 모듈 개수

**Files:**
- Modify: `wemeet/data/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Produces: `module_count(text: str) -> int` (콰이엇 존 20 포함),
  `render_clean(text: str, module_px: float, height_px: int) -> np.ndarray`
  — **`height_px` 에 기본값을 두지 않는다.** 220 이 다시 스며들 자리를 없앤다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`height_px` 의 기본값이 사라지므로 `tests/test_render.py` 의 기존 호출을 먼저 고친다.
**220 을 이 파일에서 완전히 없앤다** — 테스트에 남겨두면 다시 정당한 값처럼 보인다.

| 기존 테스트 | 고칠 것 |
|---|---|
| `test_render_clean_shape_and_dtype` | `height_px=220` → `height_px=200`, `assert img.shape[0] == 220` → `== 200` |
| `test_base_module_width_exact` | `height_px=200` 추가 |
| `test_effective_module_width_fractional` | 두 `render_clean` 호출에 `height_px=200` 추가 |
| `test_module_px_1_0_does_not_crash` | `height_px=200` 추가, `assert img.shape[0] == 220` → `== 200` |
| `test_module_px_1_2_does_not_crash` | `height_px=200` 추가, `assert img.shape[0] == 220` → `== 200` |
| `test_module_width_scales` | 두 호출에 `height_px=200` 추가 |

그다음 아래 넷을 파일 끝에 추가한다.

```python
from wemeet.data.render import module_count, render_clean


def test_render_clean_has_no_white_padding():
    """첫 행과 끝 행이 둘 다 막대를 포함한다 — 렌더러 여백이 잘려 나갔다는 뜻."""
    img = render_clean("WEMEET0001", module_px=4.0, height_px=200)
    assert img[0].min() < 128, "첫 행이 전부 흰색이다 — 여백이 남아 있다"
    assert img[-1].min() < 128, "끝 행이 전부 흰색이다 — 여백이 남아 있다"


def test_vertical_resize_does_not_change_columns():
    """세로 크기를 바꿔도 가로 내용이 그대로다 — 여백이 빠졌다는 증거.

    여백이 있으면 리사이즈가 흰 행과 막대 행을 섞는 비율을 바꿔 열 프로파일이
    통째로 흔들린다 (실측 최대 172 계조). 제거 후에는 0.64 계조다.

    "모든 행이 정확히 같은가" 로 재면 안 된다 — cv2.resize 의 반올림 때문에
    크기에 따라 1 계조짜리 행이 두 종류 나오고, 그건 이 수정과 무관하다.
    """
    for module_px in (1.6, 2.0, 4.0, 6.0):
        short = render_clean("WEMEET0001", module_px, height_px=150)
        tall = render_clean("WEMEET0001", module_px, height_px=460)
        drift = float(np.abs(short.mean(axis=0) - tall.mean(axis=0)).max())
        assert drift < 2.0, f"module_px={module_px}: {drift:.2f} 계조"


def test_module_count_matches_rendered_width():
    """래스터화 없이 센 모듈 개수가 실제 렌더 폭과 일치한다."""
    text = "WEMEET0001"
    m = module_count(text)
    for module_px in (1.6, 2.0, 3.7, 6.0):
        img = render_clean(text, module_px, height_px=200)
        assert img.shape[1] == round(m * module_px), f"module_px={module_px}"


def test_module_count_is_constant_across_wrapped_texts():
    """WEMEET0000~9999 는 전부 같은 모듈 개수여야 버킷이 곧 해상도다 (설계 §2)."""
    counts = {module_count(f"WEMEET{i:04d}") for i in range(0, 10000, 13)}
    assert counts == {154}, counts
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_render.py -q`
Expected: FAIL — `ImportError: cannot import name 'module_count'`

- [ ] **Step 3: 최소 구현**

`wemeet/data/render.py` 를 아래로 바꾼다.

```python
"""깨끗한 바코드 렌더링. python-barcode 의존을 이 파일에 가둔다."""

import io

import barcode
import cv2
import numpy as np
from barcode.writer import ImageWriter

_DPI = 300
_BASE_MODULE_PX = 8  # 기본 렌더 폭 — 정수 픽셀 경계로 정확함
_QUIET_MODULES = 20  # 양쪽 10 모듈씩. 아래 quiet_zone 과 같은 값이어야 한다


def module_count(text: str) -> int:
    """콰이엇 존을 포함한 총 모듈 개수. 래스터화하지 않는다.

    python-barcode 의 인코더가 모듈 문자열을 그대로 준다. 종횡비를 유도하려면
    렌더 '전에' 폭을 알아야 하는데, 이 방법이면 이미지를 만들지 않고 알 수 있다.
    """
    return len(barcode.get("code128", text).build()[0]) + _QUIET_MODULES


def render_clean(text: str, module_px: float, height_px: int) -> np.ndarray:
    """모듈 폭이 정확히 module_px 가 되는 Code128 을 그린다.

    height_px 에 기본값을 두지 않는다 — 예전의 H_OBS=220 은 물리적으로 틀린
    상수였다 (설계 §1.1). 호출부가 종횡비에서 유도한 값을 반드시 넘겨야 한다.
    """
    base_module_mm = _BASE_MODULE_PX * 25.4 / _DPI
    quiet_zone_mm = (_QUIET_MODULES // 2) * base_module_mm

    writer = ImageWriter()
    obj = barcode.get("code128", text, writer=writer)
    buf = io.BytesIO()
    obj.write(buf, options={
        "module_width": base_module_mm,
        "module_height": 12.0,
        "quiet_zone": quiet_zone_mm,
        "write_text": False,
        "dpi": _DPI,
    })
    buf.seek(0)
    arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # ImageWriter 가 위아래에 흰 여백을 6.67% 씩 붙인다 (설계 §1.3). 그대로 두면
    # 종횡비가 15% 어긋나고, 흰 화소가 정반사에서 포화해 sat_ratio 를 부풀리고,
    # geometric_margin 이 이미 모사하는 검출기 여백과 이중 계상된다.
    # 비율을 상수로 박지 않고 이미지에서 찾는다 — 라이브러리가 바뀌어도 맞는다.
    rows = np.where(arr.min(axis=1) < 128)[0]
    arr = arr[rows.min():rows.max() + 1]

    # 모듈 개수는 '렌더된 폭' 에서 낸다. module_count() 를 다시 부르면 위의
    # test_module_count_matches_rendered_width 가 동어반복이 된다 -- 그 테스트의
    # 값어치는 인코더 경로와 라이터 경로가 독립으로 같은 답을 낸다는 데 있다.
    M = arr.shape[1] / _BASE_MODULE_PX
    target_width = int(round(M * module_px))
    return cv2.resize(arr, (target_width, height_px), interpolation=cv2.INTER_AREA)
```

> `build()` 인코더는 1회 0.014 ms 로 ImageWriter 렌더(2.30 ms)의 **0.6%** 다 — 비용 때문에
> 이렇게 하는 것이 아니라 **테스트를 살리려고** 이렇게 한다.

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_render.py -q && uv run ruff check wemeet/data/render.py`
Expected: 10 passed, ruff 통과

- [ ] **Step 5: 커밋**

```bash
git add wemeet/data/render.py tests/test_render.py
git commit -m "[합성] 렌더러 흰 여백 제거와 module_count — 종횡비를 유도할 수 있게 한다"
```

---

## Task 2: `synthesis.py` — 종횡비·주름 폭·텍스트 wrap·평가 버킷

**Files:**
- Modify: `wemeet/data/synthesis.py`
- Modify: `scripts/label_recipes.py` (import 과 `out_shape` 만 — `H_OBS` 가 사라지므로)
- Modify: `tests/test_labeling.py` (같은 이유)
- Test: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: Task 1 의 `module_count`, `render_clean(text, module_px, height_px)`
- Produces:
  - `BUCKETS`, `EVAL_BUCKETS`, `ALL_BUCKETS` (dict[str, tuple[float, float]])
  - `Recipe` — `w_c` 가 사라지고 `w_c_f: float`, `aspect: float` 이 생긴다
  - `Sample` — `w_flat` 다음에 `h_flat: int` 이 생긴다
  - `draw_recipe(rng, bucket: str, index: int) -> Recipe`
  - `build(recipe: Recipe) -> Sample`
  - **`H_OBS` 는 없다**

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_synthesis.py` 파일 끝에 추가한다.

```python
import dataclasses

import numpy as np
import pytest

from wemeet.data.synthesis import (
    ALL_BUCKETS, ASPECT_HI, ASPECT_LO, BUCKETS, EVAL_BUCKETS,
    Recipe, _make_grad, build, draw_recipe,
)


def test_h_obs_is_gone():
    """220 은 물리적으로 틀린 상수였다. 다시 스며들지 못하게 잠근다."""
    import wemeet.data.synthesis as syn
    assert not hasattr(syn, "H_OBS")


def test_aspect_is_drawn_in_range():
    rng = np.random.default_rng(0)
    for bucket in BUCKETS:
        for i in range(30):
            r = draw_recipe(rng, bucket, i)
            assert ASPECT_LO <= r.aspect <= ASPECT_HI


def test_flat_label_aspect_matches_recipe():
    """펴진 라벨의 w/h 가 레시피의 aspect 다 (설계 검증 #2).

    관측 크롭이 아니라 펴진 것을 잰다 — 감기면 좁아 보이는 것이 물리적으로 맞다.
    """
    rng = np.random.default_rng(1)
    for bucket in BUCKETS:
        for i in range(10):
            r = draw_recipe(rng, bucket, i)
            s = build(r)
            assert s.w_flat / s.h_flat == pytest.approx(r.aspect, abs=0.02)
            assert ASPECT_LO - 0.02 <= s.w_flat / s.h_flat <= ASPECT_HI + 0.02


def test_crease_transition_is_relative_to_width():
    """주름 전이가 이미지 폭의 '같은 비율' 을 차지한다 (설계 검증 #4).

    w_c 가 절대 픽셀이던 시절에는 폭이 3배가 되면 비율이 1/3 로 줄었다 —
    고해상도 버킷만 물리적으로 3.75배 날카로운 주름을 받았다.
    cyl_share=0 으로 두어 원통 성분을 끄고 주름만 본다.
    """
    r = Recipe(
        seed=1, bucket="L", text="WEMEET0000", d_m0=2.0, d_t=1.5, aspect=2.15,
        preset="crease", cyl_share=0.0, psi=0.0, w_c_f=0.029, lam_f=1.0,
        phase=0.0, offset=0.0, light=(0.0, 0.0, 1.0), ks=0.0, p=100.0,
        sigma=0.0, noise=0.0, jpeg=90, rot_deg=0.0, margin=(0.0, 0.0, 0.0, 0.0),
    )
    make = _make_grad(r, s_t=1.0)
    fractions = []
    for w in (600, 1800):
        zx, _ = make(w, 150)
        row = zx[75]
        peak = float(np.abs(row).max())
        inside = np.where(np.abs(row) < 0.8 * peak)[0]
        fractions.append((inside.max() - inside.min() + 1) / w)
    assert fractions[0] == pytest.approx(fractions[1], rel=0.05), fractions
    # 분석해: 2*atanh(0.8)*w_c_f = 0.0637
    assert fractions[0] == pytest.approx(2 * np.arctanh(0.8) * r.w_c_f, rel=0.05)


def test_text_wraps_at_10000():
    """10,000번째부터 텍스트가 길어지면 w_flat 이 d_m0 과 무관하게 움직인다 (설계 §2)."""
    a = draw_recipe(np.random.default_rng(0), "L", 0)
    b = draw_recipe(np.random.default_rng(0), "L", 10000)
    c = draw_recipe(np.random.default_rng(0), "L", 107999)
    assert a.text == b.text == "WEMEET0000"
    assert c.text == "WEMEET7999"


def test_eval_buckets_are_inside_train_buckets():
    """학습 분포가 평가 3벌을 덮어야 한다 — 안 덮으면 성능 차이가 분포 불일치가 된다."""
    for ev, tr in (("low", "L"), ("mid", "M"), ("high", "H")):
        elo, ehi = EVAL_BUCKETS[ev]
        tlo, thi = BUCKETS[tr]
        assert tlo <= elo and ehi <= thi, f"{ev} 가 {tr} 밖으로 나간다"
    assert set(ALL_BUCKETS) == set(BUCKETS) | set(EVAL_BUCKETS)


def test_recipe_roundtrip_rejects_old_recipes():
    """aspect·w_c_f 없는 옛 JSON 은 조용히 통과하면 안 된다 (설계 §8).

    옛 레시피를 새 파이프라인으로 재생성하면 다른 이미지가 나오므로,
    조용한 성공이 조용한 오염이 된다.
    """
    from wemeet.data.synthesis import recipe_from_dict, recipe_to_dict
    r = draw_recipe(np.random.default_rng(2), "M", 3)
    d = recipe_to_dict(r)
    assert recipe_from_dict(d) == r
    old = {k: v for k, v in d.items() if k not in ("aspect", "w_c_f")}
    old["w_c"] = 4.0
    with pytest.raises(TypeError):
        recipe_from_dict(old)


def test_sample_carries_h_flat():
    s = build(draw_recipe(np.random.default_rng(3), "L", 0))
    assert isinstance(s.h_flat, int) and s.h_flat > 0
    assert dataclasses.fields(type(s))[7].name == "h_flat"
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_synthesis.py -q`
Expected: FAIL — `ImportError: cannot import name 'ALL_BUCKETS'`

- [ ] **Step 3: 최소 구현**

`wemeet/data/synthesis.py` 에서 상수·`Recipe`·`Sample`·`draw_recipe`·`_make_grad`·`build`
여섯 군데를 고친다. 나머지(`recipe_to_dict`, `recipe_from_dict`)는 그대로 둔다.

```python
from wemeet.data.render import module_count, render_clean

BUCKETS = {"L": (1.6, 2.5), "M": (2.5, 3.7), "H": (3.7, 6.0)}
EVAL_BUCKETS = {"low": (1.6, 2.2), "mid": (2.5, 3.5), "high": (4.0, 6.0)}
ALL_BUCKETS = BUCKETS | EVAL_BUCKETS
PRESETS = ("crease", "sine", "octave", "cylinder")
PRESET_P = (0.30, 0.30, 0.30, 0.10)

K_A, K_D = 0.35, 0.50
C_MIN = 1.0
PERSISTENCE = 0.7
N_X, N_Y = 16, 3

# 라벨 종횡비. 현장 실측 50x30 / 60x40 mm 에서 바 영역이 높이의 73~75% 라는
# 가정으로 2.27 / 2.00 이 나오고, 이 범위가 둘을 덮는다 (설계 §1.1).
# 예전의 H_OBS=220 은 라벨 높이를 71.5/d_m0 mm 로 만들어 L 44.7mm, H 11.9mm 로
# 3.75배 흔들었다. 상수로 두면 안 되는 값이었다.
ASPECT_LO, ASPECT_HI = 2.0, 2.3

# 주름 전이 폭. 라벨 폭의 비율이다 -- 픽셀로 두면 고해상도 버킷만 물리적으로
# 3.75배 날카로운 주름을 받는다 (설계 §1.2). L 버킷의 옛 px 분포를 보존한다.
WCF_LO, WCF_HI = 0.003, 0.029

# 텍스트가 길어지면 w_flat 이 d_m0 과 무관하게 움직여 "버킷 = 해상도" 가 깨진다.
TEXT_MOD = 10000
```

`Recipe` 에서 `w_c: float` 를 지우고 `w_c_f: float` 와 `aspect: float` 를 넣는다
(위치는 `cyl_share` 다음, `psi` 다음이면 된다 — 아래 `draw_recipe` 의 키워드 인자와 맞으면
순서는 자유다).

`Sample` 은 `w_flat: int` 바로 다음에 `h_flat: int` 를 넣는다. `g`/`u_lo`/`u_hi` 는 기본값이
있으므로 그 앞이어야 한다.

```python
@dataclass
class Sample:
    obs: np.ndarray
    dst_norm: np.ndarray
    src_norm: np.ndarray
    m_min: float
    sat_ratio: float
    scale: float
    w_flat: int
    h_flat: int
    g: np.ndarray | None = None
    u_lo: float = 0.0
    u_hi: float = 1.0
```

`draw_recipe`:

```python
def draw_recipe(rng: np.random.Generator, bucket: str, index: int) -> Recipe:
    """설계 §6 v1 분포. d_m0 만 버킷 범위에서 뽑는다."""
    lo, hi = ALL_BUCKETS[bucket]
    d_m0 = float(rng.uniform(lo, hi))
    d_t = float(rng.uniform(1.3, min(2.6, 0.97 * d_m0)))
    return Recipe(
        seed=int(rng.integers(0, 2 ** 31 - 1)),
        bucket=bucket,
        text=f"WEMEET{index % TEXT_MOD:04d}",
        d_m0=d_m0,
        d_t=d_t,
        aspect=float(rng.uniform(ASPECT_LO, ASPECT_HI)),
        preset=str(rng.choice(PRESETS, p=PRESET_P)),
        cyl_share=float(rng.uniform(0.15, 0.45)),
        psi=float(rng.uniform(-45.0, 45.0)),
        w_c_f=float(rng.uniform(WCF_LO, WCF_HI)),
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
```

`_make_grad` 안의 `grad_crease` 호출 한 줄만 바꾼다.

```python
            a, b = grad_crease(w, h, rest, recipe.w_c_f * w,
                               psi_deg=recipe.psi, offset=recipe.offset * w)
```

`build` 의 앞 세 줄과 마지막 줄을 바꾼다.

```python
def build(recipe: Recipe) -> Sample:
    """레시피 하나를 이미지와 정답 제어점으로 조립한다."""
    rng = np.random.default_rng(recipe.seed)
    w_flat = int(round(module_count(recipe.text) * recipe.d_m0))
    h_flat = int(round(w_flat / recipe.aspect))
    clean = render_clean(recipe.text, recipe.d_m0, h_flat)

    s_t = math.sqrt(max((recipe.d_m0 / recipe.d_t) ** 2 - 1.0, 1e-9))
    s_max = slope_budget(recipe.d_m0, C_MIN)

    _, zx, zy = fit_obs_width(_make_grad(recipe, s_t), w_flat, h_flat)

    # ... (peak 정규화부터 photometric 까지 기존과 동일) ...

    return Sample(obs, dst, src, float(m.min()), sat, scale, w_flat, h_flat,
                  g, u_lo, u_hi)
```

`scripts/label_recipes.py` 두 곳:

```python
from wemeet.data.synthesis import ALL_BUCKETS, Sample, build, draw_recipe, recipe_to_dict
...
        band = label_sample(sample, (sample.h_flat, sample.w_flat), tau)
```

`tests/test_labeling.py` 세 곳: `H_OBS` import 를 지우고, `(H_OBS, sample.w_flat)` 를
`(sample.h_flat, sample.w_flat)` 로, `Sample(...)` 위치인자 생성에 `h_flat` 을 넣는다
(`Sample(obs, dst, dst, 0.5, 0.0, 1.0, 40, 20)`).

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest -q && uv run lint-imports && uv run ruff check .`
Expected: 전체 통과, 계약 4개 KEPT

- [ ] **Step 5: 커밋**

```bash
git add wemeet/data/synthesis.py scripts/label_recipes.py tests/test_synthesis.py tests/test_labeling.py
git commit -m "[합성] 종횡비 유도와 주름 폭 상대화 — H_OBS 를 버린다"
```

---

## Task 3: `label_recipes.py` — 평가 버킷과 굽기 메타데이터

**Files:**
- Modify: `scripts/label_recipes.py`
- Test: `tests/test_labeling.py`

**Interfaces:**
- Consumes: Task 2 의 `ALL_BUCKETS`, `Sample.h_flat`
- Produces: `--bucket` 이 `low|mid|high` 도 받는다. `--bake <dir>` 가
  `<dir>/manifest.jsonl` 을 쓴다 — 한 줄이 한 장이고 `file`·`npz`·`band`·`sat_ratio`·
  `m_min`·`w_flat`·`h_flat` 과 레시피 전체 필드(`text` 포함)를 담는다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_labeling.py` 끝에 추가한다.

```python
import json

from wemeet.data.synthesis import ALL_BUCKETS
from scripts.label_recipes import bake, fill_bucket


def test_eval_buckets_are_selectable():
    assert {"low", "mid", "high"} <= set(ALL_BUCKETS)


def test_bake_writes_manifest_with_ground_truth(tmp_path):
    """정답 번호 없이는 디코딩률을 못 잰다 — manifest 의 존재 이유다 (설계 §3)."""
    kept, _ = fill_bucket("low", {"target": 1}, 400, seed=7, tau=0.08)
    assert kept, "400 렌더 안에 목표 구간이 하나도 없다 — 출현율을 의심하라"
    bake(kept, tmp_path)

    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    for key in ("file", "npz", "text", "band", "sat_ratio", "m_min",
                "aspect", "w_c_f", "preset", "d_m0", "seed"):
        assert key in row, key
    assert row["band"] == "target"
    assert (tmp_path / row["file"]).exists()
    assert (tmp_path / row["npz"]).exists()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_labeling.py -q`
Expected: FAIL — `ImportError: cannot import name 'bake'`

- [ ] **Step 3: 최소 구현**

`scripts/label_recipes.py` 의 `main()` 안에 있던 굽기 루프를 함수로 빼내고 manifest 를
추가한다.

```python
def bake(kept, out_dir) -> int:
    """목표 구간만 이미지로 굽는다. 평가 세트 전용.

    코드가 바뀌어도 같은 사진으로 비교해야 성능 변화가 코드 탓인지 데이터 탓인지
    갈린다 (설계 §1). preset 과 열화 파라미터를 같이 남겨 사후에 쪼개 볼 수 있게 한다.
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(f"{out_dir}/manifest.jsonl", "w", encoding="utf-8") as fh:
        for recipe, band, sat, m_min in kept:
            if band != "target":
                continue
            sample = build(recipe)
            name = f"{n:05d}"
            cv2.imwrite(f"{out_dir}/{name}.png", sample.obs)
            np.savez(f"{out_dir}/{name}.npz",
                     dst_norm=sample.dst_norm, src_norm=sample.src_norm)
            row = recipe_to_dict(recipe)
            row.update(file=f"{name}.png", npz=f"{name}.npz", band=band,
                       sat_ratio=sat, m_min=m_min,
                       w_flat=sample.w_flat, h_flat=sample.h_flat)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
```

`main()` 의 `--bucket` 선택지를 `sorted(ALL_BUCKETS)` 로 바꾸고, `--bake` 처리를
`baked = bake(kept, args.bake)` 한 줄로 줄인다. `import cv2`·`import json` 은 이미 있다.

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_labeling.py -q`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/label_recipes.py tests/test_labeling.py
git commit -m "[합성] 평가 버킷과 굽기 메타데이터 — 정답 번호를 manifest 에 남긴다"
```

---

## Task 4: `label_recipes.py` — `--shard i/N`

**Files:**
- Modify: `scripts/label_recipes.py`
- Test: `tests/test_labeling.py`

**Interfaces:**
- Produces:
  - `split_counts(total: int, shards: int) -> list[int]`
  - `fill_bucket(bucket, per_band, render_cap, seed, tau, shard=0, shards=1)`
    — `render_cap` 과 `per_band` 는 **전체 값**이고 내부에서 쪼갠다.
    `stats` 에 `target_sats: list[float]` 와 `target_presets: dict[str, int]` 를 담는다
    (드라이버가 샤드를 합쳐 백분위수를 한 번에 내야 하므로 원자료를 준다)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
import numpy as np

from scripts.label_recipes import fill_bucket, split_counts


def test_split_counts_distributes_remainder_to_front():
    assert split_counts(10, 3) == [4, 3, 3]
    assert split_counts(9, 3) == [3, 3, 3]
    assert sum(split_counts(5000, 7)) == 5000
    assert max(split_counts(5000, 7)) - min(split_counts(5000, 7)) <= 1


def _seeds(bucket, shard, shards, seed=5, cap=24):
    _, stats = fill_bucket(bucket, {"target": 10**9}, cap, seed, 0.08,
                           shard=shard, shards=shards)
    return [r.seed for r, _, _, _ in stats["labelled"]]


def test_shard_is_reproducible():
    assert _seeds("L", 0, 3) == _seeds("L", 0, 3)


def test_shards_are_independent():
    assert _seeds("L", 0, 3) != _seeds("L", 1, 3)


def test_shards_together_render_the_whole_cap():
    total = sum(len(_seeds("L", i, 3, cap=24)) for i in range(3))
    assert total == 24


def test_shard_rng_matches_documented_scheme():
    """default_rng([seed, shard]) 가 독립이면서 재현되는지 (설계 §5)."""
    a = np.random.default_rng([42, 0]).random(3)
    b = np.random.default_rng([42, 1]).random(3)
    c = np.random.default_rng([42, 0]).random(3)
    assert not np.allclose(a, b)
    assert np.allclose(a, c)
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_labeling.py -q -k "shard or split"`
Expected: FAIL — `ImportError: cannot import name 'split_counts'`

- [ ] **Step 3: 최소 구현**

```python
def split_counts(total: int, shards: int) -> list[int]:
    """샤드별 몫. 나머지는 앞쪽 샤드에 하나씩 더 준다."""
    base, rem = divmod(total, shards)
    return [base + (1 if i < rem else 0) for i in range(shards)]


def fill_bucket(bucket: str, per_band: dict, render_cap: int, seed: int,
                tau: float, shard: int = 0, shards: int = 1):
    """버킷 하나를 비중대로 채운다. 모자라면 메우지 않고 부족분으로 남긴다.

    render_cap 과 per_band 는 '전체' 값이고 여기서 샤드 몫으로 쪼갠다. 샤드마다
    독립 난수 스트림(default_rng([seed, shard]))을 쓰고, 레시피 인덱스는 앞선
    샤드들의 몫만큼 밀어 겹치지 않게 한다.

    레시피는 기록의 원본이다 — burned 나 쿼터를 넘긴 first_ok 도 버리지 않고
    stats["labelled"] 에 전부 남긴다. "버림" 은 학습 로더의 일이다.
    """
    caps = split_counts(render_cap, shards)
    my_cap = caps[shard]
    index_offset = sum(caps[:shard])
    my_band = {b: split_counts(n, shards)[shard] for b, n in per_band.items()}

    rng = np.random.default_rng([seed, shard])
    kept, kept_count, seen = [], Counter(), Counter()
    labelled = []
    rendered = 0
    sat_of_target = []

    while rendered < my_cap and any(
            kept_count[b] < my_band.get(b, 0) for b in my_band):
        recipe = draw_recipe(rng, bucket, index_offset + rendered)
        sample = build(recipe)
        rendered += 1
        band = label_sample(sample, (sample.h_flat, sample.w_flat), tau)
        seen[band] += 1
        labelled.append((recipe, band, sample.sat_ratio, sample.m_min))
        if band == "target":
            sat_of_target.append(sample.sat_ratio)
        if band == "burned":
            continue
        if kept_count[band] < my_band.get(band, 0):
            kept_count[band] += 1
            kept.append((recipe, band, sample.sat_ratio, sample.m_min))

    stats = {
        "bucket": bucket,
        "shard": shard,
        "shards": shards,
        "rendered": rendered,
        "hit_cap": rendered >= my_cap,
        "seen": dict(seen),
        "kept": dict(kept_count),
        "shortfall": {b: my_band[b] - kept_count[b] for b in my_band
                      if kept_count[b] < my_band[b]},
        # 목표 구간의 원자료만 남긴다. 드라이버가 샤드를 합쳐 한 번에 백분위수를
        # 내야 맞다 -- 샤드별 백분위수를 평균내면 틀린다 (설계 §5).
        "target_sats": sat_of_target,
        "target_presets": dict(Counter(r.preset for r, b, _, _ in labelled
                                       if b == "target")),
        "labelled": labelled,
    }
    return kept, stats
```

`main()` 에 인자를 추가하고 넘긴다.

```python
    ap.add_argument("--shard", default="0/1",
                    help="i/N. 샤드 i 만 돈다. --n 과 --per-band 는 전체 값이다")
...
    shard, shards = (int(v) for v in args.shard.split("/"))
    kept, stats = fill_bucket(args.bucket, {"target": t, "hard": h,
                                            "first_ok": f},
                              args.n, args.seed, args.tau, shard, shards)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest -q`
Expected: 전체 통과

- [ ] **Step 5: 커밋**

```bash
git add scripts/label_recipes.py tests/test_labeling.py
git commit -m "[합성] --shard i/N — 샤드마다 독립 난수와 겹치지 않는 인덱스"
```

---

## Task 5: `scripts/generate_v1.py` — 병렬 드라이버와 병합

**Files:**
- Create: `scripts/generate_v1.py`
- Modify: `.gitignore`
- Test: `tests/test_generate.py` (신규)

**Interfaces:**
- Consumes: Task 4 의 `fill_bucket`, `split_counts`; Task 3 의 `bake`
- Produces:
  - `SHARDS = {"L": 2, "M": 6, "H": 7}`
  - `merge_stats(parts: list[dict]) -> dict` — `target_sats` 를 모으고 `hit_cap` 을 샤드별로 남긴다
  - `tau_from_sats(sats: list[float]) -> float | None`
  - `preset_mix(counts: dict) -> dict`
  - `_stats_path(out: str) -> str` — 항상 `data/stats/<이름>.stats.json`
  - CLI: `uv run python -m scripts.generate_v1 --bucket L --n 214000 --per-band 5000/3500/1500 --out data/recipes/train.L.jsonl`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_generate.py` 를 만든다.

```python
import pytest

from scripts.generate_v1 import SHARDS, merge_stats, preset_mix, tau_from_sats


def _part(shard, rendered, seen, kept, shortfall, hit):
    return {"bucket": "L", "shard": shard, "shards": 3, "rendered": rendered,
            "hit_cap": hit, "seen": seen, "kept": kept, "shortfall": shortfall,
            "target_sats": [], "target_presets": {}}


def test_merge_sums_counters():
    merged = merge_stats([
        _part(0, 10, {"target": 2, "hard": 1}, {"target": 2}, {}, False),
        _part(1, 20, {"target": 3, "burned": 4}, {"target": 3}, {"hard": 1}, True),
    ])
    assert merged["rendered"] == 30
    assert merged["seen"] == {"target": 5, "hard": 1, "burned": 4}
    assert merged["kept"] == {"target": 5}
    assert merged["shortfall"] == {"hard": 1}


def test_merge_keeps_hit_cap_per_shard():
    """부족분이 분포 탓인지 쪼갠 탓인지 갈리려면 샤드별로 남아야 한다 (설계 §5)."""
    merged = merge_stats([
        _part(0, 10, {}, {}, {}, False),
        _part(1, 20, {}, {}, {}, True),
    ])
    assert merged["hit_cap"] == [False, True]


def test_merge_pools_target_sats_instead_of_averaging_percentiles():
    """샤드별 백분위수를 평균내면 틀린다. 원자료를 모아 한 번에 낸다 (설계 §5)."""
    a = _part(0, 1, {}, {}, {}, False) | {"target_sats": [0.10, 0.20]}
    b = _part(1, 1, {}, {}, {}, False) | {"target_sats": [0.90]}
    merged = merge_stats([a, b])
    assert sorted(merged["target_sats"]) == [0.10, 0.20, 0.90]
    assert "tau_suggestion" not in merged


def test_merge_sums_target_presets():
    a = _part(0, 1, {}, {}, {}, False) | {"target_presets": {"crease": 2}}
    b = _part(1, 1, {}, {}, {}, False) | {"target_presets": {"crease": 1, "sine": 1}}
    assert merge_stats([a, b])["target_presets"] == {"crease": 3, "sine": 1}


def test_tau_is_95th_percentile():
    assert tau_from_sats([0.10, 0.90]) == pytest.approx(0.86, abs=0.01)
    assert tau_from_sats([]) is None


def test_preset_mix_normalizes():
    assert preset_mix({"crease": 3, "sine": 1}) == {"crease": 0.75, "sine": 0.25}
    assert preset_mix({}) == {}


def test_shard_allocation_is_fifteen():
    assert sum(SHARDS.values()) == 15
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_generate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.generate_v1'`

- [ ] **Step 3: 최소 구현**

```python
"""샤드 병렬 생성 드라이버. 오프라인 도구다.

버킷 하나를 샤드 N 개로 쪼개 동시에 돌리고, 결과를 하나의 JSONL 로 합친다.
샤드 수는 고정 상수다 -- 바꾸면 같은 시드라도 다른 데이터가 나오므로
레시피 헤더에 기록한다 (설계 §5).
"""

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from scripts.label_recipes import bake, fill_bucket
from wemeet.data.synthesis import ALL_BUCKETS, recipe_to_dict

# 설계 §4 의 프로토타입 시간(L 1.09h / M 3.35h / H 3.89h)에 비례해 나눴다.
# 샤드당 0.54~0.56h 로 고르다. 16번째 코어는 OS 몫이다.
SHARDS = {"L": 2, "M": 6, "H": 7}
EVAL_SHARDS = 5


def merge_stats(parts: list[dict]) -> dict:
    """샤드별 stats 를 합친다. tau 는 여기서 내지 않는다 -- 원자료를 모아 준다."""
    seen: dict[str, int] = {}
    kept: dict[str, int] = {}
    short: dict[str, int] = {}
    presets: dict[str, int] = {}
    sats: list[float] = []
    for p in parts:
        for k, v in p["seen"].items():
            seen[k] = seen.get(k, 0) + v
        for k, v in p["kept"].items():
            kept[k] = kept.get(k, 0) + v
        for k, v in p["shortfall"].items():
            short[k] = short.get(k, 0) + v
        for k, v in p["target_presets"].items():
            presets[k] = presets.get(k, 0) + v
        sats.extend(p["target_sats"])
    return {
        "bucket": parts[0]["bucket"],
        "shards": len(parts),
        "rendered": sum(p["rendered"] for p in parts),
        # 샤드별로 남긴다 -- 부족분이 분포 탓인지 쪼갠 탓인지 갈려야 한다 (설계 §5).
        "hit_cap": [p["hit_cap"] for p in parts],
        "seen": seen,
        "kept": kept,
        "shortfall": short,
        "target_presets": presets,
        "target_sats": sats,
    }


def tau_from_sats(sats: list[float]) -> float | None:
    """목표 구간 포화율의 95 백분위수. 목표는 복구가 증명된 샘플이다 (설계 §7)."""
    return float(np.percentile(sats, 95)) if sats else None


def preset_mix(counts: dict) -> dict:
    """실현된 preset 비율. 30/30/30/10 에서 벗어난다 -- 보고만 한다 (설계 §7)."""
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def _worker(args):
    """샤드 하나를 돌리고 레시피를 '자기 파일에' 쓴다.

    레시피를 부모로 돌려보내지 않는 이유는 메모리다 -- 상한까지 가면 58.5만 행이고
    행당 약 1KB 라 부모가 600MB 를 들고 있게 된다. 덤으로, 중간에 죽어도 끝난
    샤드의 결과가 파일로 남는다.
    """
    bucket, per_band, cap, seed, tau, shard, shards, out = args
    kept, stats = fill_bucket(bucket, per_band, cap, seed, tau, shard, shards)
    path = f"{out}.shard{shard}"
    with open(path, "w", encoding="utf-8") as fh:
        for recipe, band, sat, m_min in stats.pop("labelled"):
            row = recipe_to_dict(recipe)
            row.update(band=band, sat_ratio=sat, m_min=m_min)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return kept, stats, path


def _stats_path(out: str) -> str:
    """통계는 항상 data/stats/ 에 둔다 -- git 에 들어가는 유일한 생성물이다."""
    name = os.path.basename(out)
    if name.endswith(".jsonl"):
        name = name[:-len(".jsonl")]
    return os.path.join("data", "stats", name + ".stats.json")


def generate(bucket, per_band, cap, seed, tau, shards, out, bake_dir=None):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    jobs = [(bucket, per_band, cap, seed, tau, i, shards, out)
            for i in range(shards)]
    with ProcessPoolExecutor(max_workers=shards) as ex:
        results = list(ex.map(_worker, jobs))

    kept = [row for k, _, _ in results for row in k]
    stats = merge_stats([s for _, s, _ in results])
    stats["tau_suggestion"] = tau_from_sats(stats.pop("target_sats"))
    stats["preset_mix_realized"] = preset_mix(stats["target_presets"])
    stats["code_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()

    # 샤드 파일을 그대로 이어붙인다. JSON 을 다시 파싱하지 않는다.
    with open(out, "w", encoding="utf-8") as dst:
        dst.write(json.dumps({"_stats": stats}, ensure_ascii=False) + "\n")
        for _, _, path in results:
            with open(path, encoding="utf-8") as src:
                shutil.copyfileobj(src, dst)
            os.remove(path)

    if bake_dir:
        stats["baked"] = bake(kept, bake_dir)

    stats_path = _stats_path(out)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, choices=sorted(ALL_BUCKETS))
    ap.add_argument("--n", type=int, required=True, help="전체 렌더 상한")
    ap.add_argument("--per-band", default="5000/3500/1500", help="target/hard/first_ok")
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bake", default=None)
    args = ap.parse_args()

    t, h, f = (int(v) for v in args.per_band.split("/"))
    shards = args.shards or SHARDS.get(args.bucket, EVAL_SHARDS)
    generate(args.bucket, {"target": t, "hard": h, "first_ok": f},
             args.n, args.seed, args.tau, shards, args.out, args.bake)


if __name__ == "__main__":
    main()
```

`.gitignore` 에 세 줄을 추가한다 (`data/stats/` 는 **넣지 않는다**).

```
data/pilot2/
data/recipes/
data/eval/
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_generate.py -q && uv run ruff check scripts/generate_v1.py`
Expected: 6 passed

스모크로 한 번 돌려 본다 (작게).

Run: `uv run python -m scripts.generate_v1 --bucket low --n 200 --per-band 2/0/0 --shards 2 --out data/pilot2/smoke.jsonl`
Expected: `_stats` 가 찍히고 `data/stats/smoke.stats.json` 이 생긴다.
`data/pilot2/smoke.jsonl.shard0`·`.shard1` 은 이어붙인 뒤 지워졌어야 한다

- [ ] **Step 5: 커밋**

```bash
git add scripts/generate_v1.py tests/test_generate.py .gitignore
git commit -m "[합성] 샤드 병렬 드라이버 — _stats 병합과 tau 재계산"
```

---

## Task 6: 재파일럿 — 렌더 상한과 `τ` 를 확정한다

**Files:**
- Test: 없음 — 이 태스크는 **실행과 측정**이 산출물이다
- Modify: `docs/superpowers/specs/2026-09-14-synthesis-v1-generation-design.md` §4

**Interfaces:**
- Consumes: Task 5 의 CLI
- Produces: 버킷별 목표 출현율·속도, `τ`, 렌더 상한 — Task 7·8 이 이 숫자를 쓴다

- [ ] **Step 1: 버킷 셋을 파일럿 규모로 돌린다**

`--per-band` 를 크게 잡아 렌더 상한까지 다 돌게 한다.

```bash
for B in L M H; do
  uv run python -m scripts.generate_v1 --bucket $B --n 3000 \
      --per-band 9999/9999/9999 --out data/pilot2/$B.jsonl
done
```

- [ ] **Step 2: 네 숫자를 받아 적는다**

각 실행의 `_stats` 에서 뽑는다.

| 항목 | 어디서 |
|---|---|
| 목표 출현율 | `seen["target"] / rendered` |
| 1차 성공률 | `seen["first_ok"] / rendered` — 설계 §1.2 의 U자형 가설 확인 |
| `τ` | 세 파일럿의 `tau_suggestion` 이 아니라, **세 JSONL 의 목표 구간 `sat_ratio` 를 합쳐** 95 백분위수를 낸다 |
| 속도 | 벽시계 ÷ `rendered` (샤드 수로 나누지 말 것 — 1코어 환산이 필요하다) |

`τ` 합산은 이 한 줄로 낸다.

```bash
uv run python -c "
import json, numpy as np
s=[]
for b in ('L','M','H'):
    for line in open(f'data/pilot2/{b}.jsonl', encoding='utf-8'):
        r=json.loads(line)
        if r.get('band')=='target': s.append(r['sat_ratio'])
print('n =', len(s), ' tau =', round(float(np.percentile(s,95)),4))
"
```

- [ ] **Step 3: 렌더 상한과 샤드 배분을 다시 잡는다**

```
필요 렌더 = 5000 / 목표_출현율
렌더 상한 = 필요 렌더 × 1.5
샤드      = round(버킷_예상시간 / 전체_예상시간 × 15),  합이 15 가 되게 조정
```

**설계 §4 의 대비책을 확인한다**: 샤드당 예상 시간이 1시간을 넘으면 `d_t` 범위를
낮추거나 목표 쿼터를 줄인다. **1차 성공으로 메우지 않는다.**

- [ ] **Step 4: 잰 값을 설계 문서에 반영한다**

`docs/superpowers/specs/2026-09-14-synthesis-v1-generation-design.md` §4 의
「프로토타입 선측정」 표 아래에 **「재파일럿 실측」** 표를 추가한다. 프로토타입 값을
지우지 말고 **나란히 둔다** — 표본 1,000 vs 3,000 의 차이가 얼마나 되는지가 다음 번
판단 근거가 된다.

`scripts/generate_v1.py` 와 `scripts/label_recipes.py` 의 `--tau` 기본값을 잰 값으로 바꾸고,
`SHARDS` 를 다시 잡은 배분으로 바꾼다.

- [ ] **Step 5: 커밋**

```bash
git add docs/superpowers/specs/2026-09-14-synthesis-v1-generation-design.md scripts/
git commit -m "[합성] 재파일럿 측정 — 렌더 상한과 tau 확정"
```

---

## Task 7: 평가 세트 1,500장 — 굽고 왕복으로 검증한다

**Files:**
- Create: `data/eval/{low,mid,high}/` (gitignore)
- Test: `tests/test_eval_set.py` (신규 — 구운 세트를 검사한다)

**Interfaces:**
- Consumes: Task 6 의 출현율·`τ`
- Produces: `data/eval/<set>/{00000.png, 00000.npz, manifest.jsonl}` × 3

- [ ] **Step 1: 세 벌을 굽는다**

렌더 상한은 **포함하는 학습 버킷의 측정 출현율**에서 낸다 (`low`←`L`, `mid`←`M`,
`high`←`H`). 아래 `<CAP_*>` 는 Task 6 의 값으로 바꾼다.

```bash
uv run python -m scripts.generate_v1 --bucket low  --n <CAP_LOW>  --per-band 500/0/0 \
    --seed 7 --out data/recipes/eval.low.jsonl  --bake data/eval/low
uv run python -m scripts.generate_v1 --bucket mid  --n <CAP_MID>  --per-band 500/0/0 \
    --seed 7 --out data/recipes/eval.mid.jsonl  --bake data/eval/mid
uv run python -m scripts.generate_v1 --bucket high --n <CAP_HIGH> --per-band 500/0/0 \
    --seed 7 --out data/recipes/eval.high.jsonl --bake data/eval/high
```

**한 벌이 500장을 못 채우면 상한을 늘리지 말고, 세 벌 모두 가장 적은 장수에 맞춘다**
(설계 §3). 버킷 간 비교가 성립하려면 장수가 같아야 한다.

- [ ] **Step 2: 왕복 검증 테스트를 쓴다**

```python
"""구운 평가 세트를 검사한다. 설계 검증 #2 #3 #5 #7.

data/eval 이 없으면 건너뛴다 -- CI 에서는 생성물이 없다.
"""
import json
import os

import cv2
import numpy as np
import pytest

from scripts.label_recipes import decode, rectify

SETS = ("low", "mid", "high")
pytestmark = pytest.mark.skipif(not os.path.isdir("data/eval/low"),
                                reason="평가 세트가 아직 안 구워졌다")


def _rows(name):
    with open(f"data/eval/{name}/manifest.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("name", SETS)
def test_every_row_has_ground_truth(name):
    for row in _rows(name):
        assert row["text"] and row["band"] == "target"


def test_three_sets_have_the_same_count():
    counts = {n: len(_rows(n)) for n in SETS}
    assert len(set(counts.values())) == 1, counts


@pytest.mark.parametrize("name", SETS)
def test_flat_aspect_is_in_range(name):
    for row in _rows(name):
        assert 2.0 - 0.02 <= row["aspect"] <= 2.3 + 0.02


@pytest.mark.parametrize("name", SETS)
def test_round_trip_decodes(name):
    """구운 PNG 를 옆 NPZ 제어점으로 펴면 전부 읽힌다.

    이게 진짜 검증이다 -- 굽기 경로와 NPZ 좌표계가 맞는지를 왕복으로 확인한다.
    설계 §8.5 가 '가장 비싼 버그' 로 지목한 방향 뒤집힘이 여기서 잡힌다.
    """
    rows = _rows(name)[:50]
    ok = 0
    for row in rows:
        obs = cv2.imread(f"data/eval/{name}/{row['file']}", cv2.IMREAD_GRAYSCALE)
        z = np.load(f"data/eval/{name}/{row['npz']}")
        fixed = rectify(obs, z["dst_norm"], z["src_norm"],
                        (row["h_flat"], row["w_flat"]))
        if decode(fixed) == row["text"]:
            ok += 1
    assert ok == len(rows), f"{name}: {ok}/{len(rows)}"
```

- [ ] **Step 3: 실행한다**

Run: `uv run pytest tests/test_eval_set.py -q`
Expected: 전부 PASS. **왕복이 100% 가 아니면 진행하지 말 것** — 좌표계가 틀린 것이다.

- [ ] **Step 4: 용량과 실현 preset 비율을 기록한다**

```bash
du -sh data/eval/low data/eval/mid data/eval/high
uv run python -c "
import json
for n in ('low','mid','high'):
    s=json.load(open(f'data/stats/eval.{n}.stats.json', encoding='utf-8'))
    print(n, s['kept'], s.get('preset_mix_realized'))
"
```

- [ ] **Step 5: 커밋**

```bash
git add tests/test_eval_set.py data/stats/
git commit -m "[데이터] 평가 세트 1,500장 — 왕복 디코딩 100% 확인"
```

---

## Task 8: 학습 세트 30,000장

**Files:**
- Create: `data/recipes/train.{L,M,H}.jsonl` (gitignore), `data/stats/train.*.stats.json` (커밋)

**Interfaces:**
- Consumes: Task 6 의 렌더 상한·`τ`·샤드 배분

- [ ] **Step 1: 재현성을 먼저 확인한다 (설계 검증 #6)**

본 생성 전에 작게 두 번 돌려 같은지 본다. **여기서 다르면 30,000장이 재현 불가능한
데이터가 된다.**

```bash
uv run python -m scripts.generate_v1 --bucket L --n 300 --per-band 9999/9999/9999 \
    --shards 2 --out data/pilot2/repro_a.jsonl
uv run python -m scripts.generate_v1 --bucket L --n 300 --per-band 9999/9999/9999 \
    --shards 2 --out data/pilot2/repro_b.jsonl
uv run python -c "
import hashlib
def h(p):
    lines=[l for l in open(p,encoding='utf-8') if not l.startswith('{\"_stats\"')]
    return hashlib.sha256(''.join(lines).encode()).hexdigest()
a,b=h('data/pilot2/repro_a.jsonl'),h('data/pilot2/repro_b.jsonl')
print(a); print(b); print('동일:', a==b)
"
```

Expected: `동일: True`

- [ ] **Step 2: 세 버킷을 돌린다**

`<CAP_*>` 는 Task 6 의 값이다.

```bash
uv run python -m scripts.generate_v1 --bucket L --n <CAP_L> --out data/recipes/train.L.jsonl
uv run python -m scripts.generate_v1 --bucket M --n <CAP_M> --out data/recipes/train.M.jsonl
uv run python -m scripts.generate_v1 --bucket H --n <CAP_H> --out data/recipes/train.H.jsonl
```

`--per-band` 는 기본값 `5000/3500/1500` 이다.

- [ ] **Step 3: 쿼터와 부족분을 확인한다 (설계 검증 #8 #9)**

```bash
uv run python -c "
import json
for b in ('L','M','H'):
    s=json.load(open(f'data/stats/train.{b}.stats.json', encoding='utf-8'))
    print(b, 'kept', s['kept'], 'shortfall', s['shortfall'],
          'hit_cap', s['hit_cap'], 'preset', s.get('preset_mix_realized'))
"
```

**부족분이 있으면 메우지 않는다.** 설계 §4 대로 그대로 두고 이 계획의 아래
「생성 결과」 절에 적는다. 샤드별 `hit_cap` 을 같이 봐서, 부족분이 분포 탓인지
샤드로 쪼갠 탓인지 구분해 적는다 (전부 `True` 면 상한이 모자란 것이고, 일부만
`True` 면 쪼갠 탓일 수 있다).

- [ ] **Step 4: 용량을 잰다**

```bash
ls -l data/recipes/
gzip -k data/recipes/train.L.jsonl data/recipes/train.M.jsonl data/recipes/train.H.jsonl
ls -l data/recipes/*.gz
```

- [ ] **Step 5: 커밋**

```bash
git add data/stats/ docs/superpowers/plans/2026-09-14-synthesis-v1-generation.md
git commit -m "[데이터] 학습 세트 30,000장 — 통계와 부족분 기록"
```

---

## Task 9: HF 발행 — `123metro/barcode-datasets`

**Files:**
- Create: `data/hf/README.md` (커밋)

**Interfaces:**
- Consumes: Task 7·8 의 생성물
- Produces: `123metro/barcode-datasets` 의 `synthetic/v1/` 과 태그 `synth-v1`

- [ ] **Step 1: 로그인과 권한을 확인한다**

```bash
uv run hf auth whoami
```

Expected: `user=<계정> orgs=123metro`
아니면 `uv run hf auth login` 을 **사람이 직접** 실행한다 (대화형이다).

- [ ] **Step 2: 데이터셋 카드를 쓴다**

`data/hf/README.md` 를 만든다. `<...>` 는 Task 6~8 의 실측값으로 채운다.

```markdown
---
license: mit
task_categories:
  - image-to-image
tags:
  - barcode
  - code128
  - synthetic
---

# 난인식 바코드 합성 데이터셋 (v1)

인천세관 송장 바코드 복원 프로젝트(WE-Meet 2026, METRO팀)의 합성 데이터다.
깨끗한 Code128 을 렌더링해 원통 곡률·주름·정반사를 입히고, **정답 제어점**을 같이 낸다.

## 구성

| 분할 | 규모 | 형식 | 용도 |
|---|---|---|---|
| `synthetic/v1/recipes/train.{L,M,H}.jsonl.gz` | 각 10,000 채택 | 레시피 JSONL | 학습 (워커가 즉석 생성) |
| `synthetic/v1/eval/{low,mid,high}/` | 각 <N> 장 | PNG + NPZ + manifest | 평가 (이미지로 구움) |
| `synthetic/v1/stats/` | — | JSON | 출현율·부족분·실현 preset 비율 |

### 해상도 버킷 (`d_m0` = 모듈 폭 px)

| 학습 | 범위 | 평가 | 범위 |
|---|---|---|---|
| `L` | 1.6 ~ 2.5 | `low` | 1.6 ~ 2.2 |
| `M` | 2.5 ~ 3.7 | `mid` | 2.5 ~ 3.5 |
| `H` | 3.7 ~ 6.0 | `high` | 4.0 ~ 6.0 |

### 구간 비중 (학습)

| 구간 | 1차 디코딩 | 정답 제어점으로 편 뒤 | 비중 |
|---|---|---|---|
| 목표 | 실패 | **성공** | 50% |
| 불가 · 기하 관측 가능 | 실패 | 실패 | 35% |
| 1차 성공 | 성공 | — | 15% |
| 불가 · 포화 소실 | 실패 | 실패 | 0% (기록만) |

평가 세트는 **목표 구간 100%** 다 — 1차 성공 구간에서는 어떤 모델도 결과가 같고,
원리적 불가 구간에서는 어떤 모델도 실패하므로 모델 차이가 안 보인다.

## 라벨 형식

제어점은 **16×3 = 48점**, `(x, y)` 정규화 좌표다.

| | |
|---|---|
| 단위 | 0.0 ~ 1.0. `0.0` = 첫 픽셀 중심, `1.0` = 마지막 픽셀 중심 |
| 픽셀 환산 | `w−1`, `h−1` 을 곱한다 |
| `dst_norm` | 펴진 격자 (목표 위치) |
| `src_norm` | 관측 이미지에서의 위치. **−0.5 ~ 1.5** 를 넘을 수 있다 (회전 잔차) |
| 배열 순서 | 행 우선 |

## 물리 규격

현장 라벨은 50×30 mm 또는 60×40 mm 다. 모듈 폭 0.325 mm (50 mm / 154 모듈).
**바 영역 종횡비는 `U(2.0, 2.3)` 에서 뽑는다** — 라벨 높이의 73~75% 가 바 영역이라는
가정이며, 미검증이다.

## 재현

레시피는 `(seed, bucket, shard, index)` 로 결정론적으로 재생성된다.
코드 커밋 해시는 각 JSONL 첫 줄의 `_stats.code_commit` 에 있다.

```bash
git clone https://github.com/kim1034/2026-Wemeet-Barcode-Restoration
uv sync
uv run python -m scripts.generate_v1 --bucket L --n <CAP_L> --out train.L.jsonl
```

## 사용 시 주의

- **`eval/` 은 학습에 쓰지 않는다.**
- 구간 라벨은 디코더(zxing-cpp)에 종속된다. 디코더가 바뀌면 **재라벨링만** 하면 되고
  레시피 재생성은 필요 없다 — `sat_ratio` 를 같이 저장해 두었다.
- `burned`(포화 소실) 행은 **학습에서 제외한다.** 기록용으로만 남겼다.

## 라이선스

MIT. 모든 이미지는 합성이며 실촬영·개인정보를 포함하지 않는다.
```

- [ ] **Step 3: 저장소를 만들고 올린다**

`hf repo create` 의 확인 프롬프트 플래그는 버전마다 다르다. API 를 쓰면 비대화형이고
이미 있어도 안전하다.

```bash
uv run python -c "
from huggingface_hub import HfApi
HfApi().create_repo('123metro/barcode-datasets', repo_type='dataset', exist_ok=True)
print('repo ready')
"

gzip -kf data/recipes/train.L.jsonl data/recipes/train.M.jsonl data/recipes/train.H.jsonl

uv run hf upload 123metro/barcode-datasets data/hf/README.md README.md --repo-type=dataset
uv run hf upload 123metro/barcode-datasets data/recipes synthetic/v1/recipes \
    --repo-type=dataset --include="*.jsonl.gz"
uv run hf upload 123metro/barcode-datasets data/stats synthetic/v1/stats --repo-type=dataset
for S in low mid high; do
  uv run hf upload 123metro/barcode-datasets data/eval/$S synthetic/v1/eval/$S --repo-type=dataset
done
```

- [ ] **Step 4: 왕복으로 검증한다 (설계 검증 #10)**

```bash
uv run python -c "
import hashlib, pathlib
from huggingface_hub import hf_hub_download
p = hf_hub_download('123metro/barcode-datasets',
                    'synthetic/v1/recipes/train.L.jsonl.gz', repo_type='dataset')
def h(x): return hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest()
a, b = h(p), h('data/recipes/train.L.jsonl.gz')
print(a); print(b); print('동일:', a == b)
"
```

Expected: `동일: True`

태그를 붙인다.

```bash
uv run python -c "
from huggingface_hub import HfApi
HfApi().create_tag('123metro/barcode-datasets', tag='synth-v1', repo_type='dataset')
print('tagged')
"
```

- [ ] **Step 5: 커밋**

```bash
git add data/hf/README.md
git commit -m "[데이터] HF 발행 — 123metro/barcode-datasets synthetic/v1, 태그 synth-v1"
```

---

## Task 10: 영향 받는 문서 여섯 개를 갱신한다

**Files:**
- Modify: `docs/superpowers/specs/2026-08-28-synthesis-design.md`
- Modify: `docs/superpowers/plans/2026-09-08-synthesis-pipeline.md`
- Modify: `docs/external/huggingface.md`
- Modify: `docs/experiments/2026-09-09-control-points/README.md`
- Modify: `docs/decisions/0003-기하-추정-모델-선정.md`
- Modify: `docs/decisions/0001-바코드-심볼로지-범위.md`

- [ ] **Step 1: 설계 본체를 갱신한다**

`specs/2026-08-28-synthesis-design.md`:

| 절 | 고칠 것 |
|---|---|
| §1 산출물 | "`data/recipes/*.jsonl` 을 커밋한다" → **HF 로 간다.** 실측 222~334 MiB |
| §6 v1 분포 | `w_c ~ U(1, 9)` → `w_c_f ~ U(0.003, 0.029)`, `aspect ~ U(2.0, 2.3)` 추가 |
| §7 「미해결」 | 용량 항목 해소 — 2026-09-14 결정으로 HF 로 갔다 |
| §7 파일럿 표 | 2026-09-08 값 옆에 재파일럿 값을 **나란히** 둔다. 지우지 않는다 |

- [ ] **Step 2: 옛 구현 계획의 고정 상수표를 고친다**

`plans/2026-09-08-synthesis-pipeline.md` 의 고정 상수표에서 `관측 높이 H_OBS 220 px` 줄을
지우고 아래로 바꾼다.

```
| 종횡비 `aspect` | **U(2.0, 2.3)** | 2026-09-14 물리 치수 반영. H_OBS=220 은 틀렸다 |
```

- [ ] **Step 3: HF 문서의 조직 이름을 고친다**

`external/huggingface.md` 의 `metro-wemeet` 을 전부 **`123metro`** 로 바꾼다
(§1 표, §2 절차, §5 업로드 예시). 실제 조직 이름은 `123metro`(`METRO_We-Meet_Project`) 다.

- [ ] **Step 4: 제어점 실험 README 에 한계를 명시한다**

`experiments/2026-09-09-control-points/README.md` 맨 위에 추가한다.

```markdown
> **이 실험은 2026-09-14 물리 치수 반영 이전에 측정됐다.** `00_harvest` 부터 `05_grid`
> 까지 전부 `H_OBS = 220`(고정 높이)으로 돌았다. `n_x = 16` 은 영향이 없지만
> (가로 나이키스트는 `w` 기준), **`n_y = 3` 과 `G` 세로 해상도 `n_v = 33` 은 재검증이
> 필요하다** — 종횡비를 고정하면 `H` 버킷의 세로 대응장 변동이 +36% 늘어난다
> (중앙값 20.30 → 27.56 px). 근거: `specs/2026-09-14-synthesis-v1-generation-design.md` §8.
>
> 재현하려면 옛 레시피에 `aspect`·`w_c_f` 가 없어 `recipe_from_dict` 가 `TypeError` 를 낸다.
> 조용한 기본값을 넣지 않았다 — 조용한 성공이 조용한 오염이 되기 때문이다.
```

- [ ] **Step 5: 결정 문서 둘을 갱신하고 커밋한다**

`decisions/0003`: 후보표의 `16×3=48점` 각주에 `n_y` 재검증 항목을 추가한다 (위 실측 표 포함).

`decisions/0001`: 「현장 확인 항목」에 물리 치수를 기록한다.

```markdown
### 2026-09-14 확인: 라벨 물리 치수

**50×30 mm 또는 60×40 mm.** 모듈 폭은 50 mm / 154 모듈 = **0.325 mm** 로,
이 문서가 가정한 "송장 표준 0.33 mm" 와 일치한다. 위 카메라 표는 그대로 유효하다.

바 영역이 라벨 높이의 몇 %인지는 **아직 미확인**이다 (73~75% 로 가정하고 합성 중).
11월 실촬영 때 같이 잰다.
```

```bash
git add docs/
git commit -m "[문서] 물리 치수 반영 — 설계 본체·HF 조직명·제어점 실험 한계·결정 문서"
```

---

## 생성 결과 (Task 8 실행 후 채운다)

| 버킷 | 렌더 | 목표 | 불가·기하 | 1차 성공 | 부족분 | 샤드 `hit_cap` |
|---|---|---|---|---|---|---|
| `L` | | | | | | |
| `M` | | | | | | |
| `H` | | | | | | |

실현된 preset 비율 (목표 구간):

| 버킷 | crease | sine | octave | cylinder |
|---|---|---|---|---|
| `L` | | | | |
| `M` | | | | |
| `H` | | | | |

---

## Self-Review

**1. 스펙 커버리지**

| 설계 절 | 어느 태스크가 |
|---|---|
| §1.0 기준 환산 | 측정 근거 — 태스크 없음 (Task 10 이 `0001` 에 기록) |
| §1.1 종횡비 | Task 1 (`module_count`), Task 2 (`aspect`, `h_flat`) |
| §1.2 주름 폭 | Task 2 (`w_c_f`) |
| §1.3 렌더러 여백 | Task 1 |
| §1.4 상대/절대 규칙 | Global Constraints |
| §1.5 파급 범위 | Task 6 이 재측정 |
| §2 텍스트 wrap | Task 1 (검증), Task 2 (구현) |
| §3 평가 버킷·manifest·렌더 상한 | Task 2 (`EVAL_BUCKETS`), Task 3 (`bake`), Task 7 |
| §4 재파일럿·대비책 | Task 6 |
| §5 샤딩 | Task 4 (`--shard`), Task 5 (드라이버·병합) |
| §6 경로·HF 레이아웃 | Task 5 (`.gitignore`), Task 9 |
| §7 검증 #1~#10 | #1 Task 2·4, #2 Task 2·7, #3 Task 1, #4 Task 2, #5 Task 1, #6 Task 8, #7 Task 7, #8·#9 Task 8, #10 Task 9 |
| §8 열린 항목 | Task 10 (문서화). **재검증 자체는 이 계획 밖** — 되돌릴 수 있어서 후속이다 |
| §9 실행 순서 | Task 1~10 의 순서 |
| §10 영향 문서 | Task 10 |

**2. 빠졌다가 채운 것 — 재현성 확인의 위치**

검증 #6(샤드 재현성)을 Task 4 의 단위 테스트에만 두면 **작은 표본에서만** 확인된다.
30,000장을 만들고 나서 재현이 안 되는 것을 알면 전부 버려야 하므로, **Task 8 Step 1 에
본 생성 직전 SHA-256 확인을 넣었다.** 두 번 확인하는 것이 맞다.

**3. 타입 정합성**

`Sample` 필드 순서 `(obs, dst_norm, src_norm, m_min, sat_ratio, scale, w_flat, h_flat,
g, u_lo, u_hi)` 가 Task 2 정의와 Task 3·4 사용처(`sample.h_flat`, `sample.w_flat`)에서
일치. `fill_bucket(bucket, per_band, render_cap, seed, tau, shard, shards)` 가 Task 4
정의와 Task 5 `_worker` 호출에서 일치. `bake(kept, out_dir) -> int` 가 Task 3 정의와
Task 5 호출에서 일치. `kept` 의 원소 모양 `(recipe, band, sat, m_min)` 이 Task 4 생산과
Task 3·5 소비에서 일치.

**4. 알려진 취약점**

| | |
|---|---|
| Task 3 의 `fill_bucket("low", ..., 400, ...)` | 목표 출현율이 3.5% 근처면 400 렌더에 목표가 없을 확률이 0.08% 다. 재파일럿에서 `low` 출현율이 1% 아래로 나오면 이 테스트의 `400` 을 올려야 한다 |
| Task 5 의 `os.popen("git rev-parse HEAD")` | 워킹트리가 더러우면 해시가 실제 코드와 다르다. 생성 전에 커밋되어 있어야 한다 |
| `ProcessPoolExecutor` on Windows | 워커가 모듈을 다시 import 하므로 `scripts/generate_v1.py` 의 최상위에 무거운 작업을 두면 안 된다. `if __name__ == "__main__"` 가드는 이미 있다 |

**5. 실행 시간 예상** (설계 §4 프로토타입 기준, 재파일럿이 갱신한다)

| 태스크 | 예상 |
|---|---|
| 1~5 (코드) | 사람 시간. 테스트는 각 1분 안 |
| 6 (재파일럿) | 9,000 렌더, 15샤드 — **5분 안** |
| 7 (평가 1,500장) | **10~15분** |
| 8 (학습 30,000장) | **기대 34분 / 최악 50분** |
| 9 (HF) | 업로드 수 분 |

---

## Execution Handoff

태스크 10개다. **1 → 2 → 3 → 4 → 5 는 순서대로 가야 한다** (2가 1의 `module_count` 를
쓰고, 4가 3의 `bake` 와 같은 파일을 고치고, 5가 3·4를 부른다). 6 은 5의 CLI 가 있어야
돌고, 7·8 은 6의 숫자가 있어야 규모가 정해진다. 9 는 7·8 의 산출물이 있어야 한다.
**10 만 언제든 병렬로 할 수 있다.**
