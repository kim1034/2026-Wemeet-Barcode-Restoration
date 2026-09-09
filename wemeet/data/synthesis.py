"""레시피 샘플링과 조립. 이미지는 저장하지 않고 레시피만 남긴다.

조립 순서는 설계 §5 다:
    렌더 -> 기하 왜곡(+G) -> 음영 -> 기하 증강(+G) -> 제어점 -> 광도 증강
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from wemeet.data.augment import geometric_margin, geometric_rotate, photometric
from wemeet.data.optics import shade
from wemeet.data.render import render_clean
from wemeet.data.surface import (
    grad_crease,
    grad_crumple,
    grad_cylinder,
    grad_sine,
    limit_slope,
    slope_budget,
)
from wemeet.data.warp import apply_warp, build_G, control_points, fit_obs_width, flat_coord

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


def _make_grad(recipe: Recipe, s_t: float):
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

    _, zx, zy = fit_obs_width(_make_grad(recipe, s_t), w_flat, H_OBS)

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
