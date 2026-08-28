"""버릴 스크립트 — §2 수식이 실제로 어떤 그림을 만드는지 확인용.

규약 (설계 §2 재검토판):
    x = 관측 좌표, s = 펴진 좌표
    z 는 관측 좌표 위의 높이. z_x 만 있으면 된다.
    J = ds/dx = sqrt(1+z_x^2) >= 1        신장률
    m = dx/ds = 1/J <= 1                  압축률
    s(x) = cumsum(J)                      <- A-1 정정: cumsum(m) 이 아니다
"""

import base64
import io
import json
import math
import os

import barcode
import cv2
import numpy as np
import zxingcpp
from barcode.writer import ImageWriter

OUT = os.path.dirname(os.path.abspath(__file__))
D_M0 = 5.5          # 펴진 상태 모듈 폭 (px)
H_OBS = 220
TEXT = "WEMEET0001"
NX, NY = 4, 3       # 제어점 격자 (기본안 12점)


# ---------------------------------------------------------------- render.py
def render_clean(module_px: float = D_M0) -> np.ndarray:
    """깨끗한 Code128 을 그린다. 모듈 폭이 정확히 module_px 가 되게 맞춘다."""
    dpi = 300
    mm = module_px * 25.4 / dpi
    writer = ImageWriter()
    obj = barcode.get("code128", TEXT, writer=writer)
    buf = io.BytesIO()
    obj.write(buf, options={
        "module_width": mm, "module_height": 12.0, "quiet_zone": 2.0,
        "write_text": False, "dpi": dpi,
    })
    buf.seek(0)
    arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = arr.shape
    return cv2.resize(arr, (w, H_OBS), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------- surface.py
def grad_cylinder(w, h, theta_deg):
    """전역 원통 감김. 관측 폭이 w 일 때 가장자리 기울기가 tan(theta)."""
    if theta_deg <= 0:
        return np.zeros((h, w)), np.zeros((h, w))
    th = math.radians(theta_deg)
    c = (w - 1) / 2.0
    r = (w - 1) / (2.0 * math.sin(th))
    x = np.arange(w, dtype=np.float64) - c
    zx = -x / np.sqrt(np.maximum(r * r - x * x, 1e-9))
    return np.tile(zx, (h, 1)), np.zeros((h, w))


def _axis(w, h, psi_deg):
    """능선 축 t = (x-cx)cos(psi) + (y-cy)sin(psi) 와 cos(psi)."""
    psi = math.radians(psi_deg)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    t = (gx - (w - 1) / 2) * math.cos(psi) + (gy - (h - 1) / 2) * math.sin(psi)
    return t, math.cos(psi), math.sin(psi)


def grad_sine(w, h, slope, lam, psi_deg=0.0, phase=0.0):
    """물결 주름. slope = 이 성분의 최대 |z_x| (= 2*pi*A/lam*cos(psi))."""
    t, cp, sp = _axis(w, h, psi_deg)
    d = slope * np.cos(2 * np.pi * t / lam + phase)   # df/dt * (2piA/lam)
    return d, d * (sp / cp if cp else 0.0)


def grad_crease(w, h, slope, w_c, psi_deg=0.0, offset=0.0):
    """접힌 능선. 기울기가 -slope -> +slope 로 tanh 전이."""
    t, cp, sp = _axis(w, h, psi_deg)
    d = slope * np.tanh((t - offset) / max(w_c, 1e-6))
    return d, d * (sp / cp if cp else 0.0)


def grad_crumple(w, h, slope, lam0, octaves=3, psi_spread=25.0, seed=42):
    """여러 스케일 능선의 합 = 구김. 옥타브마다 기울기 기여가 같다(표준 fBm)."""
    rng = np.random.default_rng(seed)
    zx = np.zeros((h, w))
    zy = np.zeros((h, w))
    per = slope / octaves
    for j in range(octaves):
        a, b = grad_sine(w, h, per, lam0 / (2 ** j),
                         psi_deg=rng.uniform(-psi_spread, psi_spread),
                         phase=rng.uniform(0, 2 * np.pi))
        zx += a
        zy += b
    return zx, zy


def budget(d_m0: float) -> float:
    """판독 하한에서 나오는 최대 허용 기울기 S_max."""
    return math.sqrt((d_m0 / 2) ** 2 - 1)


def limit_slope(zx, zy, s_max):
    """A-2 정정: m 을 clip 하지 않고 기울기를 통째로 줄인다 (z 와 m 이 계속 일관)."""
    peak = float(np.abs(zx).max())
    scale = min(1.0, s_max / peak) if peak > 0 else 1.0
    return zx * scale, zy * scale, scale


# ------------------------------------------------------------------ warp.py
def flat_coord(zx, wrong=False):
    """관측 -> 펴진 매핑. 행마다 가로로만 적분한다 (v = y).

    s(x) = cumsum(J),  J = 1/m.  wrong=True 면 문서 §4.1 의 cumsum(m).
    """
    m = 1.0 / np.sqrt(1.0 + zx ** 2)
    integrand = m if wrong else 1.0 / m
    s = np.cumsum(integrand, axis=1) - integrand[:, :1]
    s_hat = s / s[:, -1:]                       # 행마다 0~1 로 정규화
    return m, s_hat, float(s[:, -1].mean())


def fit_obs_width(make_grad, w_flat, h, iters=4):
    """펴진 길이가 w_flat 이 되도록 관측 폭을 맞춘다 (감기면 좁아 보인다)."""
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


def apply_warp(clean, s_hat):
    """깨끗한 이미지를 관측 이미지로. remap 은 arcsin 쪽(s)을 쓴다."""
    h_obs, w_obs = s_hat.shape
    h0, w0 = clean.shape
    map_x = (s_hat * (w0 - 1)).astype(np.float32)
    gy = np.linspace(0, h0 - 1, h_obs, dtype=np.float32)
    map_y = np.tile(gy[:, None], (1, w_obs))
    return cv2.remap(clean, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def control_points(s_hat, nx=NX, ny=NY):
    """정답 제어점. dst = 펴진 격자, src = 지금 있는 위치 (역보간 = sin 쪽)."""
    h_obs, w_obs = s_hat.shape
    us = np.linspace(0, 1, nx)
    vs = np.linspace(0, 1, ny)
    xs = np.arange(w_obs, dtype=np.float64)
    dst, src = [], []
    for v in vs:
        row = min(h_obs - 1, int(round(v * (h_obs - 1))))
        prof = s_hat[row]
        for u in us:
            x_obs = np.interp(u, prof, xs)
            dst.append([u, v])
            src.append([x_obs / (w_obs - 1), v])
    return np.array(dst), np.array(src)


# ------------------------------------------- sw/rectify.py (설계 §9 그대로)
def tps_flow(src, dst, shape, reg=0.0):
    n = len(dst)
    d = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(d > 0, d ** 2 * np.log(d ** 2), 0.0)
    k += reg * np.eye(n)
    p = np.hstack([np.ones((n, 1)), dst])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    dg = np.linalg.norm(grid[:, None, :] - dst[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ug = np.where(dg > 0, dg ** 2 * np.log(dg ** 2), 0.0)
    pg = np.hstack([np.ones((len(grid), 1)), grid])
    out = []
    for axis in (0, 1):
        b = np.concatenate([src[:, axis], np.zeros(3)])
        coef = np.linalg.solve(a, b)
        out.append((ug @ coef[:n] + pg @ coef[n:]).reshape(h, w).astype(np.float32))
    return out[0], out[1]


def rectify(obs, dst_norm, src_norm, out_shape):
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


# ------------------------------------------------------------------ 실행부
def run_case(name, label, make_grad, clean, s_max, flip=False, wrong=False):
    h0, w0 = clean.shape
    w_obs, zx, zy = fit_obs_width(make_grad, w0, H_OBS)
    zx, zy, scale = limit_slope(zx, zy, s_max)
    m, s_hat, _ = flat_coord(zx, wrong=wrong)
    obs = apply_warp(clean, s_hat)
    dst, src = control_points(s_hat)
    if flip:
        dst, src = src, dst
    rect = rectify(obs, dst, src, (H_OBS, w0))
    prof = m[H_OBS // 2]
    return {
        "name": name, "label": label,
        "w_obs": w_obs, "scale": round(scale, 3),
        "m_min": round(float(m.min()), 3),
        "mod_min_px": round(float(m.min()) * D_M0, 2),
        "decode_first": decode(obs),
        "decode_rect": decode(rect),
        "profile": [round(float(v), 4) for v in
                    np.interp(np.linspace(0, 1, 120),
                              np.linspace(0, 1, len(prof)), prof)],
        "src": src.round(4).tolist(), "dst": dst.round(4).tolist(),
    }, obs, rect


def png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def main():
    clean = render_clean()
    h0, w0 = clean.shape
    s_max = budget(D_M0)
    print(f"clean {w0}x{h0}  d_m0={D_M0}px  S_max={s_max:.3f} "
          f"(단독 원통 상한 {math.degrees(math.atan(s_max)):.1f}deg)")

    cases = []
    for th in (0, 20, 35, 50, 70):
        cases.append((f"cyl{th}", f"원통 {th}°",
                      lambda w, h, t=th: grad_cylinder(w, h, t), {}))
    cases += [
        ("sine_weak", "물결 A/λ=0.05",
         lambda w, h: grad_sine(w, h, 2 * math.pi * 0.05, 0.45 * w), {}),
        ("sine_strong", "물결 A/λ=0.15",
         lambda w, h: grad_sine(w, h, 2 * math.pi * 0.15, 0.30 * w), {}),
        ("crease", "접힘 α=55°, w_c=6px",
         lambda w, h: grad_crease(w, h, math.tan(math.radians(55)), 6.0), {}),
        ("ridge30", "접힘 α=55°, 능선각 ψ=30°",
         lambda w, h: grad_crease(w, h, math.tan(math.radians(55)), 6.0, psi_deg=30), {}),
        ("crumple", "구김 (3옥타브, ψ 랜덤)",
         lambda w, h: grad_crumple(w, h, 2.0, 0.5 * w), {}),
        ("mixed", "원통 45° + 물결 + 구김",
         lambda w, h: tuple(
             a + b + c for a, b, c in zip(
                 grad_cylinder(w, h, 45),
                 grad_sine(w, h, 0.6, 0.35 * w, psi_deg=15),
                 grad_crumple(w, h, 0.8, 0.4 * w, seed=7))), {}),
        ("cyl50_flip", "원통 50° — src/dst 뒤집은 대조군",
         lambda w, h: grad_cylinder(w, h, 50), {"flip": True}),
        ("cyl50_wrong", "원통 50° — cumsum(m) 오류판",
         lambda w, h: grad_cylinder(w, h, 50), {"wrong": True}),
    ]

    out = {"clean_png": png_b64(clean), "clean_w": w0, "clean_h": h0,
           "d_m0": D_M0, "s_max": round(s_max, 3), "text": TEXT, "cases": []}
    for name, label, fn, kw in cases:
        rec, obs, rect = run_case(name, label, fn, clean, s_max, **kw)
        rec["obs_png"] = png_b64(obs)
        rec["rect_png"] = png_b64(rect)
        out["cases"].append(rec)
        print(f"{name:14s} w_obs={rec['w_obs']:4d} m_min={rec['m_min']:.3f} "
              f"({rec['mod_min_px']:.1f}px)  1차={rec['decode_first'] or '실패':10s} "
              f"보정후={rec['decode_rect'] or '실패'}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f)
    print("\nsaved results.json")


if __name__ == "__main__":
    main()
