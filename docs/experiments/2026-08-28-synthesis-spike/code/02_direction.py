"""확인 실험 — (1) 방향 오류가 수치로 잡히는가 (2) 12점 표현 한계 (3) 제어점 sweep."""
import json, math, os
import numpy as np
from common import (D_M0, H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_cylinder,
                   grad_sine, limit_slope, rectify, render_clean, _axis)

OUT = os.path.dirname(os.path.abspath(__file__))
clean = render_clean(); H0, W0 = clean.shape
S_MAX = budget(D_M0)
res = {}

# ---- (1) 원통에서 해석해와 비교. cumsum(1/m) 이 arcsin 을 재현하는가 -------
th = 50.0
w_obs, zx, zy = fit_obs_width(lambda w, h: grad_cylinder(w, h, th), W0, H_OBS)
_, s_right, _ = flat_coord(zx, wrong=False)
_, s_wrong, _ = flat_coord(zx, wrong=True)
x = np.arange(w_obs, dtype=np.float64); c = (w_obs - 1) / 2
r = (w_obs - 1) / (2 * math.sin(math.radians(th)))
analytic = r * np.arcsin((x - c) / r); analytic -= analytic[0]; analytic /= analytic[-1]
res["direction"] = {
    "theta": th,
    "err_right_px": round(float(np.abs(s_right[H_OBS // 2] - analytic).max() * (w_obs - 1)), 3),
    "err_wrong_px": round(float(np.abs(s_wrong[H_OBS // 2] - analytic).max() * (w_obs - 1)), 3),
    "prof_right": [round(float(v), 4) for v in np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, w_obs), s_right[H_OBS // 2])],
    "prof_wrong": [round(float(v), 4) for v in np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, w_obs), s_wrong[H_OBS // 2])],
    "prof_analytic": [round(float(v), 4) for v in np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, w_obs), analytic)],
}
print(f"[방향] 정정판 최대오차 {res['direction']['err_right_px']}px / "
      f"오류판 {res['direction']['err_wrong_px']}px")

# ---- flip 대조군을 강한 왜곡에서 --------------------------------------------
res["flip"] = []
for t in (35, 50, 70):
    w, zx, zy = fit_obs_width(lambda w, h, tt=t: grad_cylinder(w, h, tt), W0, H_OBS)
    zx, zy, _ = limit_slope(zx, zy, S_MAX)
    m, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat); dst, src = control_points(s_hat)
    ok = decode(rectify(obs, dst, src, (H_OBS, W0)))
    flip = decode(rectify(obs, src, dst, (H_OBS, W0)))
    res["flip"].append({"theta": t, "ok": ok, "flipped": flip})
    print(f"[flip] {t}deg  정상={ok or '실패'}  뒤집음={flip or '실패'}")

# ---- (2) 물결 파장 x 제어점 개수 --------------------------------------------
def sweep_sine():
    rows = []
    for lam_f in (0.15, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50):
        w, zx, zy = fit_obs_width(lambda w, h, L=lam_f: grad_sine(w, h, 1.2, L * w), W0, H_OBS)
        zx, zy, _ = limit_slope(zx, zy, S_MAX)
        m, s_hat, _ = flat_coord(zx)
        obs = apply_warp(clean, s_hat)
        row = {"lam_frac": lam_f, "m_min": round(float(m.min()), 3),
               "first": decode(obs), "by_nx": {}}
        for nx in (4, 6, 8, 12):
            dst, src = control_points(s_hat, nx=nx, ny=3)
            row["by_nx"][nx] = decode(rectify(obs, dst, src, (H_OBS, W0)))
        rows.append(row)
        print(f"[물결] lam={lam_f:.2f}W m_min={row['m_min']:.3f} 1차={'O' if row['first'] else 'X'} "
              + " ".join(f"{k}점={'O' if v else 'X'}" for k, v in row["by_nx"].items()))
    return rows
res["sine_sweep"] = sweep_sine()

# ---- (3) 세로로 굽이치는 능선 x 세로 제어점 개수 (§7.1 본체) ----------------
def grad_ridge_curved(w, h, slope, w_c, amp_y, lam_y):
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = (w - 1) / 2
    shift = amp_y * np.sin(2 * np.pi * gy / lam_y)
    return slope * np.tanh((gx - cx - shift) / w_c), np.zeros((h, w))

def sweep_ridge():
    rows = []
    for lamy_f in (0.5, 1.0, 2.0, 4.0):
        w, zx, zy = fit_obs_width(
            lambda w, h, L=lamy_f: grad_ridge_curved(w, h, math.tan(math.radians(55)), 8.0, 0.10 * w, L * h),
            W0, H_OBS)
        zx, zy, _ = limit_slope(zx, zy, S_MAX)
        m, s_hat, _ = flat_coord(zx)
        obs = apply_warp(clean, s_hat)
        row = {"lam_y_frac": lamy_f, "m_min": round(float(m.min()), 3),
               "first": decode(obs), "by_ny": {}}
        for ny in (2, 3, 5, 7):
            dst, src = control_points(s_hat, nx=6, ny=ny)
            row["by_ny"][ny] = decode(rectify(obs, dst, src, (H_OBS, W0)))
        rows.append(row)
        print(f"[능선] lam_y={lamy_f:.1f}H m_min={row['m_min']:.3f} 1차={'O' if row['first'] else 'X'} "
              + " ".join(f"ny={k}:{'O' if v else 'X'}" for k, v in row["by_ny"].items()))
    return rows
res["ridge_sweep"] = sweep_ridge()

json.dump(res, open(os.path.join(OUT, "results2.json"), "w"))
print("\nsaved results2.json")
