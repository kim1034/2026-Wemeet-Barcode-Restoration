"""뒤집기 대조군을 설계 §8.5 형태로 다시. + 원통 nx sweep + 아티팩트용 이미지."""
import json, math, os
import numpy as np
from common import (D_M0, H_OBS, NX, NY, budget, control_points, decode,
                   flat_coord, fit_obs_width, apply_warp, grad_cylinder,
                   grad_sine, limit_slope, png_b64, rectify, render_clean)

OUT = os.path.dirname(os.path.abspath(__file__))
clean = render_clean(); H0, W0 = clean.shape
S_MAX = budget(D_M0)
res = {}

def control_points_wrong(s_hat, nx=NX, ny=NY):
    """§8.5 의 틀린 방향 — 역보간을 안 하고 프로파일을 그대로 쓴다."""
    h, w = s_hat.shape
    xs = np.arange(w, dtype=np.float64)
    dst, src = [], []
    for v in np.linspace(0, 1, ny):
        row = min(h - 1, int(round(v * (h - 1))))
        prof = s_hat[row]
        for u in np.linspace(0, 1, nx):
            x_obs = np.interp(u * (w - 1), xs, prof) * (w - 1)   # 역함수를 안 씀
            dst.append([u, v]); src.append([x_obs / (w - 1), v])
    return np.array(dst), np.array(src)

print("=== §8.5 방향 함정 재현 (원통) ===")
res["flip85"] = []
for t in (20, 35, 50, 70):
    w, zx, zy = fit_obs_width(lambda w, h, tt=t: grad_cylinder(w, h, tt), W0, H_OBS)
    zx, zy, _ = limit_slope(zx, zy, S_MAX)
    m, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat)
    d_ok, s_ok = control_points(s_hat, nx=6, ny=3)
    d_no, s_no = control_points_wrong(s_hat, nx=6, ny=3)
    ok = decode(rectify(obs, d_ok, s_ok, (H_OBS, W0)))
    bad = decode(rectify(obs, d_no, s_no, (H_OBS, W0)))
    err = float(np.abs(s_ok - s_no).max())
    res["flip85"].append({"theta": t, "ok": ok, "wrong": bad, "cp_gap_norm": round(err, 4)})
    print(f"  {t}deg  정상={ok or '실패':11s} 뒤집음={bad or '실패':11s} 제어점차이={err:.4f}")

print("\n=== 원통: 가로 제어점 개수 (nx) sweep ===")
res["cyl_nx"] = []
for t in (35, 50, 70):
    w, zx, zy = fit_obs_width(lambda w, h, tt=t: grad_cylinder(w, h, tt), W0, H_OBS)
    zx, zy, _ = limit_slope(zx, zy, S_MAX)
    m, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat)
    row = {"theta": t, "m_min": round(float(m.min()), 3), "by_nx": {}}
    for nx in (2, 3, 4, 6, 8):
        d, s = control_points(s_hat, nx=nx, ny=3)
        row["by_nx"][nx] = decode(rectify(obs, d, s, (H_OBS, W0)))
    res["cyl_nx"].append(row)
    print(f"  {t}deg m_min={row['m_min']:.3f} " +
          " ".join(f"nx={k}:{'O' if v else 'X'}" for k, v in row["by_nx"].items()))

print("\n=== 아티팩트용: 정보는 살아있는데 12점으로 못 펴는 케이스 ===")
res["images"] = []
for name, label, fn, nxs in [
    ("sine035", "물결 λ=0.35W (m_min 0.64 — 정보는 충분)",
     lambda w, h: grad_sine(w, h, 1.2, 0.35 * w), (4, 6, 12)),
    ("sine075", "물결 λ=0.75W (같은 세기, 파장만 김)",
     lambda w, h: grad_sine(w, h, 1.2, 0.75 * w), (4, 6, 12)),
]:
    w, zx, zy = fit_obs_width(fn, W0, H_OBS)
    zx, zy, _ = limit_slope(zx, zy, S_MAX)
    m, s_hat, _ = flat_coord(zx)
    obs = apply_warp(clean, s_hat)
    rec = {"name": name, "label": label, "w_obs": w,
           "m_min": round(float(m.min()), 3),
           "mod_min_px": round(float(m.min()) * D_M0, 2),
           "first": decode(obs), "obs_png": png_b64(obs), "rect": {}}
    for nx in nxs:
        d, s = control_points(s_hat, nx=nx, ny=3)
        r = rectify(obs, d, s, (H_OBS, W0))
        rec["rect"][nx] = {"text": decode(r), "png": png_b64(r)}
    rec["profile"] = [round(float(v), 4) for v in np.interp(
        np.linspace(0, 1, 120), np.linspace(0, 1, w), m[H_OBS // 2])]
    res["images"].append(rec)
    print(f"  {name} m_min={rec['m_min']} 1차={'O' if rec['first'] else 'X'} " +
          " ".join(f"{k}점:{'O' if v['text'] else 'X'}" for k, v in rec["rect"].items()))

res["clean_png"] = png_b64(clean); res["clean_w"] = W0; res["clean_h"] = H0
json.dump(res, open(os.path.join(OUT, "results3.json"), "w"))
print("\nsaved results3.json")
