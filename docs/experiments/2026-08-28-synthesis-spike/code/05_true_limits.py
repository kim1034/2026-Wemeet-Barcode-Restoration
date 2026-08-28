"""캡을 풀고 진짜 한계를 잰다. 문서 §8 실측 조건을 역추정."""
import json, math, os
import numpy as np
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_cylinder, rectify, render_clean)

OUT = os.path.dirname(os.path.abspath(__file__))
THETAS = (0, 20, 35, 50, 60, 70, 78, 84)
DMS = (2.3, 3.0, 5.5)
res = {"note": "캡 없음. 셀 = 1차/보정후 (가장 압축된 모듈 폭 px)", "grid": []}

print(f"{'d_m0':>5} {'폭':>5} | " + " ".join(f"{t:>10}" for t in THETAS))
for dm in DMS:
    clean = render_clean(dm); H0, W0 = clean.shape
    row = {"d_m0": dm, "clean_w": W0, "cells": []}; cells = []
    for t in THETAS:
        w, zx, zy = fit_obs_width(lambda w, h, tt=t: grad_cylinder(w, h, tt), W0, H_OBS)
        m, s_hat, _ = flat_coord(zx)              # <- limit_slope 안 씀
        obs = apply_warp(clean, s_hat)
        d, s = control_points(s_hat, nx=6, ny=3)
        first = bool(decode(obs)); rect = bool(decode(rectify(obs, d, s, (H_OBS, W0))))
        mod = round(float(m.min()) * dm, 2)
        row["cells"].append({"theta": t, "mod_min": mod, "first": first, "rect": rect})
        cells.append(f"{'O' if first else 'X'}/{'O' if rect else 'X'}({mod:.1f})")
    res["grid"].append(row); print(f"{dm:>5.1f} {W0:>5} | " + " ".join(f"{c:>10}" for c in cells))

# 판독 하한을 역으로 찾는다: 보정 후에도 실패하는 최소 모듈 폭
print("\n=== 보정 후 실패가 시작되는 모듈 폭 ===")
res["threshold"] = []
for dm in (3.0, 5.5, 8.0):
    clean = render_clean(dm); H0, W0 = clean.shape
    lo = None
    for t in np.arange(40, 89, 2.0):
        w, zx, zy = fit_obs_width(lambda w, h, tt=t: grad_cylinder(w, h, float(tt)), W0, H_OBS)
        m, s_hat, _ = flat_coord(zx)
        obs = apply_warp(clean, s_hat); d, s = control_points(s_hat, nx=6, ny=3)
        if not decode(rectify(obs, d, s, (H_OBS, W0))):
            lo = {"theta": float(t), "mod_min": round(float(m.min()) * dm, 2)}
            break
    res["threshold"].append({"d_m0": dm, "fail_at": lo})
    print(f"  d_m0={dm}: {lo}")

json.dump(res, open(os.path.join(OUT, "results5.json"), "w"))
print("saved results5.json")
