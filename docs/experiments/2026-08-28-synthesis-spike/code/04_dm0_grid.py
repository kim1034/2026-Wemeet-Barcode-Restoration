"""문서 §8 실측이 왜 재현 안 되는가 — 모듈 폭 d_m0 가 진짜 변수인지 확인."""
import json, math, os
import numpy as np
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_cylinder, limit_slope,
                   rectify, render_clean)

OUT = os.path.dirname(os.path.abspath(__file__))
res = {"grid": [], "note": "각 셀: 1차 디코딩 / 6x3 제어점 보정 후"}
THETAS = (0, 20, 35, 50, 70, 84)
DMS = (2.0, 2.5, 3.0, 4.0, 5.5, 8.0)

print(f"{'d_m0':>5} {'폭':>5} | " + " ".join(f"{t:>9}" for t in THETAS))
for dm in DMS:
    clean = render_clean(dm); H0, W0 = clean.shape
    s_max = budget(dm) if dm > 2 else 0.05
    row = {"d_m0": dm, "clean_w": W0, "cells": []}
    cells = []
    for t in THETAS:
        w, zx, zy = fit_obs_width(lambda w, h, tt=t: grad_cylinder(w, h, tt), W0, H_OBS)
        zx, zy, scale = limit_slope(zx, zy, s_max)
        m, s_hat, _ = flat_coord(zx)
        obs = apply_warp(clean, s_hat)
        d, s = control_points(s_hat, nx=6, ny=3)
        first = decode(obs)
        rect = decode(rectify(obs, d, s, (H_OBS, W0)))
        cell = {"theta": t, "m_min": round(float(m.min()), 3),
                "mod_min": round(float(m.min()) * dm, 2), "scale": round(scale, 3),
                "first": bool(first), "rect": bool(rect)}
        row["cells"].append(cell)
        cells.append(f"{'O' if first else 'X'}/{'O' if rect else 'X'}({cell['mod_min']:.1f})")
    res["grid"].append(row)
    print(f"{dm:>5.1f} {W0:>5} | " + " ".join(f"{c:>9}" for c in cells))

json.dump(res, open(os.path.join(OUT, "results4.json"), "w"))
print("\n셀 표기: 1차디코딩/보정후 (가장 압축된 지점의 모듈 폭 px)")
print("saved results4.json")
