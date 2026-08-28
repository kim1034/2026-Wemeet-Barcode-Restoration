"""§3 증강 검증 — 밀집 대응장 G(펴진->관측) 방식.
   기준 샘플은 08 에서 목표 구간으로 확인된 것: d_m0=5.5, 접힘 α=55°, w_c=4."""
import json, math, os
import numpy as np, cv2
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, limit_slope,
                   rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
clean = render_clean(5.5); H0, W0 = clean.shape; SM = budget(5.5)
NV, NU = 33, 513

def build_G(s_hat):
    h, w = s_hat.shape; xs = np.arange(w, dtype=np.float64)
    us = np.linspace(0,1,NU); G = np.empty((NV,NU,2))
    for i, v in enumerate(np.linspace(0,1,NV)):
        row = min(h-1, int(round(v*(h-1))))
        G[i,:,0] = np.interp(us, s_hat[row], xs); G[i,:,1] = v*(h-1)
    return G

def sample_G(G, u, v):
    fu, fv = u*(NU-1), v*(NV-1)
    i0, j0 = int(np.floor(fv)), int(np.floor(fu))
    i1, j1 = min(i0+1,NV-1), min(j0+1,NU-1); a, b = fv-i0, fu-j0
    return ((1-a)*(1-b)*G[i0,j0] + (1-a)*b*G[i0,j1] + a*(1-b)*G[i1,j0] + a*b*G[i1,j1])

def cp_from_G(G, nx, ny, shape, u_lo=0.0, u_hi=1.0):
    h, w = shape; dst=[]; src=[]
    for v in np.linspace(0,1,ny):
        for u in np.linspace(u_lo,u_hi,nx):
            p = sample_G(G, u, v)
            dst.append([(u-u_lo)/max(u_hi-u_lo,1e-9), v])
            src.append([p[0]/(w-1), p[1]/(h-1)])
    return np.array(dst), np.array(src)

def rot(img, G, deg):
    h, w = img.shape
    M = cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2), deg, 1.0)
    out = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    P = np.concatenate([G, np.ones(G.shape[:2]+(1,))], -1)
    return out, P @ M.T

def crop(img, G, l, r, t, b):
    h, w = img.shape
    x0, x1 = int(l*w), w-int(r*w); y0, y1 = int(t*h), h-int(b*h)
    out = img[y0:y1, x0:x1].copy(); Gn = G - np.array([x0,y0])
    hh, ww = out.shape
    ok = ((Gn[...,0] >= 0) & (Gn[...,0] <= ww-1)).all(axis=0)
    us = np.linspace(0,1,NU)
    if not ok.any(): return out, Gn, 0.0, 1.0
    return out, Gn, float(us[ok][0]), float(us[ok][-1])

w, zx, zy = fit_obs_width(lambda w,h: grad_crease(w,h,math.tan(math.radians(55)),4.0), W0, H_OBS)
zx, zy, _ = limit_slope(zx, zy, SM)
m, s_hat, _ = flat_coord(zx); obs0 = apply_warp(clean, s_hat); G0 = build_G(s_hat)
d_ref, s_ref = control_points(s_hat, nx=6, ny=3)
d_g, s_g = cp_from_G(G0, 6, 3, obs0.shape)
gap = float(np.abs(s_ref - s_g).max())
base_first = decode(obs0); base_rect = decode(rectify(obs0, d_g, s_g, (H_OBS, W0)))
print(f"기준 샘플  m_min={m.min():.3f} ({m.min()*5.5:.2f}px)  1차={base_first or '실패'}  보정후={base_rect or '실패'}")
print(f"  G 방식 제어점 vs 기존 control_points 차이: {gap:.6f}")
res["base"] = {"cp_gap": round(gap,7), "first": bool(base_first), "rect": bool(base_rect)}

print("\n=== 증강: G 를 같이 변환했을 때 / 안 했을 때 ===")
res["aug"]=[]
cases = [("회전 +5°",  lambda: rot(obs0,G0, 5.0)+(0.0,1.0)),
         ("회전 -8°",  lambda: rot(obs0,G0,-8.0)+(0.0,1.0)),
         ("회전 +12°", lambda: rot(obs0,G0,12.0)+(0.0,1.0)),
         ("여백 좌6%우4%", lambda: crop(obs0,G0,.06,.04,.05,.05)),
         ("여백 좌12%우8%", lambda: crop(obs0,G0,.12,.08,.06,.06)),
         ("회전6° + 여백5%", lambda: crop(*rot(obs0,G0,6.0), .05,.05,.04,.04))]
for label, fn in cases:
    img, G, ulo, uhi = fn()
    d, s = cp_from_G(G, 6, 3, img.shape, ulo, uhi)
    good = decode(rectify(img, d, s, (H_OBS, W0)))
    bad  = decode(rectify(img, d_g, s_g, (H_OBS, W0)))     # G 미변환 대조군
    first = decode(img)
    res["aug"].append({"case":label,"u_range":[round(ulo,3),round(uhi,3)],
                       "first":bool(first),"rect_G":bool(good),"rect_stale":bool(bad)})
    print(f"  {label:16s} 1차={'O' if first else 'X'}  G변환={good or '실패':12s} 미변환={bad or '실패'}")

json.dump(res, open(f"{OUT}/results13.json","w"), ensure_ascii=False)
print("\nsaved results13.json")
