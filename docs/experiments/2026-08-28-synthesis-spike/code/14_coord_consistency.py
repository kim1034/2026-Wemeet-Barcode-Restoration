"""좌표 정합성을 디코딩이 아니라 '펴진 이미지 일치'로 검증한다.
   증강 후 G 에서 뽑은 제어점으로 펴면, 증강 전에 편 것과 같아야 한다."""
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
d0, s0 = cp_from_G(G0, 6, 3, obs0.shape)
ref = rectify(obs0, d0, s0, (H_OBS, W0)).astype(np.float64)

def diff(a, b):
    """펴진 이미지 두 장의 차이. 0 에 가까우면 좌표가 맞은 것."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.abs(a-b).mean())

print(f"기준: m_min={m.min():.3f} ({m.min()*5.5:.2f}px), 보정후 판독={decode(obs0.copy()) or '실패'} -> "
      f"{decode(ref.astype(np.uint8))}")
print(f"\n{'증강':>18} | {'G 변환':>10} {'G 미변환':>10} | 판정")
print("-"*62)
res["rows"]=[]
for label, deg in [("회전 +2°",2.),("회전 +5°",5.),("회전 -8°",-8.),
                   ("회전 +12°",12.),("회전 +20°",20.),("회전 -25°",-25.)]:
    img, G = rot(obs0, G0, deg)
    d, s = cp_from_G(G, 6, 3, img.shape)
    good = rectify(img, d, s, (H_OBS, W0))
    stale = rectify(img, d0, s0, (H_OBS, W0))
    dg, ds = diff(good, ref), diff(stale, ref)
    ok = dg < ds * 0.5
    res["rows"].append({"case":label,"diff_G":round(dg,2),"diff_stale":round(ds,2),
                        "decode_G":bool(decode(good)),"decode_stale":bool(decode(stale))})
    print(f"{label:>18} | {dg:10.2f} {ds:10.2f} | {'G가 명확히 낫다' if ok else '차이 불충분'}"
          f"   판독 G={'O' if decode(good) else 'X'} 미변환={'O' if decode(stale) else 'X'}")

print("\n=== 여백: 바코드를 잘라먹는가 vs 패딩만 하는가 ===")
def pad(img, G, l, r, t, b):
    h, w = img.shape
    L,R,T,B = int(l*w),int(r*w),int(t*h),int(b*h)
    out = cv2.copyMakeBorder(img, T,B,L,R, cv2.BORDER_REPLICATE)
    return out, G + np.array([L,T])
res["margin"]=[]
for label, fn in [("패딩 +8%/+5%", lambda: pad(obs0,G0,.08,.05,.06,.06)+(0.,1.)),
                  ("패딩 +15%/+15%", lambda: pad(obs0,G0,.15,.15,.10,.10)+(0.,1.)),
                  ("잘라먹기 -2%/-2%", lambda: crop(obs0,G0,.02,.02,.03,.03)),
                  ("잘라먹기 -6%/-4%", lambda: crop(obs0,G0,.06,.04,.05,.05))]:
    img, G, ulo, uhi = fn()
    d, s = cp_from_G(G, 6, 3, img.shape, ulo, uhi)
    good = rectify(img, d, s, (H_OBS, W0))
    res["margin"].append({"case":label,"u_range":[round(ulo,3),round(uhi,3)],
                          "first":bool(decode(img)),"rect":bool(decode(good))})
    print(f"  {label:18s} u범위=[{ulo:.2f},{uhi:.2f}]  1차={'O' if decode(img) else 'X'}  "
          f"보정후={'O' if decode(good) else 'X'}")

json.dump(res, open(f"{OUT}/results14.json","w"), ensure_ascii=False)
print("\nsaved results14.json")
