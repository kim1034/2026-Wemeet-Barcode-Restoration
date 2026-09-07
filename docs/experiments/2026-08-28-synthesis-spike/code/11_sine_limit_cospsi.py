"""§3 미검증 항목 — 밀집 대응장 G(펴진->관측) 방식 증강, 물결 상한, 옥타브, cos(psi)."""
import json, math, os
import numpy as np, cv2
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_cylinder,
                   grad_sine, grad_crumple, limit_slope, rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
clean = render_clean(2.0); H0, W0 = clean.shape      # 저해상도 = 목표 구간
SM = budget(2.0) if 2.0 > 2 else 1.0
print(f"원본 {W0}x{H0}, d_m0=2.0px")
clean5 = render_clean(5.5); H5, W5 = clean5.shape

# ---------------------------------------------------------------- G 만들기
NV, NU = 33, 513          # 밀집 대응장 해상도 (펴진 격자 (v,u))

def build_G(s_hat):
    """G[v,u] = 그 펴진 좌표의 내용이 있는 관측 픽셀 좌표 (x,y).  s_hat 의 행별 역보간."""
    h, w = s_hat.shape
    xs = np.arange(w, dtype=np.float64)
    us = np.linspace(0, 1, NU)
    vs = np.linspace(0, 1, NV)
    G = np.empty((NV, NU, 2))
    for i, v in enumerate(vs):
        row = min(h - 1, int(round(v * (h - 1))))
        G[i, :, 0] = np.interp(us, s_hat[row], xs)
        G[i, :, 1] = v * (h - 1)
    return G

def cp_from_G(G, nx, ny, obs_shape, u_lo=0.0, u_hi=1.0):
    """최종 G 에서 제어점을 뽑는다. 좌표 변환 코드가 필요 없다."""
    h, w = obs_shape
    us = np.linspace(u_lo, u_hi, nx); vs = np.linspace(0, 1, ny)
    ui = us * (NU - 1); vi = vs * (NV - 1)
    dst, src = [], []
    for k, v in enumerate(vs):
        for u in us:
            # G 를 이중선형 보간
            fu, fv = u * (NU - 1), v * (NV - 1)
            i0, j0 = int(np.floor(fv)), int(np.floor(fu))
            i1, j1 = min(i0 + 1, NV - 1), min(j0 + 1, NU - 1)
            a, b = fv - i0, fu - j0
            p = ((1-a)*(1-b)*G[i0,j0] + (1-a)*b*G[i0,j1]
                 + a*(1-b)*G[i1,j0] + a*b*G[i1,j1])
            dst.append([(u - u_lo) / max(u_hi - u_lo, 1e-9), v])
            src.append([p[0] / (w - 1), p[1] / (h - 1)])
    return np.array(dst), np.array(src)

def aug_rotate(img, G, deg):
    """이미지는 리샘플, G 는 저장된 좌표에 아핀을 곱한다."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    P = np.concatenate([G, np.ones(G.shape[:2] + (1,))], axis=-1)
    return out, P @ M.T

def aug_crop(img, G, l, r, t, b):
    h, w = img.shape
    x0, x1 = int(l*w), w - int(r*w); y0, y1 = int(t*h), h - int(b*h)
    out = img[y0:y1, x0:x1].copy()
    Gn = G - np.array([x0, y0])
    hh, ww = out.shape
    inside = (Gn[..., 0] >= 0) & (Gn[..., 0] <= ww-1)
    us = np.linspace(0, 1, NU)
    ok = inside.all(axis=0)                        # 모든 행에서 보이는 u
    if not ok.any(): return out, Gn, 0.0, 1.0
    return out, Gn, float(us[ok][0]), float(us[ok][-1])

# --------------------------------------------- (1) 증강이 제어점과 맞는가
print("\n=== (1) 밀집 대응장 G 방식 증강 검증 ===")
res["aug"] = []
w, zx, zy = fit_obs_width(lambda w,h: grad_crease(w,h,math.tan(math.radians(62)),4.0), W0, H_OBS)
zx, zy, _ = limit_slope(zx, zy, SM)
m, s_hat, _ = flat_coord(zx)
obs0 = apply_warp(clean, s_hat)
G0 = build_G(s_hat)

# a. 증강 없이 — 기존 control_points 와 일치하는가
d_ref, s_ref = control_points(s_hat, nx=6, ny=3)
d_g, s_g = cp_from_G(G0, 6, 3, obs0.shape)
gap = float(np.abs(s_ref - s_g).max())
base = decode(rectify(obs0, d_g, s_g, (H_OBS, W0)))
print(f"  a. 증강 없음        제어점 차이 {gap:.5f}   1차={'O' if decode(obs0) else 'X'}  보정후={base or '실패'}")
res["aug"].append({"case":"none","cp_gap":round(gap,6),"rect":bool(base)})

for label, fn in [
    ("b. 회전 +5°",        lambda i,g: aug_rotate(i,g, 5.0)),
    ("c. 회전 -8°",        lambda i,g: aug_rotate(i,g,-8.0)),
    ("d. 여백 좌6%우4%",    lambda i,g: aug_crop(i,g,.06,.04,.05,.05)[:2]),
]:
    if label.startswith("d"):
        img, G, ulo, uhi = aug_crop(obs0, G0, .06,.04,.05,.05)
    else:
        img, G = fn(obs0, G0); ulo, uhi = 0.0, 1.0
    d, s = cp_from_G(G, 6, 3, img.shape, ulo, uhi)
    r = decode(rectify(img, d, s, (H_OBS, W0)))
    # 대조군: G 를 변환하지 않고 옛 제어점을 그대로 쓴다 (반드시 나빠져야 한다)
    r_bad = decode(rectify(img, d_g, s_g, (H_OBS, W0)))
    print(f"  {label:18s} 보정후={r or '실패':12s} (G 미변환 대조군={r_bad or '실패'})")
    res["aug"].append({"case":label,"rect":bool(r),"control_no_transform":bool(r_bad)})

# e. 회전 + 여백 조합
img, G, ulo, uhi = aug_crop(*aug_rotate(obs0, G0, 6.0), .05,.05,.04,.04)
d, s = cp_from_G(G, 6, 3, img.shape, ulo, uhi)
r = decode(rectify(img, d, s, (H_OBS, W0)))
r_bad = decode(rectify(img, d_g, s_g, (H_OBS, W0)))
print(f"  e. 회전+여백 조합    보정후={r or '실패':12s} (G 미변환 대조군={r_bad or '실패'})")
res["aug"].append({"case":"rot+crop","rect":bool(r),"control_no_transform":bool(r_bad)})

# --------------------------------------------- (2) 물결 상한 A/lambda
print("\n=== (2) 물결 상한 — 유도값 A/λ ≤ 0.41 (d_m0=5.5, λ=1.0W, nx=8) ===")
res["sine_limit"] = []
for al in (0.05,0.10,0.15,0.20,0.30,0.41,0.55,0.70):
    slope = 2*math.pi*al
    w, zx, zy = fit_obs_width(lambda w,h,S=slope: grad_sine(w,h,S,1.0*w), W5, H_OBS)
    m2, s2, _ = flat_coord(zx)
    o = apply_warp(clean5, s2); d, s = control_points(s2, nx=8, ny=3)
    f = bool(decode(o)); r = bool(decode(rectify(o, d, s, (H_OBS, W5))))
    mm = float(m2.min())
    res["sine_limit"].append({"A_over_lam":al,"m_min":round(mm,3),
                              "mod_min":round(mm*5.5,2),"first":f,"rect":r})
    print(f"  A/λ={al:.2f}  m_min={mm:.3f} ({mm*5.5:.2f}px)  1차={'O' if f else 'X'}  보정후={'O' if r else 'X'}")

# --------------------------------------------- (3) cos(psi) 보정
print("\n=== (3) cos(ψ) 보정 — A 로 파라미터화했을 때 max|z_x| = 2πA·cosψ/λ 인가 ===")
res["cospsi"] = []
A, lam = 6.0, 300.0
for psi in (0, 15, 30, 45, 60):
    # A 로 직접 파라미터화: z_x = (2πA/λ)·cos(2πt/λ)·cosψ
    slope_raw = 2*math.pi*A/lam
    zx_t, _ = grad_sine(400, 200, slope_raw*math.cos(math.radians(psi)), lam, psi_deg=psi)
    meas = float(np.abs(zx_t).max()); pred = slope_raw*math.cos(math.radians(psi))
    res["cospsi"].append({"psi":psi,"pred":round(pred,5),"meas":round(meas,5)})
    print(f"  ψ={psi:2d}°  예측 {pred:.5f}  측정 {meas:.5f}  오차 {abs(meas-pred):.2e}")

# --------------------------------------------- (4) 옥타브 감쇠
print("\n=== (4) 옥타브 감쇠(persistence) — 어느 게 목표 구간에 잘 드나 (N=60, d_m0 2.0) ===")
def grad_octaves(w, h, slope, lam0, oct_n, pers, seed):
    rng = np.random.default_rng(seed)
    wts = np.array([pers**j for j in range(oct_n)]); wts /= wts.sum()
    zx = np.zeros((h,w)); zy = np.zeros((h,w))
    for j in range(oct_n):
        a,b = grad_sine(w,h,slope*wts[j], lam0/(2**j),
                        psi_deg=rng.uniform(-25,25), phase=rng.uniform(0,6.28))
        zx+=a; zy+=b
    return zx, zy
res["octave"] = []
for pers in (0.35, 0.5, 0.7, 1.0):
    rng = np.random.default_rng(3); cnt={"목표":0,"불가":0,"1차성공":0}
    for _ in range(60):
        dt = rng.uniform(1.4, 2.0)
        St = math.sqrt(max((2.0/dt)**2-1, 1e-9))
        sd = int(rng.integers(1e6))
        w, zx, zy = fit_obs_width(lambda w,h: grad_octaves(w,h,St,rng.uniform(.5,1.)*w,3,pers,sd), W0, H_OBS)
        pk = float(np.abs(zx).max())
        if pk>1e-6: zx *= St/pk
        m3, s3, _ = flat_coord(zx); o = apply_warp(clean, s3)
        d, s = control_points(s3, nx=6, ny=3)
        f = bool(decode(o)); r = bool(decode(rectify(o, d, s, (H_OBS, W0))))
        cnt["1차성공" if f else ("목표" if r else "불가")] += 1
    out = {k: round(100*v/60) for k,v in cnt.items()}
    res["octave"].append({"persistence":pers, **out})
    print(f"  persistence={pers:.2f}  목표 {out['목표']:3d}%  불가 {out['불가']:3d}%  1차성공 {out['1차성공']:3d}%")

json.dump(res, open(f"{OUT}/results11.json","w"), ensure_ascii=False)
print("\nsaved results11.json")
