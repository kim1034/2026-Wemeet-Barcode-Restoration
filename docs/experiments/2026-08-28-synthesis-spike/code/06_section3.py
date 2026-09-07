"""§3 검증 — (a) 행별 정규화 근사 크기 (b) 정반사가 정보를 죽이는가 (c) §2.7 구간 비율."""
import json, math, os
import numpy as np
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_crumple,
                   grad_cylinder, grad_sine, limit_slope, rectify, render_clean)

OUT = os.path.dirname(os.path.abspath(__file__))
res = {}

# ---------- (a) 행별 s_total 편차 = 행별 정규화가 감추는 것 -------------------
print("=== (a) 행별 펴진 길이 편차 (행별 정규화 근사의 크기) ===")
res["row_var"] = []
def ridge_curved(w, h, slope, w_c, amp, lam_y):
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    return slope * np.tanh((gx - (w-1)/2 - amp*np.sin(2*np.pi*gy/lam_y)) / w_c), np.zeros((h,w))

clean = render_clean(); H0, W0 = clean.shape; S_MAX = budget(5.5)
for name, fn in [
    ("원통 50° (ψ=0)", lambda w,h: grad_cylinder(w,h,50)),
    ("접힘 ψ=0",       lambda w,h: grad_crease(w,h,math.tan(math.radians(55)),8.0)),
    ("접힘 ψ=30°",     lambda w,h: grad_crease(w,h,math.tan(math.radians(55)),8.0,psi_deg=30)),
    ("접힘 ψ=45°",     lambda w,h: grad_crease(w,h,math.tan(math.radians(55)),8.0,psi_deg=45)),
    ("굽이치는 능선",   lambda w,h: ridge_curved(w,h,math.tan(math.radians(55)),8.0,0.10*w,1.0*h)),
    ("구김 3옥타브",    lambda w,h: grad_crumple(w,h,2.0,0.5*w)),
]:
    w, zx, zy = fit_obs_width(fn, W0, H_OBS)
    zx, zy, _ = limit_slope(zx, zy, S_MAX)
    m = 1.0/np.sqrt(1.0+zx**2); J = 1.0/m
    tot = J.sum(axis=1)
    spread = (tot.max()-tot.min())/tot.mean()
    # 행별 정규화가 만드는 최대 가로 위치 오차(px)
    err_px = spread * (w-1) / 2
    res["row_var"].append({"case": name, "spread": round(float(spread),4),
                           "err_px": round(float(err_px),2), "w_obs": w})
    print(f"  {name:16s} 행간 편차 {spread*100:5.2f}%  →  최대 위치 오차 {err_px:5.1f}px")

# ---------- (b) 정반사가 디코딩을 죽이는가 -----------------------------------
print("\n=== (b) Blinn-Phong 정반사 ===")
def shade(base, zx, zy, l, ka, kd, ks, p):
    n = np.stack([-zx, -zy, np.ones_like(zx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    l = np.array(l, float); l /= np.linalg.norm(l)
    v = np.array([0.,0.,1.]); hv = l+v; hv /= np.linalg.norm(hv)
    diff = np.clip(n @ l, 0, None)
    spec = ks * np.clip(n @ hv, 0, None) ** p
    img = base/255.0 * (ka + kd*diff) + spec
    return (np.clip(img,0,1)*255).astype(np.uint8), float((img>=0.995).mean())

res["specular"] = []
w, zx, zy = fit_obs_width(lambda w,h: grad_cylinder(w,h,45), W0, H_OBS)
zx, zy, _ = limit_slope(zx, zy, S_MAX)
m, s_hat, _ = flat_coord(zx); obs = apply_warp(clean, s_hat).astype(np.float64)
d, s = control_points(s_hat, nx=6, ny=3)
for label, ks, p, l in [("반사 없음", 0.0, 200, (0.3,0.2,1)),
                        ("약한 반사 ks=0.4", 0.4, 200, (0.3,0.2,1)),
                        ("강한 반사 ks=1.0", 1.0, 200, (0.3,0.2,1)),
                        ("넓은 반사 p=40",  1.0, 40,  (0.3,0.2,1)),
                        ("곱하기로 넣으면", None, 200, (0.3,0.2,1))]:
    if ks is None:
        n = np.stack([-zx,-zy,np.ones_like(zx)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
        lv=np.array([0.3,0.2,1.]); lv/=np.linalg.norm(lv); hv=lv+np.array([0,0,1.]); hv/=np.linalg.norm(hv)
        img = obs/255.0 * (0.35+0.5*np.clip(n@lv,0,None) + 1.0*np.clip(n@hv,0,None)**200)
        sh = (np.clip(img,0,1)*255).astype(np.uint8); sat = float((img>=0.995).mean())
    else:
        sh, sat = shade(obs, zx, zy, l, 0.35, 0.5, ks, p)
    first = decode(sh); rect = decode(rectify(sh, d, s, (H_OBS, W0)))
    res["specular"].append({"label":label,"sat":round(sat,4),
                            "first":bool(first),"rect":bool(rect)})
    print(f"  {label:18s} 포화화소 {sat*100:5.2f}%  1차={'O' if first else 'X'}  보정후={'O' if rect else 'X'}")

# ---------- (c) §2.7 분포에서 구간 비율 ---------------------------------------
print("\n=== (c) §2.7 분포 실측 — 구간 비율 (N=120) ===")
DMS = [2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0]
CACHE = {d: render_clean(d) for d in DMS}
rng = np.random.default_rng(42)

def sample(rng, nx):
    dm = float(rng.choice(DMS)); cl = CACHE[dm]; h0, w0 = cl.shape
    smax = budget(dm)
    if rng.random() < 0.10:                       # 평탄 예약
        preset, rho = "flat", 0.0
    else:
        preset = rng.choice(["crease","sine","crumple","cyl"], p=[.40,.35,.15,.10])
        rho = rng.uniform(0.3, 1.0)
    K = int(rng.choice([1,2,3], p=[.45,.35,.20]))
    def make(w, h):
        zx = np.zeros((h,w)); zy = np.zeros((h,w))
        if preset == "flat": return zx, zy
        shares = rng.dirichlet(np.ones(K+1))
        s_cyl = rho*smax*shares[0]
        a,b = grad_cylinder(w,h,math.degrees(math.atan(s_cyl))); zx+=a; zy+=b
        if preset == "cyl": return zx, zy
        for i in range(K):
            si = rho*smax*shares[i+1]; psi = rng.uniform(-45,45)
            if preset == "crease":
                a,b = grad_crease(w,h,si,rng.uniform(2,20),psi_deg=psi,
                                  offset=rng.uniform(-0.3,0.3)*w)
            elif preset == "sine":
                a,b = grad_sine(w,h,si,rng.uniform(0.35,1.2)*w,psi_deg=psi,
                                phase=rng.uniform(0,2*np.pi))
            else:
                a,b = grad_crumple(w,h,si,rng.uniform(0.4,0.9)*w,octaves=3,
                                   seed=int(rng.integers(1e6)))
            zx+=a; zy+=b
        return zx, zy
    w, zx, zy = fit_obs_width(make, w0, H_OBS)
    zx, zy, sc = limit_slope(zx, zy, smax)
    m, s_hat, _ = flat_coord(zx)
    obs = apply_warp(cl, s_hat)
    if preset != "flat":
        obs, _ = shade(obs.astype(np.float64), zx, zy, (rng.uniform(-.5,.5), rng.uniform(-.5,.5), 1),
                       0.35, 0.5, float(rng.uniform(0,1.0)), float(rng.uniform(80,400)))
    d, s = control_points(s_hat, nx=nx, ny=3)
    return bool(decode(obs)), bool(decode(rectify(obs, d, s, (H_OBS, w0)))), preset

res["zones"] = {}
for nx in (4, 6):
    rng = np.random.default_rng(42)
    cnt = {"목표":0, "불가":0, "1차성공":0}; per = {}
    for i in range(120):
        f, r, pre = sample(rng, nx)
        z = "1차성공" if f else ("목표" if r else "불가")
        cnt[z]+=1; per.setdefault(pre, {"목표":0,"불가":0,"1차성공":0}); per[pre][z]+=1
    res["zones"][nx] = {"total":cnt, "per_preset":per}
    print(f"  nx={nx}: " + "  ".join(f"{k} {100*v/120:.0f}%" for k,v in cnt.items()))
    for pre, c in sorted(per.items()):
        n = sum(c.values())
        print(f"      {pre:8s} (n={n:3d})  " + " ".join(f"{k} {100*v/n:3.0f}%" for k,v in c.items()))

json.dump(res, open(f"{OUT}/results6.json","w"))
print("\nsaved results6.json")
