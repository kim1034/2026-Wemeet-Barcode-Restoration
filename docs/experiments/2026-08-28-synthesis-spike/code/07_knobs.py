"""(b) 정반사 더하기 vs 곱하기를 제대로. (c') 목표 구간을 넓히는 손잡이 찾기."""
import json, math, os
import numpy as np, cv2
from common import (H_OBS, control_points, decode, flat_coord, fit_obs_width,
                   apply_warp, grad_crease, grad_crumple, grad_cylinder,
                   grad_sine, limit_slope, rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}

def normals(zx, zy):
    n = np.stack([-zx, -zy, np.ones_like(zx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)

def shade(base, zx, zy, l, ka, kd, ks, p, mode="add"):
    n = normals(zx, zy); l = np.array(l,float); l/=np.linalg.norm(l)
    hv = l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    diff = np.clip(n@l,0,None); spec = ks*np.clip(n@hv,0,None)**p
    b = base/255.0
    img = b*(ka+kd*diff)+spec if mode=="add" else b*(ka+kd*diff+spec)
    return np.clip(img,0,1)

print("=== (b) 정반사: 검은 막대가 실제로 사라지는가 ===")
clean = render_clean(); H0, W0 = clean.shape
from common import budget
w, zx, zy = fit_obs_width(lambda w,h: grad_cylinder(w,h,45), W0, H_OBS)
zx, zy, _ = limit_slope(zx, zy, budget(5.5))
m, s_hat, _ = flat_coord(zx); obs = apply_warp(clean, s_hat).astype(np.float64)
dark = obs < 64                                   # 원래 검은 막대 화소
d6, s6 = control_points(s_hat, nx=6, ny=3)
res["spec"] = []
for p in (40, 200):
    for mode in ("add", "mul"):
        img = shade(obs, zx, zy, (0.3,0.2,1), 0.35, 0.5, 1.0, p, mode)
        u8 = (img*255).astype(np.uint8)
        dark_mean = float(img[dark].mean())
        lost = float((img[dark] > 0.6).mean())     # 검은 막대가 밝게 뒤덮인 비율
        f = bool(decode(u8)); r = bool(decode(rectify(u8, d6, s6, (H_OBS, W0))))
        res["spec"].append({"p":p,"mode":mode,"dark_mean":round(dark_mean,3),
                            "lost":round(lost,4),"first":f,"rect":r})
        print(f"  p={p:3d} {mode}  검은막대 평균밝기 {dark_mean:.3f}  "
              f"덮인비율 {lost*100:5.2f}%  1차={'O' if f else 'X'} 보정후={'O' if r else 'X'}")

print("\n=== (c') 목표 구간을 넓히는 손잡이 ===")
DMS_WIDE = [2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0]
DMS_LOW  = [2.5,2.8,3.2,3.6,4.0]
CACHE = {}
def clean_for(d):
    if d not in CACHE: CACHE[d] = render_clean(d)
    return CACHE[d]
for d in set(DMS_WIDE+DMS_LOW): clean_for(d)

def degrade(u8, rng):
    img = u8.astype(np.float32)
    k = float(rng.uniform(0.6, 2.2))                       # 초점 흐림
    img = cv2.GaussianBlur(img, (0,0), k)
    if rng.random() < 0.7:                                 # 컨베이어 모션 블러
        L = int(rng.integers(3, 11)); ker = np.zeros((L,L),np.float32)
        ker[L//2,:] = 1.0/L
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((L/2-.5,L/2-.5),
              float(rng.uniform(-25,25)),1.0), (L,L))
        ker /= ker.sum(); img = cv2.filter2D(img, -1, ker)
    img += rng.normal(0, rng.uniform(2,9), img.shape)      # 센서 노이즈
    img = np.clip(img,0,255).astype(np.uint8)
    q = int(rng.integers(45, 92))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,q])[1], 0)

def run(dms, c_min, use_degrade, n=120, nx=6, seed=42):
    rng = np.random.default_rng(seed); cnt = {"목표":0,"불가":0,"1차성공":0}
    for _ in range(n):
        dm = float(rng.choice(dms)); cl = clean_for(dm); h0, w0 = cl.shape
        smax = math.sqrt(max((dm/c_min)**2 - 1, 1e-6))
        flat = rng.random() < 0.10
        preset = "flat" if flat else rng.choice(["crease","sine","crumple","cyl"], p=[.40,.35,.15,.10])
        rho = 0.0 if flat else rng.uniform(0.3,1.0)
        K = int(rng.choice([1,2,3], p=[.45,.35,.20]))
        def make(w,h):
            zx=np.zeros((h,w)); zy=np.zeros((h,w))
            if preset=="flat": return zx,zy
            sh = rng.dirichlet(np.ones(K+1))
            a,b = grad_cylinder(w,h,math.degrees(math.atan(rho*smax*sh[0]))); zx+=a; zy+=b
            if preset=="cyl": return zx,zy
            for i in range(K):
                si = rho*smax*sh[i+1]; psi = rng.uniform(-45,45)
                if preset=="crease":
                    a,b = grad_crease(w,h,si,rng.uniform(2,20),psi_deg=psi,offset=rng.uniform(-.3,.3)*w)
                elif preset=="sine":
                    a,b = grad_sine(w,h,si,rng.uniform(0.35,1.2)*w,psi_deg=psi,phase=rng.uniform(0,6.28))
                else:
                    a,b = grad_crumple(w,h,si,rng.uniform(.4,.9)*w,3,seed=int(rng.integers(1e6)))
                zx+=a; zy+=b
            return zx,zy
        w, zx, zy = fit_obs_width(make, w0, H_OBS)
        zx, zy, _ = limit_slope(zx, zy, smax)
        mm, s_hat, _ = flat_coord(zx)
        obs = apply_warp(cl, s_hat)
        if not flat:
            obs = (shade(obs.astype(np.float64), zx, zy,
                         (rng.uniform(-.5,.5),rng.uniform(-.5,.5),1), 0.35, 0.5,
                         float(rng.uniform(0,1.0)), float(rng.uniform(40,400)))*255).astype(np.uint8)
        if use_degrade: obs = degrade(obs, rng)
        d, s = control_points(s_hat, nx=nx, ny=3)
        f = bool(decode(obs)); r = bool(decode(rectify(obs, d, s, (H_OBS, w0))))
        cnt["1차성공" if f else ("목표" if r else "불가")] += 1
    return {k: round(100*v/n) for k,v in cnt.items()}

res["knobs"] = []
for label, dms, c_min, deg in [
    ("A 현재 (c=2.0, 열화없음)",      DMS_WIDE, 2.0, False),
    ("B c_min=1.0",                   DMS_WIDE, 1.0, False),
    ("C c_min=2.0 + 블러·노이즈",     DMS_WIDE, 2.0, True),
    ("D c_min=1.0 + 블러·노이즈",     DMS_WIDE, 1.0, True),
    ("E d_m0 2.5~4.0 + c=1.0 + 열화", DMS_LOW,  1.0, True),
]:
    out = run(dms, c_min, deg)
    res["knobs"].append({"label":label, **out})
    print(f"  {label:32s} 목표 {out['목표']:3d}%  불가 {out['불가']:3d}%  1차성공 {out['1차성공']:3d}%")

json.dump(res, open(f"{OUT}/results7.json","w")); print("\nsaved results7.json")
