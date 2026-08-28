"""목표 구간을 '겨냥해서' 뽑으면 비율이 오르는가.
   핵심: 기울기 예산을 '상한'이 아니라 '목표 최소 모듈 폭'으로 역산해 정확히 맞춘다."""
import json, math, os
import numpy as np, cv2
from common import (H_OBS, control_points, decode, flat_coord, fit_obs_width,
                   apply_warp, grad_crease, grad_crumple, grad_cylinder,
                   grad_sine, rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
DMS = [3.0,3.5,4.0,4.5,5.0,5.5,6.5,8.0]
CACHE = {d: render_clean(d) for d in DMS}

def shade(base, zx, zy, l, ka, kd, ks, p):
    n = np.stack([-zx,-zy,np.ones_like(zx)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
    l = np.array(l,float); l/=np.linalg.norm(l); hv = l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    img = base/255.0*(ka+kd*np.clip(n@l,0,None)) + ks*np.clip(n@hv,0,None)**p
    return (np.clip(img,0,1)*255).astype(np.uint8)

def mild(u8, rng, smax_sigma):
    s = float(rng.uniform(0, smax_sigma))
    img = cv2.GaussianBlur(u8.astype(np.float32),(0,0),s) if s>0.05 else u8.astype(np.float32)
    img = np.clip(img + rng.normal(0, rng.uniform(1,5), img.shape),0,255).astype(np.uint8)
    q = int(rng.integers(70,95))
    return cv2.imdecode(cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,q])[1],0)

def run(n, nx, d_lo, d_hi, sig, seed=7, optics=True):
    rng = np.random.default_rng(seed); cnt={"목표":0,"불가":0,"1차성공":0}
    for _ in range(n):
        dm = float(rng.choice(DMS)); cl = CACHE[dm]; h0,w0 = cl.shape
        d_t = float(rng.uniform(d_lo, min(d_hi, dm*0.95)))
        S_t = math.sqrt(max((dm/d_t)**2 - 1, 1e-9))          # 목표를 정확히 겨냥
        preset = rng.choice(["crease","sine","crumple","cyl"], p=[.50,.30,.10,.10])
        K = int(rng.choice([1,2], p=[.7,.3]))
        def make(w,h):
            zx=np.zeros((h,w)); zy=np.zeros((h,w))
            share = rng.uniform(0.15,0.45)                    # 원통은 조연
            a,b = grad_cylinder(w,h,math.degrees(math.atan(S_t*share))); zx+=a; zy+=b
            if preset=="cyl": return zx,zy
            for i in range(K):
                si = S_t*(1-share)/K; psi = rng.uniform(-45,45)
                if preset=="crease":
                    a,b = grad_crease(w,h,si,rng.uniform(1.0,9.0),psi_deg=psi,
                                      offset=rng.uniform(-.3,.3)*w)
                elif preset=="sine":
                    a,b = grad_sine(w,h,si,rng.uniform(0.6,1.4)*w,psi_deg=psi,phase=rng.uniform(0,6.28))
                else:
                    a,b = grad_crumple(w,h,si,rng.uniform(.6,1.0)*w,3,seed=int(rng.integers(1e6)))
                zx+=a; zy+=b
            return zx,zy
        w, zx, zy = fit_obs_width(make, w0, H_OBS)
        peak = float(np.abs(zx).max())
        if peak > 1e-6: zx *= S_t/peak; zy *= S_t/peak         # 정확히 맞춘다
        m, s_hat, _ = flat_coord(zx)
        obs = apply_warp(cl, s_hat)
        if optics:
            obs = shade(obs.astype(np.float64), zx, zy,
                        (rng.uniform(-.5,.5),rng.uniform(-.5,.5),1), 0.35, 0.5,
                        float(rng.uniform(0,0.9)), float(rng.uniform(60,400)))
        if sig>0: obs = mild(obs, rng, sig)
        d, s = control_points(s_hat, nx=nx, ny=3)
        f = bool(decode(obs)); r = bool(decode(rectify(obs, d, s, (H_OBS, w0))))
        cnt["1차성공" if f else ("목표" if r else "불가")] += 1
    return {k: round(100*v/n) for k,v in cnt.items()}

print("=== 목표 최소 모듈 폭을 겨냥해서 뽑기 (N=120, nx=6) ===")
res["targeted"]=[]
for lab, lo, hi, sg in [
    ("d_target 1.8~3.4px, 열화없음", 1.8, 3.4, 0.0),
    ("d_target 2.0~3.2px, 열화없음", 2.0, 3.2, 0.0),
    ("d_target 2.0~3.2px, 약한열화", 2.0, 3.2, 1.4),
    ("d_target 2.4~4.0px, 약한열화", 2.4, 4.0, 1.6),
    ("d_target 2.8~4.5px, 열화 강",  2.8, 4.5, 2.2),
]:
    o = run(120, 6, lo, hi, sg)
    res["targeted"].append({"label":lab, **o})
    print(f"  {lab:30s} 목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

print("\n=== 최고 설정에서 nx 비교 ===")
res["nx"]=[]
for nx in (4,6,8):
    o = run(120, nx, 2.4, 4.0, 1.6)
    res["nx"].append({"nx":nx, **o})
    print(f"  nx={nx}: 목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

json.dump(res, open(f"{OUT}/results9.json","w")); print("\nsaved results9.json")
