"""저해상도 구간을 열어본다. spike4 에서 d_m0=2.0~2.5 가 목표 구간이었다."""
import json, math, os
import numpy as np, cv2
from common import (H_OBS, control_points, decode, flat_coord, fit_obs_width,
                   apply_warp, grad_crease, grad_crumple, grad_cylinder,
                   grad_sine, rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
ALL = [1.6,1.8,2.0,2.2,2.5,2.8,3.2,3.6,4.0,5.0,6.5,8.0]
CACHE = {d: render_clean(d) for d in ALL}
for d in ALL: print(f"  d_m0={d}: 원본폭 {CACHE[d].shape[1]}px")

def shade(base, zx, zy, l, ka, kd, ks, p):
    n = np.stack([-zx,-zy,np.ones_like(zx)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
    l=np.array(l,float); l/=np.linalg.norm(l); hv=l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    return (np.clip(base/255.0*(ka+kd*np.clip(n@l,0,None))+ks*np.clip(n@hv,0,None)**p,0,1)*255).astype(np.uint8)

def run(n, nx, dms, d_lo, d_hi, sig, seed=11, optics=True):
    rng = np.random.default_rng(seed); cnt={"목표":0,"불가":0,"1차성공":0}; keep=[]
    for _ in range(n):
        dm = float(rng.choice(dms)); cl = CACHE[dm]; h0,w0 = cl.shape
        d_t = float(rng.uniform(d_lo, min(d_hi, dm*0.97)))
        S_t = math.sqrt(max((dm/d_t)**2-1,1e-9))
        preset = rng.choice(["crease","sine","crumple","cyl"], p=[.45,.30,.15,.10])
        K = int(rng.choice([1,2],p=[.7,.3]))
        def make(w,h):
            zx=np.zeros((h,w)); zy=np.zeros((h,w)); sh=rng.uniform(0.15,0.45)
            a,b=grad_cylinder(w,h,math.degrees(math.atan(S_t*sh))); zx+=a; zy+=b
            if preset=="cyl": return zx,zy
            for i in range(K):
                si=S_t*(1-sh)/K; psi=rng.uniform(-45,45)
                if preset=="crease": a,b=grad_crease(w,h,si,rng.uniform(1.,9.),psi_deg=psi,offset=rng.uniform(-.3,.3)*w)
                elif preset=="sine": a,b=grad_sine(w,h,si,rng.uniform(.6,1.4)*w,psi_deg=psi,phase=rng.uniform(0,6.28))
                else: a,b=grad_crumple(w,h,si,rng.uniform(.6,1.)*w,3,seed=int(rng.integers(1e6)))
                zx+=a; zy+=b
            return zx,zy
        w, zx, zy = fit_obs_width(make, w0, H_OBS)
        pk=float(np.abs(zx).max())
        if pk>1e-6: zx*=S_t/pk; zy*=S_t/pk
        m, s_hat, _ = flat_coord(zx); obs = apply_warp(cl, s_hat)
        if optics:
            obs = shade(obs.astype(np.float64), zx, zy,(rng.uniform(-.5,.5),rng.uniform(-.5,.5),1),
                        0.35,0.5,float(rng.uniform(0,0.9)),float(rng.uniform(40,300)))
        if sig>0:
            s=float(rng.uniform(0,sig))
            im=cv2.GaussianBlur(obs.astype(np.float32),(0,0),s) if s>.05 else obs.astype(np.float32)
            im=np.clip(im+rng.normal(0,rng.uniform(1,5),im.shape),0,255).astype(np.uint8)
            obs=cv2.imdecode(cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(70,95))])[1],0)
        d,s = control_points(s_hat,nx=nx,ny=3)
        f=bool(decode(obs)); r=bool(decode(rectify(obs,d,s,(H_OBS,w0))))
        z = "1차성공" if f else ("목표" if r else "불가"); cnt[z]+=1
        keep.append({"d_m0":dm,"d_t":round(d_t,2),"preset":preset,"zone":z})
    return {k:round(100*v/n) for k,v in cnt.items()}, keep

print("\n=== 해상도 구간별 (N=120, nx=6, d_target 1.4~2.6px, 약한열화) ===")
res["byres"]=[]
for lab, dms in [("저해상도 1.6~2.2", [1.6,1.8,2.0,2.2]),
                 ("중저 2.0~2.8",    [2.0,2.2,2.5,2.8]),
                 ("중 2.8~4.0",      [2.8,3.2,3.6,4.0]),
                 ("고 5.0~8.0",      [5.0,6.5,8.0]),
                 ("전체 1.6~8.0",    ALL)]:
    o,_ = run(120, 6, dms, 1.4, 2.6, 1.2)
    res["byres"].append({"label":lab, **o})
    print(f"  {lab:16s} 목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

print("\n=== 저해상도에서 d_target 범위 조정 ===")
res["tune"]=[]
for lo,hi,sg in [(1.2,2.0,0.8),(1.4,2.2,1.0),(1.6,2.4,1.2),(1.8,2.8,1.2)]:
    o,_ = run(120, 6, [1.8,2.0,2.2,2.5], lo, hi, sg)
    res["tune"].append({"lo":lo,"hi":hi,"sig":sg, **o})
    print(f"  d_target {lo}~{hi}px σ≤{sg}  목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

json.dump(res, open(f"{OUT}/results10.json","w")); print("\nsaved results10.json")
