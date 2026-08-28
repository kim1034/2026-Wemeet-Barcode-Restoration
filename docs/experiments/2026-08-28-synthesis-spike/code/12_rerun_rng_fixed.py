"""버그 수정 재측정 — 난수를 make() 밖에서 한 번만 뽑는다.
   + 증강(밀집 대응장 G) 검증을 목표 구간 샘플로 다시."""
import json, math, os
import numpy as np, cv2
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_cylinder,
                   grad_sine, limit_slope, rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
ALL = [1.6,1.8,2.0,2.2,2.5,2.8,3.2,3.6,4.0,5.0,6.5,8.0]
CACHE = {d: render_clean(d) for d in ALL}

def shade(base, zx, zy, l, ka, kd, ks, p):
    n = np.stack([-zx,-zy,np.ones_like(zx)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
    l=np.array(l,float); l/=np.linalg.norm(l); hv=l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    return (np.clip(base/255.0*(ka+kd*np.clip(n@l,0,None))+ks*np.clip(n@hv,0,None)**p,0,1)*255).astype(np.uint8)

def sample_params(rng, dms, d_lo, d_hi):
    """난수는 전부 여기서 한 번만 뽑는다.  make() 는 순수 함수가 된다."""
    dm = float(rng.choice(dms))
    d_t = float(rng.uniform(d_lo, min(d_hi, dm*0.97)))
    P = {"d_m0": dm, "d_t": d_t,
         "S_t": math.sqrt(max((dm/d_t)**2-1, 1e-9)),
         "preset": str(rng.choice(["crease","sine","crumple","cyl"], p=[.45,.30,.15,.10])),
         "K": int(rng.choice([1,2], p=[.7,.3])),
         "cyl_share": float(rng.uniform(0.15,0.45)),
         "light": (float(rng.uniform(-.5,.5)), float(rng.uniform(-.5,.5)), 1.0),
         "ks": float(rng.uniform(0,0.9)), "p": float(rng.uniform(40,300)),
         "sigma": float(rng.uniform(0,1.2)), "noise": float(rng.uniform(1,5)),
         "jpeg": int(rng.integers(70,95)), "comps": []}
    for _ in range(P["K"]):
        P["comps"].append({"psi": float(rng.uniform(-45,45)),
                           "w_c": float(rng.uniform(1.,9.)),
                           "lam_f": float(rng.uniform(.6,1.4)),
                           "phase": float(rng.uniform(0,6.28)),
                           "lam0_f": float(rng.uniform(.6,1.0)),
                           "seed": int(rng.integers(1e6)),
                           "offset_f": float(rng.uniform(-.3,.3))})
    return P

def make_grad(P):
    def make(w, h):
        zx = np.zeros((h,w)); zy = np.zeros((h,w)); S = P["S_t"]
        a,b = grad_cylinder(w,h,math.degrees(math.atan(S*P["cyl_share"]))); zx+=a; zy+=b
        if P["preset"] == "cyl": return zx, zy
        for c in P["comps"]:
            si = S*(1-P["cyl_share"])/P["K"]
            if P["preset"] == "crease":
                a,b = grad_crease(w,h,si,c["w_c"],psi_deg=c["psi"],offset=c["offset_f"]*w)
            elif P["preset"] == "sine":
                a,b = grad_sine(w,h,si,c["lam_f"]*w,psi_deg=c["psi"],phase=c["phase"])
            else:
                r2 = np.random.default_rng(c["seed"]); zz = np.zeros((h,w)); yy = np.zeros((h,w))
                for j in range(3):
                    aa,bb = grad_sine(w,h,si/3, c["lam0_f"]*w/(2**j),
                                      psi_deg=float(r2.uniform(-25,25)), phase=float(r2.uniform(0,6.28)))
                    zz+=aa; yy+=bb
                a,b = zz,yy
            zx+=a; zy+=b
        return zx, zy
    return make

def run(n, nx, dms, d_lo, d_hi, use_deg, seed=42):
    rng = np.random.default_rng(seed); cnt={"목표":0,"불가":0,"1차성공":0}
    for _ in range(n):
        P = sample_params(rng, dms, d_lo, d_hi)
        cl = CACHE[P["d_m0"]]; h0, w0 = cl.shape
        w, zx, zy = fit_obs_width(make_grad(P), w0, H_OBS)
        pk = float(np.abs(zx).max())
        if pk > 1e-6: zx *= P["S_t"]/pk; zy *= P["S_t"]/pk
        m, s_hat, _ = flat_coord(zx)
        obs = apply_warp(cl, s_hat)
        obs = shade(obs.astype(np.float64), zx, zy, P["light"], 0.35, 0.5, P["ks"], P["p"])
        if use_deg:
            s = P["sigma"]
            im = cv2.GaussianBlur(obs.astype(np.float32),(0,0),s) if s>.05 else obs.astype(np.float32)
            im = np.clip(im + np.random.default_rng(P["jpeg"]).normal(0,P["noise"],im.shape),0,255).astype(np.uint8)
            obs = cv2.imdecode(cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,P["jpeg"]])[1],0)
        d, s = control_points(s_hat, nx=nx, ny=3)
        f = bool(decode(obs)); r = bool(decode(rectify(obs, d, s, (H_OBS, w0))))
        cnt["1차성공" if f else ("목표" if r else "불가")] += 1
    return {k: round(100*v/n) for k,v in cnt.items()}

print("=== 버그 수정 후 재측정 (N=120, nx=6, d_target 1.4~2.6px, 약한열화) ===")
res["byres"]=[]
for lab, dms in [("저해상도 1.6~2.2",[1.6,1.8,2.0,2.2]), ("중저 2.0~2.8",[2.0,2.2,2.5,2.8]),
                 ("중 2.8~4.0",[2.8,3.2,3.6,4.0]), ("고 5.0~8.0",[5.0,6.5,8.0]),
                 ("전체 1.6~8.0",ALL)]:
    o = run(120, 6, dms, 1.4, 2.6, True)
    res["byres"].append({"label":lab, **o})
    print(f"  {lab:16s} 목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

print("\n=== 저해상도에서 열화 유무 ===")
res["deg"]=[]
for lab, dg in [("열화 없음",False),("약한 열화",True)]:
    o = run(120, 6, [1.6,1.8,2.0,2.2], 1.4, 2.6, dg)
    res["deg"].append({"label":lab, **o}); print(f"  {lab:10s} 목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

print("\n=== nx 비교 (저해상도) ===")
res["nx"]=[]
for nx in (4,6,8):
    o = run(120, nx, [1.6,1.8,2.0,2.2], 1.4, 2.6, True)
    res["nx"].append({"nx":nx, **o}); print(f"  nx={nx}: 목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

json.dump(res, open(f"{OUT}/results12.json","w"), ensure_ascii=False)
print("\nsaved results12.json")
