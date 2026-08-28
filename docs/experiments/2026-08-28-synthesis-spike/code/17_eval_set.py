"""5단계 — 목표 구간 100% 평가 세트 실현성 + confidence GT 가 학습 가능한 신호인가."""
import json, math, os, time
import numpy as np, cv2
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_cylinder,
                   grad_sine, rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
DMS = [1.6,1.8,2.0,2.2,2.5]
CACHE = {d: render_clean(d) for d in DMS}

def shade(base, zx, zy, l, ka, kd, ks, p):
    n = np.stack([-zx,-zy,np.ones_like(zx)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
    l=np.array(l,float); l/=np.linalg.norm(l); hv=l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    img = np.clip(base/255.0*(ka+kd*np.clip(n@l,0,None))+ks*np.clip(n@hv,0,None)**p,0,1)
    return (img*255).astype(np.uint8), float((img>=0.99).mean())

def draw(rng):
    dm = float(rng.choice(DMS))
    dt = float(rng.uniform(1.3, min(2.6, dm*0.95)))
    return {"d_m0":dm,"d_t":dt,"S_t":math.sqrt(max((dm/dt)**2-1,1e-9)),
            "preset":str(rng.choice(["crease","sine"],p=[.65,.35])),
            "cyl":float(rng.uniform(.15,.45)),"w_c":float(rng.uniform(1.,9.)),
            "lam_f":float(rng.uniform(.5,1.3)),"psi":float(rng.uniform(-45,45)),
            "off":float(rng.uniform(-.3,.3)),"phase":float(rng.uniform(0,6.28)),
            "light":(float(rng.uniform(-.5,.5)),float(rng.uniform(-.5,.5)),1.0),
            "ks":float(rng.uniform(0,.9)),"p":float(rng.uniform(40,300)),
            "sigma":float(rng.uniform(0,.8))}

def build(R, nx=6):
    cl = CACHE[R["d_m0"]]; h0, w0 = cl.shape; St = R["S_t"]
    def make(w,h):
        a,b = grad_cylinder(w,h,math.degrees(math.atan(St*R["cyl"])))
        if R["preset"]=="crease":
            c,d = grad_crease(w,h,St*(1-R["cyl"]),R["w_c"],psi_deg=R["psi"],offset=R["off"]*w)
        else:
            c,d = grad_sine(w,h,St*(1-R["cyl"]),R["lam_f"]*w,psi_deg=R["psi"],phase=R["phase"])
        return a+c, b+d
    w, zx, zy = fit_obs_width(make, w0, H_OBS)
    pk=float(np.abs(zx).max())
    if pk>1e-6: zx*=St/pk; zy*=St/pk
    m, s_hat, _ = flat_coord(zx)
    obs, sat = shade(apply_warp(cl, s_hat).astype(np.float64), zx, zy, R["light"], .35,.5,R["ks"],R["p"])
    if R["sigma"]>.05:
        obs = cv2.GaussianBlur(obs.astype(np.float32),(0,0),R["sigma"]).astype(np.uint8)
    d, s = control_points(s_hat, nx=nx, ny=3)
    return obs, d, s, w0, float(m.min()), sat

print("=== 5a. 목표 구간 100% 평가 세트 200장 뽑기 ===")
rng = np.random.default_rng(2026)
TARGET = 200
kept=[]; tried=0; t0=time.perf_counter()
zone_all = {"목표":0,"불가":0,"1차성공":0}
records=[]
while len(kept) < TARGET and tried < 4000:
    R = draw(rng); tried += 1
    obs, d, s, w0, mmin, sat = build(R)
    f = bool(decode(obs)); r = bool(decode(rectify(obs, d, s, (H_OBS, w0))))
    z = "1차성공" if f else ("목표" if r else "불가")
    zone_all[z]+=1
    records.append({"zone":z,"m_min":round(mmin,4),"mod_min":round(mmin*R["d_m0"],3),
                    "sat":round(sat,4),"d_m0":R["d_m0"],"sigma":round(R["sigma"],2),
                    "ks":round(R["ks"],2),"p":round(R["p"],0),"preset":R["preset"]})
    if z=="목표": kept.append(R)
el = time.perf_counter()-t0
rate = len(kept)/tried
res["eval_set"]={"target":TARGET,"kept":len(kept),"tried":tried,
                 "rate":round(rate,4),"oversample_x":round(1/rate,1) if rate>0 else None,
                 "seconds":round(el,1),"per_sample_ms":round(el/tried*1000,1),
                 "zones":zone_all}
print(f"  {len(kept)}장 확보 / {tried}장 생성  → 목표 구간 {rate*100:.1f}%  초과 생성 {1/rate:.1f}배")
print(f"  총 {el:.1f}초  (1장당 {el/tried*1000:.0f}ms, 1코어)")
print(f"  1000장 평가 세트 예상: {1000/rate*el/tried:.0f}초")

print("\n=== 5b. confidence 정답이 학습 가능한 신호인가 ===")
import statistics
by = {"목표":[], "불가":[], "1차성공":[]}
for r in records: by[r["zone"]].append(r)
print(f"  {'구간':<8} {'n':>4} {'최소모듈px':>11} {'포화율%':>9} {'블러σ':>7} {'ks':>6}")
res["signal"]=[]
for z, rows in by.items():
    if not rows: continue
    mm = statistics.mean(r["mod_min"] for r in rows)
    st = statistics.mean(r["sat"] for r in rows)*100
    sg = statistics.mean(r["sigma"] for r in rows)
    ks = statistics.mean(r["ks"] for r in rows)
    res["signal"].append({"zone":z,"n":len(rows),"mod_min":round(mm,2),
                          "sat_pct":round(st,2),"sigma":round(sg,3),"ks":round(ks,3)})
    print(f"  {z:<8} {len(rows):>4} {mm:>11.2f} {st:>9.2f} {sg:>7.2f} {ks:>6.2f}")

# 최소 모듈 폭 단독으로 구간을 얼마나 가르는가
th = np.arange(1.2, 2.8, 0.05); best=(0,None)
lab = np.array([1 if r["zone"]=="목표" else 0 for r in records])
val = np.array([r["mod_min"] for r in records])
for t in th:
    pred = (val>=t).astype(int)
    acc = float((pred==lab).mean()); 
    if acc>best[0]: best=(acc,float(t))
res["single_feature"]={"best_acc":round(best[0],3),"threshold":round(best[1],2) if best[1] else None,
                       "base_rate":round(float(lab.mean()),3)}
print(f"\n  최소 모듈 폭 하나로 '목표 구간'을 맞히면 정확도 {best[0]*100:.1f}% "
      f"(문턱 {best[1]:.2f}px, 다수 클래스 기준선 {max(lab.mean(),1-lab.mean())*100:.1f}%)")

json.dump({**res,"records":records}, open(f"{OUT}/results17.json","w"), ensure_ascii=False)
print("\nsaved results17.json")
