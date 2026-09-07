"""재실험 R1~R5."""
import json, math, os, time
import numpy as np, cv2
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_cylinder,
                   grad_sine, rectify, render_clean, tps_flow)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
DMS = [1.6,1.8,2.0,2.2,2.5]
CACHE = {d: render_clean(d) for d in DMS}
CL55 = render_clean(5.5)

def shade(base, zx, zy, l, ka, kd, ks, p):
    n = np.stack([-zx,-zy,np.ones_like(zx)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
    l=np.array(l,float); l/=np.linalg.norm(l); hv=l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    img = np.clip(base/255.0*(ka+kd*np.clip(n@l,0,None))+ks*np.clip(n@hv,0,None)**p,0,1)
    return (img*255).astype(np.uint8), float((img>=0.99).mean())

# ============ R1. 재조정 위치가 결과를 바꾸는가 =============================
def make_fn(R, normalize_inside):
    """normalize_inside=True 면 make() 안에서 목표 기울기로 맞춘다."""
    def make(w,h):
        St = R["S_t"]; zx=np.zeros((h,w)); zy=np.zeros((h,w))
        a,b = grad_cylinder(w,h,math.degrees(math.atan(St*R["cyl"]))); zx+=a; zy+=b
        rest = St*(1-R["cyl"])
        if R["preset"]=="crease":
            c,d = grad_crease(w,h,rest,R["w_c"],psi_deg=R["psi"],offset=R["off"]*w)
        elif R["preset"]=="sine":
            c,d = grad_sine(w,h,rest,R["lam_f"]*w,psi_deg=R["psi"],phase=R["phase"])
        else:
            wts = np.array([R["pers"]**j for j in range(3)]); wts/=wts.sum()
            c=np.zeros((h,w)); d=np.zeros((h,w))
            for j in range(3):
                aa,bb = grad_sine(w,h,rest*wts[j],R["lam_f"]*w/(2**j),
                                  psi_deg=R["psis"][j],phase=R["phs"][j])
                c+=aa; d+=bb
        zx+=c; zy+=d
        if normalize_inside:
            pk=float(np.abs(zx).max())
            if pk>1e-6: zx=zx*St/pk; zy=zy*St/pk
        return zx, zy
    return make

def run_zone(R, normalize_inside, nx=6, deg=True):
    cl = CACHE[R["d_m0"]]; h0,w0 = cl.shape
    w, zx, zy = fit_obs_width(make_fn(R, normalize_inside), w0, H_OBS)
    if not normalize_inside:
        pk=float(np.abs(zx).max())
        if pk>1e-6: zx*=R["S_t"]/pk; zy*=R["S_t"]/pk
    m, s_hat, _ = flat_coord(zx)
    obs, sat = shade(apply_warp(cl, s_hat).astype(np.float64), zx, zy, R["light"], .35,.5,R["ks"],R["p"])
    if deg and R["sigma"]>.05:
        obs = cv2.GaussianBlur(obs.astype(np.float32),(0,0),R["sigma"]).astype(np.uint8)
    d,s = control_points(s_hat, nx=nx, ny=3)
    f=bool(decode(obs)); r=bool(decode(rectify(obs,d,s,(H_OBS,w0))))
    return ("1차성공" if f else ("목표" if r else "불가")), float(m.min()), sat, obs, d, s, w0

def draw(rng, presets=("crease","sine","octave"), probs=(.55,.30,.15)):
    dm=float(rng.choice(DMS)); dt=float(rng.uniform(1.3,min(2.6,dm*.95)))
    return {"d_m0":dm,"d_t":dt,"S_t":math.sqrt(max((dm/dt)**2-1,1e-9)),
            "preset":str(rng.choice(presets,p=probs)),"cyl":float(rng.uniform(.15,.45)),
            "w_c":float(rng.uniform(1.,9.)),"lam_f":float(rng.uniform(.5,1.3)),
            "psi":float(rng.uniform(-45,45)),"off":float(rng.uniform(-.3,.3)),
            "phase":float(rng.uniform(0,6.28)),"pers":0.5,
            "psis":[float(rng.uniform(-25,25)) for _ in range(3)],
            "phs":[float(rng.uniform(0,6.28)) for _ in range(3)],
            "light":(float(rng.uniform(-.5,.5)),float(rng.uniform(-.5,.5)),1.0),
            "ks":float(rng.uniform(0,.9)),"p":float(rng.uniform(40,300)),
            "sigma":float(rng.uniform(0,.8))}

print("=== R1. 재조정 위치 비교 (N=120, 저해상도) ===")
res["R1"]=[]
for inside in (False, True):
    rng=np.random.default_rng(77); cnt={"목표":0,"불가":0,"1차성공":0}
    for _ in range(120):
        z,_,_,_,_,_,_ = run_zone(draw(rng), inside)
        cnt[z]+=1
    o={k:round(100*v/120) for k,v in cnt.items()}
    res["R1"].append({"normalize_inside":inside, **o})
    print(f"  make() 안에서 정규화={str(inside):5s}  목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

print("\n=== R1b. 옥타브 감쇠 (정규화 위치 수정, N=60) ===")
res["R1b"]=[]
for pers in (0.35,0.5,0.7,1.0):
    rng=np.random.default_rng(5); cnt={"목표":0,"불가":0,"1차성공":0}
    for _ in range(60):
        R=draw(rng,("octave",),(1.0,)); R["pers"]=pers
        z,_,_,_,_,_,_ = run_zone(R, True)
        cnt[z]+=1
    o={k:round(100*v/60) for k,v in cnt.items()}
    res["R1b"].append({"persistence":pers, **o})
    print(f"  persistence={pers:.2f}  목표 {o['목표']:3d}%  불가 {o['불가']:3d}%  1차성공 {o['1차성공']:3d}%")

# ============ R2. 하이라이트 ↔ 접힘 =========================================
print("\n=== R2. 하이라이트 위치 ↔ 접힘 (접힘 = |dz_x/dx| 최대점) ===")
SM=budget(5.5); H5,W5 = CL55.shape
w, zx, zy = fit_obs_width(lambda w,h: grad_crease(w,h,math.tan(math.radians(60)),6.0,
                                                  offset=0.15*784), W5, H_OBS)
k=min(1.0,SM/float(np.abs(zx).max())); zx*=k; zy*=k
row=zx[H_OBS//2]; fold=int(np.argmax(np.abs(np.gradient(row))))
def spec(zx,zy,l,p):
    n=np.stack([-zx,-zy,np.ones_like(zx)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
    l=np.array(l,float); l/=np.linalg.norm(l); hv=l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    return np.clip(n@hv,0,None)**p
rng=np.random.default_rng(9); bp=[]; mk=[]
for az in np.linspace(0,2*np.pi,12,endpoint=False):
    l=(0.6*math.cos(az),0.6*math.sin(az),1.0)
    bp.append(int(np.argmax(spec(zx,zy,l,200)[H_OBS//2])))
    mk.append(int(rng.integers(0,zx.shape[1])))
W=zx.shape[1]
bpd=[abs(c-fold) for c in bp]; mkd=[abs(c-fold) for c in mk]
res["R2"]={"fold_col":fold,"w_obs":W,"bp_mean":round(float(np.mean(bpd)),1),
           "bp_max":int(max(bpd)),"mask_mean":round(float(np.mean(mkd)),1),
           "bp_spread":int(max(bp)-min(bp)),"bp_cols":bp}
print(f"  접힘 {fold}px / 폭 {W}px")
print(f"  Blinn-Phong: 접힘에서 평균 {np.mean(bpd):.1f}px (최대 {max(bpd)}px), 조명에 따른 이동폭 {max(bp)-min(bp)}px")
print(f"  무작위 마스크: 접힘에서 평균 {np.mean(mkd):.1f}px")

# ============ R3. 리사이즈 불변성 · letterbox ===============================
print("\n=== R3. 리사이즈 불변성 (목표 구간 샘플로) ===")
rng=np.random.default_rng(31); sample=None
for _ in range(400):
    R=draw(rng,("crease",),(1.0,))
    z,mm,sat,obs,d,s,w0 = run_zone(R, True, deg=False)
    if z=="목표": sample=(R,obs,d,s,w0,mm); break
res["R3"]={}
if sample is None:
    print("  목표 구간 샘플을 못 찾았다"); res["R3"]["found"]=False
else:
    R,obs,d,s,w0,mm = sample
    print(f"  샘플: d_m0={R['d_m0']}, 최소모듈 {mm*R['d_m0']:.2f}px, 관측폭 {obs.shape[1]}px")
    ref = rectify(obs, d, s, (H_OBS, w0)).astype(np.float64)
    print(f"  원본 그대로 → {decode(ref.astype(np.uint8)) or '실패'}")
    rows=[]
    for target in (192, 256, 384, 512):
        sc=target/obs.shape[1]
        small=cv2.resize(obs,(target,max(8,int(obs.shape[0]*sc))),interpolation=cv2.INTER_AREA)
        rect=rectify(small,d,s,(H_OBS,w0))
        txt=decode(rect); diff=float(np.abs(rect.astype(np.float64)-ref).mean())
        rows.append({"input_w":target,"decode":txt,"diff_vs_full":round(diff,2)})
        print(f"  입력 폭 {target:4d}px → {txt or '실패':12s} 원본기준 평균차 {diff:6.2f}")
    res["R3"]={"found":True,"mod_min":round(mm*R['d_m0'],2),"rows":rows}
    # letterbox
    h_,w_=obs.shape; side=max(h_,w_); top=(side-h_)//2; left=(side-w_)//2
    lb=cv2.copyMakeBorder(obs,top,side-h_-top,left,side-w_-left,cv2.BORDER_CONSTANT,value=255)
    s_lb=(s*np.array([w_-1,h_-1])+np.array([left,top]))/np.array([side-1,side-1])
    ok=decode(rectify(lb,d,s_lb,(H_OBS,w0))); bad=decode(rectify(lb,d,s,(H_OBS,w0)))
    res["R3"]["letterbox"]={"converted":ok,"naive":bad}
    print(f"  letterbox 환산함 → {ok or '실패'}   환산 안 함 → {bad or '실패'}  (뒤쪽이 실패해야 정상)")

# ============ R4. TPS 격자 축소 배수 ========================================
print("\n=== R4. tps_flow — 격자를 얼마나 줄이면 20ms 에 드나 ===")
res["R4"]=[]
for (h,w),label in [((280,380),"380x280"),((220,784),"784x220")]:
    for nx in (4,6):
        n=nx*3
        dst=np.stack(np.meshgrid(np.linspace(0,w-1,nx),np.linspace(0,h-1,3)),-1).reshape(-1,2).astype(np.float64)
        src=dst+np.random.default_rng(0).normal(0,6,dst.shape)
        row={"shape":label,"nx":nx}
        for div in (1,2,4,8,16):
            hh,ww=max(8,h//div),max(8,w//div)
            tps_flow(src/div,dst/div,(hh,ww))
            t0=time.perf_counter()
            for _ in range(3):
                mx,my=tps_flow(src/div,dst/div,(hh,ww))
                if div>1:
                    mx=cv2.resize(mx,(w,h),interpolation=cv2.INTER_LINEAR)*div
                    my=cv2.resize(my,(w,h),interpolation=cv2.INTER_LINEAR)*div
            row[f"div{div}_ms"]=round((time.perf_counter()-t0)/3*1000,1)
        res["R4"].append(row)
        print(f"  {label} nx={nx}: " + "  ".join(f"1/{d}={row[f'div{d}_ms']:6.1f}ms" for d in (1,2,4,8,16)))

# ============ R5. confidence 라벨 안정성 ====================================
print("\n=== R5. 라벨 안정성 — 파라미터를 ±1% 흔들면 라벨이 유지되나 (N=60) ===")
rng=np.random.default_rng(404); same=0; tot=0; flips={}
for _ in range(60):
    R=draw(rng)
    z0,_,_,_,_,_,_ = run_zone(R, True)
    R2=dict(R); R2["S_t"]=R["S_t"]*1.01; R2["off"]=R["off"]+0.01; R2["psi"]=R["psi"]+0.45
    z1,_,_,_,_,_,_ = run_zone(R2, True)
    tot+=1; same += (z0==z1)
    if z0!=z1: flips[f"{z0}->{z1}"]=flips.get(f"{z0}->{z1}",0)+1
res["R5"]={"n":tot,"stable":same,"stable_pct":round(100*same/tot,1),"flips":flips}
print(f"  라벨 유지 {same}/{tot} = {100*same/tot:.1f}%")
for k,v in sorted(flips.items(), key=lambda x:-x[1]): print(f"    {k}: {v}")

json.dump(res, open(f"{OUT}/results18.json","w"), ensure_ascii=False)
print("\nsaved results18.json")
