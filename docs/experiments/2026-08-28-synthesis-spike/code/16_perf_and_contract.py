"""3단계 성능·재현성 + 4단계 계약 불변성."""
import hashlib, json, math, os, time
import numpy as np, cv2
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_cylinder,
                   grad_sine, rectify, render_clean, tps_flow)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}

def shade(base, zx, zy, l, ka, kd, ks, p):
    n = np.stack([-zx,-zy,np.ones_like(zx)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
    l=np.array(l,float); l/=np.linalg.norm(l); hv=l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    return (np.clip(base/255.0*(ka+kd*np.clip(n@l,0,None))+ks*np.clip(n@hv,0,None)**p,0,1)*255).astype(np.uint8)

CACHE = {}
def clean_for(d):
    if d not in CACHE: CACHE[d] = render_clean(d)
    return CACHE[d]

def build(recipe):
    """레시피 -> (이미지, 제어점).  난수는 레시피에만 있고 여기서 안 뽑는다."""
    dm = recipe["d_m0"]; cl = clean_for(dm); h0, w0 = cl.shape
    St = recipe["S_t"]
    def make(w,h):
        a,b = grad_cylinder(w,h,math.degrees(math.atan(St*recipe["cyl"])))
        c,d = grad_crease(w,h,St*(1-recipe["cyl"]), recipe["w_c"],
                          psi_deg=recipe["psi"], offset=recipe["off"]*w)
        return a+c, b+d
    w, zx, zy = fit_obs_width(make, w0, H_OBS)
    pk = float(np.abs(zx).max())
    if pk>1e-6: zx*=St/pk; zy*=St/pk
    m, s_hat, _ = flat_coord(zx)
    obs = shade(apply_warp(cl, s_hat).astype(np.float64), zx, zy,
                recipe["light"], .35,.5, recipe["ks"], recipe["p"])
    d, s = control_points(s_hat, nx=recipe["nx"], ny=3)
    return obs, d, s, w0

REC = {"d_m0":5.5,"S_t":1.4,"cyl":0.3,"w_c":5.0,"psi":22.0,"off":0.1,
       "light":(0.4,0.2,1.0),"ks":0.7,"p":150.0,"nx":6}

print("=== 3a. 레시피 재현성 ===")
h1 = hashlib.sha256(build(REC)[0].tobytes()).hexdigest()
h2 = hashlib.sha256(build(REC)[0].tobytes()).hexdigest()
res["repro"] = {"hash1":h1[:16],"hash2":h2[:16],"identical":h1==h2}
print(f"  같은 레시피 두 번 → {'비트 단위 동일' if h1==h2 else '다르다!'}  ({h1[:16]})")

print("\n=== 3b. 샘플 생성 속도 ===")
res["speed"]=[]
for dm, label in [(2.0,"저해상도 315px"), (5.5,"기준 784px")]:
    r = dict(REC, d_m0=dm)
    build(r)                                  # 워밍업
    t0 = time.perf_counter()
    for _ in range(20): build(r)
    dt = (time.perf_counter()-t0)/20
    res["speed"].append({"d_m0":dm,"ms":round(dt*1000,1),"per_sec":round(1/dt,1)})
    print(f"  {label:14s} {dt*1000:7.1f} ms/장  →  {1/dt:6.1f} 장/초 (1코어)")

print("\n=== 3c. tps_flow 시간 예산 20ms ===")
res["tps"]=[]
for (h,w), label in [((220,384),"384x220"), ((220,784),"784x220"), ((280,380),"380x280 설계기준")]:
    for nx in (4,6,8):
        n = nx*3
        dst = np.stack(np.meshgrid(np.linspace(0,w-1,nx), np.linspace(0,h-1,3)),-1).reshape(-1,2).astype(np.float64)
        src = dst + np.random.default_rng(0).normal(0,6,dst.shape)
        tps_flow(src,dst,(h,w))               # 워밍업
        t0=time.perf_counter()
        for _ in range(3): tps_flow(src,dst,(h,w))
        full=(time.perf_counter()-t0)/3
        # 격자 1/4 축소 + resize
        t0=time.perf_counter()
        for _ in range(3):
            mx,my = tps_flow(src/4.0, dst/4.0, (h//4, w//4))
            mx = cv2.resize(mx,(w,h),interpolation=cv2.INTER_LINEAR)*4.0
            my = cv2.resize(my,(w,h),interpolation=cv2.INTER_LINEAR)*4.0
        quarter=(time.perf_counter()-t0)/3
        res["tps"].append({"shape":label,"nx":nx,"n":n,"full_ms":round(full*1000,1),
                           "quarter_ms":round(quarter*1000,1)})
        print(f"  {label:16s} nx={nx} (n={n:2d})  전체격자 {full*1000:7.1f} ms   "
              f"1/4격자+resize {quarter*1000:6.1f} ms  {'OK' if quarter*1000<20 else '초과'}")

print("\n=== 4a. 정규화 좌표가 리사이즈에 불변인가 ===")
obs, d, s, w0 = build(REC)
res["resize"]=[]
ref = None
for target in (256, 384, 512, 784):
    scale = target/obs.shape[1]
    small = cv2.resize(obs, (target, max(8,int(obs.shape[0]*scale))), interpolation=cv2.INTER_AREA)
    rect = rectify(small, d, s, (H_OBS, w0))      # 같은 정규화 제어점을 그대로 사용
    txt = decode(rect)
    if ref is None: ref = rect.astype(np.float64)
    diff = float(np.abs(rect.astype(np.float64)-ref).mean())
    res["resize"].append({"input_w":target,"decode":txt,"diff_vs_ref":round(diff,2)})
    print(f"  입력 폭 {target:4d}px → 판독 {txt or '실패':12s} 기준(784)과 평균차 {diff:6.2f}")

print("\n=== 4b. letterbox 왕복이 좌표를 보존하는가 ===")
h_, w_ = obs.shape; side = max(h_, w_)
top = (side-h_)//2; left = (side-w_)//2
lb = cv2.copyMakeBorder(obs, top, side-h_-top, left, side-w_-left, cv2.BORDER_CONSTANT, value=255)
# 패딩된 이미지 기준으로 좌표를 옮기고, 다시 원본 기준으로 환산
s_lb = (s*np.array([w_-1,h_-1]) + np.array([left,top]))/np.array([side-1,side-1])
s_back = (s_lb*np.array([side-1,side-1]) - np.array([left,top]))/np.array([w_-1,h_-1])
gap = float(np.abs(s_back - s).max())
rect_lb = rectify(lb, d, s_lb, (H_OBS, w0))
res["letterbox"]={"roundtrip_max_err":gap,"decode_after_letterbox":decode(rect_lb),
                  "naive_decode": decode(rectify(lb, d, s, (H_OBS, w0)))}
print(f"  왕복 최대 오차 {gap:.2e}")
print(f"  환산해서 보정 → {decode(rect_lb) or '실패'}")
print(f"  환산 없이 그대로 → {decode(rectify(lb, d, s, (H_OBS, w0))) or '실패'}  (실패해야 정상)")

json.dump(res, open(f"{OUT}/results16.json","w"), ensure_ascii=False)
print("\nsaved results16.json")
