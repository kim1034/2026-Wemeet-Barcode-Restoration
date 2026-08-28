"""1단계 §2 수식 마무리 + 2단계 §3 광학 정합성."""
import json, math, os
import numpy as np, cv2
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_cylinder,
                   grad_sine, rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
clean = render_clean(5.5); H0, W0 = clean.shape; SM = budget(5.5)

def normals(zx, zy):
    n = np.stack([-zx,-zy,np.ones_like(zx)],-1)
    return n/np.linalg.norm(n,axis=-1,keepdims=True)

def spec_map(zx, zy, l, p):
    n = normals(zx,zy); l=np.array(l,float); l/=np.linalg.norm(l)
    hv = l+np.array([0.,0.,1.]); hv/=np.linalg.norm(hv)
    return np.clip(n@hv,0,None)**p

def shade(base, zx, zy, l, ka, kd, ks, p):
    n = normals(zx,zy); l=np.array(l,float); l/=np.linalg.norm(l)
    return np.clip(base/255.0*(ka+kd*np.clip(n@l,0,None)) + ks*spec_map(zx,zy,l,p), 0, 1)

# ---- 1a. 접힘을 α 로 파라미터화하면 max|z_x| = tan α 인가 -------------------
print("=== 1a. 접힘 tan α (W=784) ===")
res["crease_tan"]=[]
for a in (30,45,55,65):
    for wc in (1.0, 4.0, 20.0, 80.0, 200.0):
        zx,_ = grad_crease(784, 60, math.tan(math.radians(a)), wc)
        meas = float(np.abs(zx).max()); pred = math.tan(math.radians(a))
        res["crease_tan"].append({"alpha":a,"w_c":wc,"pred":round(pred,4),
                                  "meas":round(meas,4),"ratio":round(meas/pred,4)})
    r = [x for x in res["crease_tan"] if x["alpha"]==a]
    print(f"  α={a}° pred={r[0]['pred']:.4f}  " +
          "  ".join(f"w_c={x['w_c']:.0f}:{x['ratio']:.3f}" for x in r))

# ---- 1b. 옥타브 감쇠 재측정 (난수 버그 없음) --------------------------------
print("\n=== 1b. 옥타브 감쇠 persistence (N=60, d_m0=2.0) ===")
c2 = render_clean(2.0); H2, W2 = c2.shape
def octaves(w,h,slope,lam0,pers,psis,phs):
    wts = np.array([pers**j for j in range(len(psis))]); wts/=wts.sum()
    zx=np.zeros((h,w)); zy=np.zeros((h,w))
    for j,(ps,ph) in enumerate(zip(psis,phs)):
        a,b = grad_sine(w,h,slope*wts[j], lam0/(2**j), psi_deg=ps, phase=ph)
        zx+=a; zy+=b
    return zx,zy
res["octave"]=[]
for pers in (0.35,0.5,0.7,1.0):
    rng = np.random.default_rng(5); cnt={"목표":0,"불가":0,"1차성공":0}
    for _ in range(60):
        dt = float(rng.uniform(1.3,1.9)); St = math.sqrt(max((2.0/dt)**2-1,1e-9))
        lam0f = float(rng.uniform(.5,1.0))
        psis = [float(rng.uniform(-25,25)) for _ in range(3)]
        phs  = [float(rng.uniform(0,6.28)) for _ in range(3)]
        w, zx, zy = fit_obs_width(lambda w,h: octaves(w,h,St,lam0f*w,pers,psis,phs), W2, H_OBS)
        pk=float(np.abs(zx).max())
        if pk>1e-6: zx*=St/pk; zy*=St/pk
        m,s,_ = flat_coord(zx); o = apply_warp(c2, s)
        d,sc = control_points(s, nx=6, ny=3)
        f=bool(decode(o)); r=bool(decode(rectify(o,d,sc,(H_OBS,W2))))
        cnt["1차성공" if f else ("목표" if r else "불가")]+=1
    out={k:round(100*v/60) for k,v in cnt.items()}
    res["octave"].append({"persistence":pers, **out})
    print(f"  persistence={pers:.2f}  목표 {out['목표']:3d}%  불가 {out['불가']:3d}%  1차성공 {out['1차성공']:3d}%")

# ---- 1c. clip 이 기하와 음영을 어긋나게 하는가 -----------------------------
print("\n=== 1c. clip vs 스케일 다운 — 기하와 음영의 일관성 ===")
w, zx_raw, zy_raw = fit_obs_width(lambda w,h: grad_cylinder(w,h,80), W0, H_OBS)  # 예산 초과
c_min = 2.0; thr = c_min/5.5
res["clip"]={}
for name in ("clip","scale"):
    if name=="clip":
        zx_shade, zy_shade = zx_raw, zy_raw                      # 음영은 원래 z 로
        m_used = np.clip(1/np.sqrt(1+zx_raw**2), thr, None)      # 기하는 잘린 m 으로
    else:
        k = min(1.0, budget(5.5)/float(np.abs(zx_raw).max()))
        zx_shade, zy_shade = zx_raw*k, zy_raw*k
        m_used = 1/np.sqrt(1+zx_shade**2)
    zx_implied = np.sqrt(np.maximum(1/m_used**2 - 1, 0))          # m 이 함의하는 z_x
    err = np.abs(zx_implied - np.abs(zx_shade))
    ang = np.abs(np.degrees(np.arctan(zx_implied)) - np.degrees(np.arctan(np.abs(zx_shade))))
    frac = float((err > 1e-6).mean())
    res["clip"][name]={"max_slope_err":round(float(err.max()),4),
                       "max_angle_err_deg":round(float(ang.max()),3),
                       "inconsistent_frac":round(frac,4)}
    print(f"  {name:6s} 불일치 화소 {frac*100:5.1f}%  최대 기울기 오차 {err.max():.4f}  "
          f"최대 법선각 오차 {ang.max():.2f}°")

# ---- 2a. 하이라이트가 곡률과 상관되는가 (Blinn-Phong vs 타원 마스크) --------
print("\n=== 2a. 하이라이트 위치 ↔ 곡률 상관 ===")
rng = np.random.default_rng(9)
w, zx, zy = fit_obs_width(lambda w,h: grad_crease(w,h,math.tan(math.radians(60)),6.0,
                                                  offset=0.15*784), W0, H_OBS)
zx, zy, = zx*min(1.0,SM/float(np.abs(zx).max())), zy*min(1.0,SM/float(np.abs(zy).max()) if np.abs(zy).max()>0 else 1)
prof = np.abs(zx[H_OBS//2]); fold_col = int(np.argmax(prof))
bp_cols=[]; mask_cols=[]
for az in np.linspace(0, 2*np.pi, 12, endpoint=False):
    l = (0.6*math.cos(az), 0.6*math.sin(az), 1.0)
    sm = spec_map(zx, zy, l, 200)
    bp_cols.append(int(np.argmax(sm[H_OBS//2])))
    mask_cols.append(int(rng.integers(0, zx.shape[1])))          # 타원 마스크 = 곡률 무관
bp_d = [abs(c-fold_col) for c in bp_cols]; mk_d = [abs(c-fold_col) for c in mask_cols]
res["highlight"]={"fold_col":fold_col,"w_obs":int(zx.shape[1]),
                  "bp_dist_mean":round(float(np.mean(bp_d)),1),
                  "bp_dist_max":int(max(bp_d)),
                  "mask_dist_mean":round(float(np.mean(mk_d)),1),
                  "bp_cols":bp_cols}
print(f"  접힘 위치 {fold_col}px / 폭 {zx.shape[1]}px")
print(f"  Blinn-Phong 하이라이트: 접힘에서 평균 {np.mean(bp_d):.1f}px (최대 {max(bp_d)}px)")
print(f"  타원 마스크(무관):      접힘에서 평균 {np.mean(mk_d):.1f}px")

# ---- 2b. 조립 순서 ---------------------------------------------------------
print("\n=== 2b. 조립 순서 ===")
m, s_hat, _ = flat_coord(zx); base = apply_warp(clean, s_hat).astype(np.float64)
L = (0.5,0.3,1.0)
correct = shade(base, zx, zy, L, .35,.5,.9,200)
res["order"]=[]
# (i) 크롭: 광학->크롭 vs 크롭->광학
h_, w_ = base.shape; x0,x1 = int(.08*w_), w_-int(.05*w_)
a1 = correct[:, x0:x1]
a2 = shade(base[:, x0:x1], zx[:, x0:x1], zy[:, x0:x1], L, .35,.5,.9,200)
d_crop = float(np.abs(a1-a2).max())
print(f"  크롭:   순서 바꿔도 최대 차이 {d_crop:.2e}  → {'동일(교환 가능)' if d_crop<1e-9 else '다름'}")
res["order"].append({"aug":"crop","max_diff":d_crop})
# (ii) 회전: 광학->회전 (맞음) vs 회전->광학(회전 안 된 z 로) (틀림)
M = cv2.getRotationMatrix2D(((w_-1)/2,(h_-1)/2), 12.0, 1.0)
b1 = cv2.warpAffine((correct*255).astype(np.uint8), M, (w_,h_), flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE).astype(np.float64)/255
rot_base = cv2.warpAffine(base.astype(np.uint8), M, (w_,h_), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
b2 = shade(rot_base, zx, zy, L, .35,.5,.9,200)
hp1 = int(np.argmax(b1[H_OBS//2])); hp2 = int(np.argmax(b2[H_OBS//2]))
print(f"  회전:   평균 차이 {np.abs(b1-b2).mean():.4f}   하이라이트 열 {hp1} vs {hp2} "
      f"({abs(hp1-hp2)}px 어긋남)")
res["order"].append({"aug":"rotate","mean_diff":round(float(np.abs(b1-b2).mean()),4),
                     "hl_shift_px":abs(hp1-hp2)})
# (iii) 블러: 광학->블러 (맞음) vs 블러->광학 (정반사가 안 흐려짐)
c1 = cv2.GaussianBlur((correct*255).astype(np.float32),(0,0),1.6)/255
c2b = shade(cv2.GaussianBlur(base.astype(np.float32),(0,0),1.6).astype(np.float64), zx, zy, L, .35,.5,.9,200)
print(f"  블러:   평균 차이 {np.abs(c1-c2b).mean():.4f}  "
      f"정반사 최대 {c1.max():.3f} vs {c2b.max():.3f}")
res["order"].append({"aug":"blur","mean_diff":round(float(np.abs(c1-c2b).mean()),4),
                     "peak_correct":round(float(c1.max()),3),"peak_wrong":round(float(c2b.max()),3)})

json.dump(res, open(f"{OUT}/results15.json","w"), ensure_ascii=False)
print("\nsaved results15.json")
