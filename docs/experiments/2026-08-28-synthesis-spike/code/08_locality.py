"""가설: 목표 구간을 만드는 건 압축의 '깊이'가 아니라 '국소 급격함'이다."""
import json, math, os
import numpy as np, cv2
from common import (H_OBS, budget, control_points, decode, flat_coord,
                   fit_obs_width, apply_warp, grad_crease, grad_sine,
                   limit_slope, rectify, render_clean)
OUT = os.path.dirname(os.path.abspath(__file__)); res = {}
clean = render_clean(5.5); H0, W0 = clean.shape; SM = budget(5.5)

def zone(zx, zy, nx=6, cl=clean, w0=W0, deg=None):
    m, s_hat, _ = flat_coord(zx)
    obs = apply_warp(cl, s_hat)
    if deg is not None: obs = deg(obs)
    d, s = control_points(s_hat, nx=nx, ny=3)
    f = bool(decode(obs)); r = bool(decode(rectify(obs, d, s, (H_OBS, w0))))
    return ("1차성공" if f else ("목표" if r else "불가")), float(m.min())

print("=== 접힘: 날카로움(w_c) x 각도(α) ===")
print(f"{'w_c':>5} | " + " ".join(f"{a:>12}" for a in (30,45,55,65,75)))
res["crease"] = []
for wc in (1.0,1.5,2.0,3.0,4.0,6.0,9.0,14.0,20.0):
    cells=[]
    for a in (30,45,55,65,75):
        w, zx, zy = fit_obs_width(
            lambda w,h,A=a,W=wc: grad_crease(w,h,math.tan(math.radians(A)),W), W0, H_OBS)
        zx, zy, sc = limit_slope(zx, zy, SM)
        z, mn = zone(zx, zy)
        res["crease"].append({"w_c":wc,"alpha":a,"zone":z,"m_min":round(mn,3),"scale":round(sc,3)})
        cells.append(f"{z}({mn*5.5:.1f})")
    print(f"{wc:>5.1f} | " + " ".join(f"{c:>12}" for c in cells))

print("\n=== 물결: 파장 x 세기 (참고) ===")
res["sine"]=[]
print(f"{'λ/W':>5} | " + " ".join(f"{s:>12}" for s in (0.4,0.8,1.2,1.8,2.5)))
for lf in (0.30,0.45,0.60,0.80,1.10):
    cells=[]
    for sl in (0.4,0.8,1.2,1.8,2.5):
        w, zx, zy = fit_obs_width(lambda w,h,L=lf,S=sl: grad_sine(w,h,S,L*w), W0, H_OBS)
        zx, zy, _ = limit_slope(zx, zy, SM)
        z, mn = zone(zx, zy)
        res["sine"].append({"lam":lf,"slope":sl,"zone":z,"m_min":round(mn,3)})
        cells.append(f"{z}({mn*5.5:.1f})")
    print(f"{lf:>5.2f} | " + " ".join(f"{c:>12}" for c in cells))

print("\n=== 목표 구간이 약한 열화를 견디는가 (w_c=2, α=55) ===")
rng = np.random.default_rng(0)
def mild(sig, q):
    def f(u8):
        img = cv2.GaussianBlur(u8.astype(np.float32),(0,0),sig)
        img = np.clip(img + rng.normal(0,3,img.shape),0,255).astype(np.uint8)
        return cv2.imdecode(cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,q])[1],0)
    return f
res["robust"]=[]
w, zx0, zy0 = fit_obs_width(lambda w,h: grad_crease(w,h,math.tan(math.radians(55)),2.0), W0, H_OBS)
zx0, zy0, _ = limit_slope(zx0, zy0, SM)
for sig,q in [(0.0,100),(0.6,92),(1.0,85),(1.6,75),(2.2,60)]:
    d = None if sig==0 else mild(sig,q)
    z,_ = zone(zx0, zy0, deg=d)
    res["robust"].append({"sigma":sig,"jpeg":q,"zone":z})
    print(f"  블러 σ={sig:.1f} JPEG q={q:3d}  →  {z}")

json.dump(res, open(f"{OUT}/results8.json","w")); print("\nsaved results8.json")
