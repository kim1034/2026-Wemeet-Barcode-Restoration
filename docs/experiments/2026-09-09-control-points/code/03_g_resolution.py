"""표본 정의의 타당성 점검 — 복원 상한이 G 의 이산화 산물은 아닌가.

이 실험은 "밀집 대응장 G 로 직접 편 것" 을 복원 가능성의 상한으로 삼아 표본을
n_x 와 무관하게 고정한다. 그런데 build_G 는 펴진 축을 n_u=513 으로만 샘플한다.
크롭이 그보다 넓으면(H 버킷 평균 593 px) 상한 자체가 해상도에 깎여서, n_x 에
돌릴 헤드룸을 G 의 이산화가 만들어낸 것일 수 있다. 그러면 실험 전체가 무효다.

그래서 n_u 를 8배까지 올려 가며 상한이 움직이는지 본다. 안 움직이면 상한은
실제 물리이고 표본 정의가 안전하다.

1차 디코딩 성공률은 G 를 안 쓰므로 n_u 와 무관해야 한다 — 이 열이 안 변하는
것이 셋업이 제대로 물려 있다는 확인이다.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import wemeet.data.synthesis as syn
import wemeet.data.warp as warp
from common import decode, remap_G
from wemeet.data.synthesis import H_OBS, build, draw_recipe

N = 300
SEED = 2026
N_US = (513, 1025, 2049, 4097)
_ORIG = warp.build_G


def patched(n_u):
    def f(s_hat, n_v=33, _ignored=None):
        return _ORIG(s_hat, n_v=n_v, n_u=n_u)
    return f


def sweep(bucket):
    out = {}
    for n_u in N_US:
        # synthesis 는 build_G 를 이름으로 가져다 쓰므로 양쪽 다 갈아끼운다.
        syn.build_G = warp.build_G = patched(n_u)
        rng = np.random.default_rng(SEED)   # 루프 밖에서 한 번만 (스파이크 09·10 의 버그)
        first = g_ok = 0
        widths = []
        for i in range(N):
            s = build(draw_recipe(rng, bucket, i))
            widths.append(int(s.obs.shape[1]))
            if decode(s.obs) is not None:
                first += 1
                continue
            if decode(remap_G(s.obs, s.g, s.u_lo, s.u_hi, (H_OBS, s.w_flat))) is not None:
                g_ok += 1
        out[n_u] = {"first_ok": first, "g_remap_ok": g_ok, "n": N,
                    "mean_crop_w": float(np.mean(widths))}
    syn.build_G = warp.build_G = _ORIG
    return out


if __name__ == "__main__":
    result = {"seed": SEED, "n_per_cell": N, "buckets": {b: sweep(b) for b in ("L", "M", "H")}}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "03_g_resolution.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    for b, rows in result["buckets"].items():
        w = rows[513]["mean_crop_w"]
        cells = "  ".join(f"n_u={k}: {100*v['g_remap_ok']/N:5.2f}%" for k, v in rows.items())
        first = {v["first_ok"] for v in rows.values()}
        print(f"[{b}] mean crop {w:.0f}px  first_ok(should be constant)={sorted(first)}  {cells}")
