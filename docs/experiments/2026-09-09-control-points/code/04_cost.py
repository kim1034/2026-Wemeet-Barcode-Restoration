"""제어점 개수가 3단계 20ms 예산에 얼마를 먹는가. **단독 실행 전용.**

tps_flow 의 지배항은 n^3 선형계가 아니라 밀집 커널 O(H*W*n) 이다 -- 격자점
사이 모든 쌍의 거리를 재기 때문이다. 그래서 시간이 점 개수에 **선형**이고,
정확도가 "많을수록 좋다" 고 해도 비용이 상한을 만든다.

반드시 단독으로 돌릴 것. 인계 문서 실측: 병렬로 재면 같은 조건이
3.4ms -> 22.8ms 로 7배 부풀려진다. 그래서 스레드도 1로 묶는다.

인계 문서의 784x220 조건을 같이 재서 셋업이 그쪽과 맞는지 대조한다.
안 맞으면 이 표가 아니라 대조 자체가 결과다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(__file__))
from common import tps_flow  # noqa: E402

# (이름, 폭, 높이). 폭은 이번 실험에서 실측한 버킷 평균 크롭이다.
# 784x220 은 인계 문서가 실측을 보고한 조건 -- 대조용이다.
SHAPES = [("L 313x220", 313, 220), ("M 445x220", 445, 220),
          ("H 585x220", 585, 220), ("대조 784x220", 784, 220)]
N_XS = (4, 6, 8, 10, 12, 16)
N_YS = (2, 3, 5, 7)
REPEAT = 7
BUDGET_MS = 20.0


def grid(n_x, n_y, w, h):
    """정답 제어점 자리만 필요하다 -- 시간은 좌표값이 아니라 개수에 걸린다."""
    us = np.linspace(0, w - 1, n_x)
    vs = np.linspace(0, h - 1, n_y)
    dst = np.array([[u, v] for v in vs for u in us], dtype=np.float64)
    # src 는 약간 흔들어 둔다. 완전히 같은 점이면 선형계가 퇴화해 비현실적으로 빠를 수 있다.
    src = dst + np.random.default_rng(0).normal(0, 3.0, dst.shape)
    return dst, src


def timed(fn):
    """중앙값을 쓴다 -- OS 지터에 평균보다 덜 흔들린다."""
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(np.min(ts))


def full(src, dst, h, w):
    return lambda: tps_flow(src, dst, (h, w))


def quarter(src, dst, h, w):
    """인계 문서가 '선택이 아니라 필수' 라고 한 1/4 축소. 맵만 줄여 계산하고 늘린다."""
    hq, wq = max(2, h // 4), max(2, w // 4)

    def run():
        mx, my = tps_flow(src, dst / 4.0, (hq, wq))
        cv2.resize(mx, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.resize(my, (w, h), interpolation=cv2.INTER_LINEAR)
    return run


if __name__ == "__main__":
    out = {"repeat": REPEAT, "budget_ms": BUDGET_MS, "threads": 1, "rows": []}
    for name, w, h in SHAPES:
        for n_y in N_YS:
            for n_x in N_XS:
                dst, src = grid(n_x, n_y, w, h)
                f_med, f_min = timed(full(src, dst, h, w))
                q_med, q_min = timed(quarter(src, dst, h, w))
                out["rows"].append({
                    "shape": name, "w": w, "h": h, "n_x": n_x, "n_y": n_y,
                    "n_pts": n_x * n_y,
                    "full_ms_median": f_med, "full_ms_min": f_min,
                    "quarter_ms_median": q_med, "quarter_ms_min": q_min,
                    "quarter_fits_budget": q_med <= BUDGET_MS,
                })
                print(f"{name:14s} {n_x:2d}x{n_y}={n_x*n_y:2d}pts  "
                      f"full {f_med:8.2f} ms   1/4 {q_med:7.2f} ms"
                      f"{'' if q_med <= BUDGET_MS else '   <-- 예산 초과'}", flush=True)
    path = os.path.join(os.path.dirname(__file__), "..", "results", "04_cost.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
