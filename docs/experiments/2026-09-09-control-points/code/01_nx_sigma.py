"""01 -- n_x x sigma 실험. 00 의 수확 표본을 재구성해서 그리드 전체를 돈다.

n_x in {4,6,8,10,12,16} (n_y=3 고정) x sigma_px in {0,.5,1,2,4,8,12} x 400 표본/버킷.

레시피만 저장돼 있으므로 표본(이미지 + G)은 여기서 다시 build() 한다. 표본 하나
안에서는 n_x 마다 정답 제어점을 한 번만 뽑고, sigma 마다 결정론적 셀 시드로
섭동을 넣는다 -- rng_cell = default_rng((sample_idx, n_x, int(sigma*10))).

멀티프로세싱: 표본 단위(레시피 dict, 작다)로 워커에 분배하고, 워커 안에서
표본을 재조립해 42셀을 전부 처리한다. 표본 자체(이미지, G)를 pickle 해서
넘기지 않는다 -- 그러면 IPC 비용이 커진다.
"""
import json
import multiprocessing as mp
import os
import sys

# BLAS/OpenCV 는 프로세스당 기본으로 코어 전부를 쓰려 든다. 워커 16개 x 스레드
# 16개가 겹치면 극심하게 재구독돼 벽시계 시간이 5~10배 느려진다 (실측). numpy
# import 전에 반드시 걸어야 한다 -- spawn 워커는 이 파일을 처음부터 다시 읽는다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import decode, mcnemar, perturb_src, rectify_and_error, wilson  # noqa: E402

from wemeet.data.synthesis import H_OBS, build, recipe_from_dict  # noqa: E402
from wemeet.data.warp import control_points  # noqa: E402

N_X_LIST = [4, 6, 8, 10, 12, 16]
SIGMA_LIST = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
N_Y_FIXED = 3

HERE = os.path.dirname(os.path.abspath(__file__))
HARVEST_PATH = os.path.join(os.path.dirname(HERE), "results", "00_harvest.json")
OUT_PATH = os.path.join(os.path.dirname(HERE), "results", "01_nx_sigma.json")


def _worker_init():
    """워커마다 cv2 내부 스레드풀도 1로 -- 환경변수만으론 못 막는 부분(Concurrency)."""
    cv2.setNumThreads(1)


def process_sample(task):
    """표본 하나에 대해 (n_x, sigma) 42셀 전부를 돈다. 워커에서 실행된다."""
    bucket, sample_idx, recipe_dict = task
    recipe = recipe_from_dict(recipe_dict)
    s = build(recipe)
    out_shape = (H_OBS, s.w_flat)
    rows = []
    for n_x in N_X_LIST:
        dst_norm, src_norm = control_points(s.g, n_x, N_Y_FIXED, s.obs.shape, s.u_lo, s.u_hi)
        for sigma in SIGMA_LIST:
            rng_cell = np.random.default_rng((sample_idx, n_x, int(round(sigma * 10))))
            src_p = perturb_src(src_norm, sigma, s.obs.shape, rng_cell)
            fixed, max_err, med_err = rectify_and_error(
                s.obs, dst_norm, src_p, s.g, s.u_lo, s.u_hi, out_shape)
            ok = decode(fixed) is not None
            rows.append((bucket, sample_idx, n_x, sigma, bool(ok), max_err, med_err))
    return rows


def aggregate(all_rows, harvest):
    """all_rows: (bucket, sample_idx, n_x, sigma, ok, max_err, med_err) 리스트."""
    buckets = sorted(harvest["buckets"])

    # 표본별 lam_eff -- 버킷 내 5분위수 경계에 쓴다
    lam_of = {}
    for b in buckets:
        for i, samp in enumerate(harvest["buckets"][b]["samples"]):
            lam_of[(b, i)] = samp["lam_eff"]

    by_cell = {}  # (bucket, n_x, sigma) -> list of ok
    by_cell_err = {}  # (bucket, n_x, sigma) -> list of (max_err, med_err)
    by_cell_sample_ok = {}  # (bucket, n_x, sigma) -> {sample_idx: ok}  (mcnemar 용)
    by_quintile = {}  # (bucket, quintile, n_x, sigma) -> list of ok

    for bucket, idx, n_x, sigma, ok, max_err, med_err in all_rows:
        key = (bucket, n_x, sigma)
        by_cell.setdefault(key, []).append(ok)
        by_cell_err.setdefault(key, []).append((max_err, med_err))
        by_cell_sample_ok.setdefault(key, {})[idx] = ok

    # 버킷별 lam_eff 5분위 경계
    quintile_edges = {}
    quintile_of = {}
    for b in buckets:
        lams = sorted(lam_of[(b, i)] for i in range(len(harvest["buckets"][b]["samples"])))
        edges = [np.percentile(lams, p) for p in (20, 40, 60, 80)]
        quintile_edges[b] = edges
        for i in range(len(harvest["buckets"][b]["samples"])):
            lam = lam_of[(b, i)]
            q = sum(lam > e for e in edges)  # 0..4
            quintile_of[(b, i)] = q

    for bucket, idx, n_x, sigma, ok, max_err, med_err in all_rows:
        q = quintile_of[(bucket, idx)]
        by_quintile.setdefault((bucket, q, n_x, sigma), []).append(ok)

    decode_rate = {}
    for key, oks in by_cell.items():
        k, n = sum(oks), len(oks)
        p, lo, hi = wilson(k, n)
        decode_rate[key] = {"k": k, "n": n, "p": p, "ci_lo": lo, "ci_hi": hi}

    err_stats = {}
    for key, pairs in by_cell_err.items():
        max_errs = [p[0] for p in pairs]
        med_errs = [p[1] for p in pairs]
        err_stats[key] = {
            "mean_max_err": float(np.mean(max_errs)),
            "median_max_err": float(np.median(max_errs)),
            "mean_med_err": float(np.mean(med_errs)),
            "median_med_err": float(np.median(med_errs)),
        }

    quintile_rate = {}
    for key, oks in by_quintile.items():
        k, n = sum(oks), len(oks)
        p, lo, hi = wilson(k, n)
        quintile_rate[key] = {"k": k, "n": n, "p": p, "ci_lo": lo, "ci_hi": hi}

    # 맥니마 검정: n_x 인접쌍, sigma in {0, 2}, 버킷별
    mcnemar_res = {}
    pairs_nx = list(zip(N_X_LIST[:-1], N_X_LIST[1:]))
    for bucket in buckets:
        for sigma in (0.0, 2.0):
            for nx_a, nx_b in pairs_nx:
                ok_a = by_cell_sample_ok.get((bucket, nx_a, sigma), {})
                ok_b = by_cell_sample_ok.get((bucket, nx_b, sigma), {})
                common_idx = set(ok_a) & set(ok_b)
                b_cnt = sum(1 for i in common_idx if ok_a[i] and not ok_b[i])
                c_cnt = sum(1 for i in common_idx if ok_b[i] and not ok_a[i])
                p_val = mcnemar(b_cnt, c_cnt)
                mcnemar_res[(bucket, sigma, nx_a, nx_b)] = {
                    "n": len(common_idx), "b_a_only": b_cnt, "c_b_only": c_cnt, "p": p_val,
                }

    def keystr(k):
        return "|".join(str(x) for x in k)

    return {
        "quintile_edges_lam": quintile_edges,
        "decode_rate": {keystr(k): v for k, v in decode_rate.items()},
        "err_stats": {keystr(k): v for k, v in err_stats.items()},
        "quintile_rate": {keystr(k): v for k, v in quintile_rate.items()},
        "mcnemar": {keystr(k): v for k, v in mcnemar_res.items()},
    }


def main():
    with open(HARVEST_PATH, encoding="utf-8") as fh:
        harvest = json.load(fh)

    tasks = []
    for bucket in sorted(harvest["buckets"]):
        samples = harvest["buckets"][bucket]["samples"]
        for idx, samp in enumerate(samples):
            tasks.append((bucket, idx, samp["recipe"]))
    print(f"total samples (tasks) = {len(tasks)}, cells/sample = "
          f"{len(N_X_LIST) * len(SIGMA_LIST)}, total rows = "
          f"{len(tasks) * len(N_X_LIST) * len(SIGMA_LIST)}", flush=True)

    all_rows = []
    done = 0
    with mp.Pool(processes=min(16, mp.cpu_count()), initializer=_worker_init) as pool:
        for rows in pool.imap_unordered(process_sample, tasks, chunksize=1):
            all_rows.extend(rows)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"progress {done}/{len(tasks)} samples", flush=True)

    print("aggregating...", flush=True)
    agg = aggregate(all_rows, harvest)

    out = {
        "n_x_list": N_X_LIST,
        "sigma_list": SIGMA_LIST,
        "n_y_fixed": N_Y_FIXED,
        "rows": all_rows,  # (bucket, sample_idx, n_x, sigma, ok, max_err, med_err)
        "aggregate": agg,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
