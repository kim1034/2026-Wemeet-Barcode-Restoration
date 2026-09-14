"""05 -- n_x x n_y x sigma 전체 그리드.

01(n_x x sigma, n_y=3 고정)과 02(n_x in {6,8} x n_y in {2,3,5}, sigma in {0,2})가
부분적으로만 채운 격자를 전부 채운다.

n_x in {4,6,8,10,12,16} x n_y in {2,3,5,7} x sigma_px in {0,1,2,4} x 400 표본/버킷
= 96 셀/표본 x 1200 표본 = 115200 행.

00 의 수확 표본(레시피만 저장)을 재구성해서 쓴다. 표본 하나 안에서는 (n_x, n_y)
쌍마다 정답 제어점을 한 번만 뽑고, sigma 마다 결정론적 셀 시드로 섭동을 넣는다
-- rng_cell = default_rng((sample_idx, n_x, n_y, int(round(sigma*10)))). 01/02 와
같은 셀은 같은 시드로 재생산되므로 겹치는 셀은 bit-identical 이어야 한다(본
스크립트가 직접 검증한다).
"""
import json
import multiprocessing as mp
import os
import sys

# 01/02 와 동일한 이유 -- BLAS/OpenCV 스레드풀 재구독 방지. numpy import 전에 건다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import decode, mcnemar, perturb_src, rectify_and_error, wilson  # noqa: E402

from wemeet.data.synthesis import H_OBS, build, recipe_from_dict  # noqa: E402
from wemeet.data.warp import control_points  # noqa: E402

N_X_LIST = [4, 6, 8, 10, 12, 16]
N_Y_LIST = [2, 3, 5, 7]
SIGMA_LIST = [0.0, 1.0, 2.0, 4.0]

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")
HARVEST_PATH = os.path.join(RESULTS_DIR, "00_harvest.json")
NX_SIGMA_PATH = os.path.join(RESULTS_DIR, "01_nx_sigma.json")
NY_PATH = os.path.join(RESULTS_DIR, "02_ny.json")
OUT_PATH = os.path.join(RESULTS_DIR, "05_grid.json")


def _worker_init():
    cv2.setNumThreads(1)


def process_sample(task):
    """표본 하나에 대해 (n_x, n_y, sigma) 96셀 전부를 돈다. 워커에서 실행된다."""
    bucket, sample_idx, recipe_dict = task
    recipe = recipe_from_dict(recipe_dict)
    s = build(recipe)
    out_shape = (H_OBS, s.w_flat)
    rows = []
    for n_x in N_X_LIST:
        for n_y in N_Y_LIST:
            dst_norm, src_norm = control_points(s.g, n_x, n_y, s.obs.shape, s.u_lo, s.u_hi)
            for sigma in SIGMA_LIST:
                rng_cell = np.random.default_rng((sample_idx, n_x, n_y, int(round(sigma * 10))))
                src_p = perturb_src(src_norm, sigma, s.obs.shape, rng_cell)
                fixed, max_err, med_err = rectify_and_error(
                    s.obs, dst_norm, src_p, s.g, s.u_lo, s.u_hi, out_shape)
                ok = decode(fixed) is not None
                rows.append((bucket, sample_idx, n_x, n_y, sigma, bool(ok), max_err, med_err))
    return rows


def compare_overlap(all_rows, nx_sigma_json, ny_json):
    """01/02 와 겹치는 셀을 표본 단위로 정확히 비교한다 (bit-identical 여부)."""
    ours = {(b, i, nx, ny, sg): (ok, me, de) for b, i, nx, ny, sg, ok, me, de in all_rows}

    # 01: n_y=3 고정, sigma in {0,0.5,1,2,4,8,12} -- 겹치는 sigma 는 {0,1,2,4}
    n01_total = n01_match = 0
    mismatches_01 = []
    for b, i, nx, sg, ok, me, de in nx_sigma_json["rows"]:
        if sg not in SIGMA_LIST:
            continue
        key = (b, i, nx, 3, sg)
        if key not in ours:
            continue
        n01_total += 1
        ours_ok, ours_me, ours_de = ours[key]
        if ours_ok == ok and ours_me == me and ours_de == de:
            n01_match += 1
        elif len(mismatches_01) < 10:
            mismatches_01.append({"key": list(key), "theirs": [ok, me, de],
                                   "ours": [ours_ok, ours_me, ours_de]})

    # 02: n_x in {6, nx_best}, n_y in {2,3,5}, sigma in {0,2}
    n02_total = n02_match = 0
    mismatches_02 = []
    for b, i, nx, ny, sg, ok, me, de in ny_json["rows"]:
        key = (b, i, nx, ny, sg)
        if key not in ours:
            continue
        n02_total += 1
        ours_ok, ours_me, ours_de = ours[key]
        if ours_ok == ok and ours_me == me and ours_de == de:
            n02_match += 1
        elif len(mismatches_02) < 10:
            mismatches_02.append({"key": list(key), "theirs": [ok, me, de],
                                   "ours": [ours_ok, ours_me, ours_de]})

    return {
        "vs_01_nx_sigma": {"n_compared": n01_total, "n_match": n01_match,
                            "all_match": n01_total > 0 and n01_total == n01_match,
                            "mismatches_sample": mismatches_01},
        "vs_02_ny": {"n_compared": n02_total, "n_match": n02_match,
                     "all_match": n02_total > 0 and n02_total == n02_match,
                     "mismatches_sample": mismatches_02},
    }


def aggregate(all_rows, buckets):
    by_cell = {}       # (bucket, n_x, n_y, sigma) -> [ok...]
    by_cell_err = {}   # (bucket, n_x, n_y, sigma) -> [max_err...]
    by_cell_sample_ok = {}  # (bucket, n_x, n_y, sigma) -> {sample_idx: ok}  (맥니마용)

    for bucket, idx, n_x, n_y, sigma, ok, max_err, med_err in all_rows:
        key = (bucket, n_x, n_y, sigma)
        by_cell.setdefault(key, []).append(ok)
        by_cell_err.setdefault(key, []).append(max_err)
        by_cell_sample_ok.setdefault(key, {})[idx] = ok

    decode_rate = {}
    for key, oks in by_cell.items():
        k, n = sum(oks), len(oks)
        p, lo, hi = wilson(k, n)
        decode_rate[key] = {"k": k, "n": n, "p": p, "ci_lo": lo, "ci_hi": hi}

    err_stats = {}
    for key, errs in by_cell_err.items():
        err_stats[key] = {"mean_max_err": float(np.mean(errs)), "median_max_err": float(np.median(errs))}

    # 버킷 균등가중 평균 (버킷마다 n=400 로 동일하므로 rate 의 단순평균 = 표본 단순평균)
    bucket_avg_rate = {}
    for n_x in N_X_LIST:
        for n_y in N_Y_LIST:
            for sigma in SIGMA_LIST:
                ps = [decode_rate[(b, n_x, n_y, sigma)]["p"] for b in buckets
                      if (b, n_x, n_y, sigma) in decode_rate]
                if ps:
                    bucket_avg_rate[(n_x, n_y, sigma)] = float(np.mean(ps))

    # 맥니마: 지정된 비교쌍, sigma in {0,2}, 버킷별 + 전체 풀링(버킷 균등, n 동일이라
    # 풀링이 곧 버킷 균등가중과 같다)
    compare_pairs = [
        ((8, 5), (12, 5)),
        ((8, 5), (16, 5)),
        ((8, 5), (10, 5)),
        ((8, 5), (8, 7)),
        ((12, 5), (12, 7)),
        ((16, 5), (16, 7)),
    ]
    mcnemar_res = {}
    for (nxa, nya), (nxb, nyb) in compare_pairs:
        for sigma in (0.0, 2.0):
            pooled_b = pooled_c = pooled_n = 0
            per_bucket = {}
            for bucket in buckets:
                ok_a = by_cell_sample_ok.get((bucket, nxa, nya, sigma), {})
                ok_b = by_cell_sample_ok.get((bucket, nxb, nyb, sigma), {})
                common_idx = set(ok_a) & set(ok_b)
                b_cnt = sum(1 for i in common_idx if ok_a[i] and not ok_b[i])
                c_cnt = sum(1 for i in common_idx if ok_b[i] and not ok_a[i])
                per_bucket[bucket] = {"n": len(common_idx), "b_a_only": b_cnt,
                                      "c_b_only": c_cnt, "p": mcnemar(b_cnt, c_cnt)}
                pooled_b += b_cnt
                pooled_c += c_cnt
                pooled_n += len(common_idx)
            key = f"{nxa}x{nya}_vs_{nxb}x{nyb}|sigma={sigma}"
            mcnemar_res[key] = {
                "pooled": {"n": pooled_n, "b_a_only": pooled_b, "c_b_only": pooled_c,
                           "p": mcnemar(pooled_b, pooled_c)},
                "per_bucket": per_bucket,
            }

    def keystr(k):
        return "|".join(str(x) for x in k)

    return {
        "decode_rate": {keystr(k): v for k, v in decode_rate.items()},
        "err_stats": {keystr(k): v for k, v in err_stats.items()},
        "bucket_avg_decode_rate": {keystr(k): v for k, v in bucket_avg_rate.items()},
        "mcnemar": mcnemar_res,
    }


def main():
    with open(HARVEST_PATH, encoding="utf-8") as fh:
        harvest = json.load(fh)

    tasks = []
    for bucket in sorted(harvest["buckets"]):
        samples = harvest["buckets"][bucket]["samples"]
        for idx, samp in enumerate(samples):
            tasks.append((bucket, idx, samp["recipe"]))
    cells_per_sample = len(N_X_LIST) * len(N_Y_LIST) * len(SIGMA_LIST)
    print(f"total samples (tasks) = {len(tasks)}, cells/sample = {cells_per_sample}, "
          f"total rows = {len(tasks) * cells_per_sample}", flush=True)

    all_rows = []
    done = 0
    with mp.Pool(processes=min(16, mp.cpu_count()), initializer=_worker_init) as pool:
        for rows in pool.imap_unordered(process_sample, tasks, chunksize=1):
            all_rows.extend(rows)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"progress {done}/{len(tasks)} samples", flush=True)

    print("aggregating...", flush=True)
    buckets = sorted(harvest["buckets"])
    agg = aggregate(all_rows, buckets)

    print("comparing against 01_nx_sigma.json / 02_ny.json...", flush=True)
    with open(NX_SIGMA_PATH, encoding="utf-8") as fh:
        nx_sigma_json = json.load(fh)
    with open(NY_PATH, encoding="utf-8") as fh:
        ny_json = json.load(fh)
    compare = compare_overlap(all_rows, nx_sigma_json, ny_json)
    print(json.dumps({
        "vs_01": {k: v for k, v in compare["vs_01_nx_sigma"].items() if k != "mismatches_sample"},
        "vs_02": {k: v for k, v in compare["vs_02_ny"].items() if k != "mismatches_sample"},
    }, indent=2), flush=True)

    out = {
        "n_x_list": N_X_LIST,
        "n_y_list": N_Y_LIST,
        "sigma_list": SIGMA_LIST,
        "rows": all_rows,  # (bucket, sample_idx, n_x, n_y, sigma, ok, max_mag_err, median_mag_err)
        "aggregate": agg,
        "overlap_sanity_check": compare,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
