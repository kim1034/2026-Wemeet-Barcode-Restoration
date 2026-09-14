"""02 -- n_y 실험.

표본 규칙은 00 과 다르다: "1차 디코딩이 실패한 표본" 이면 끝 (G remap 성공 여부는
안 본다). 00 이 수확한 400개/버킷은 이미 전부 1차 디코딩 실패 표본이므로 (IN
규칙의 앞부분이 그것이다) 그대로 재사용한다 -- 별도로 더 뽑지 않는다.

n_y in {2,3,5}, n_x in {6, NX_BEST} (NX_BEST 는 실험 (1) 의 결과로 정한다.
모호하면 8 을 쓰라고 지시받았다 -- 그 경우 --nx-best 8 로 명시해서 부른다),
sigma_px in {0, 2}.

능선 곡률(ridge_curvature) 삼분위로 층화한다: n_y 는 능선이 실제로 굽이칠 때만
의미가 있다는 가설이다.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

# 01 과 동일한 이유 -- BLAS/OpenCV 스레드풀 재구독 방지. numpy import 전에 건다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import decode, perturb_src, rectify_and_error, wilson  # noqa: E402

from wemeet.data.synthesis import H_OBS, build, recipe_from_dict  # noqa: E402
from wemeet.data.warp import control_points  # noqa: E402

N_Y_LIST = [2, 3, 5]
SIGMA_LIST = [0.0, 2.0]

HERE = os.path.dirname(os.path.abspath(__file__))
HARVEST_PATH = os.path.join(os.path.dirname(HERE), "results", "00_harvest.json")
OUT_PATH = os.path.join(os.path.dirname(HERE), "results", "02_ny.json")


def _worker_init():
    cv2.setNumThreads(1)


def process_sample(task):
    bucket, sample_idx, recipe_dict, n_x_list = task
    recipe = recipe_from_dict(recipe_dict)
    s = build(recipe)
    out_shape = (H_OBS, s.w_flat)
    rows = []
    for n_x in n_x_list:
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


def aggregate(all_rows, harvest, n_x_list):
    buckets = sorted(harvest["buckets"])

    curv_of = {}
    dm0_of = {}
    preset_of = {}
    for b in buckets:
        for i, samp in enumerate(harvest["buckets"][b]["samples"]):
            curv_of[(b, i)] = samp["ridge_curvature"]
            dm0_of[(b, i)] = samp["recipe"]["d_m0"]
            preset_of[(b, i)] = samp["recipe"]["preset"]

    # 버킷별 능선 곡률 삼분위 경계
    tertile_edges = {}
    tertile_of = {}
    for b in buckets:
        curvs = sorted(curv_of[(b, i)] for i in range(len(harvest["buckets"][b]["samples"])))
        edges = [np.percentile(curvs, p) for p in (100 / 3, 200 / 3)]
        tertile_edges[b] = edges
        for i in range(len(harvest["buckets"][b]["samples"])):
            c = curv_of[(b, i)]
            t = sum(c > e for e in edges)  # 0,1,2
            tertile_of[(b, i)] = t

    by_cell = {}  # (bucket, n_x, n_y, sigma) -> oks
    by_cell_tertile = {}  # (bucket, tertile, n_x, n_y, sigma) -> oks
    by_cell_err = {}

    for bucket, idx, n_x, n_y, sigma, ok, max_err, med_err in all_rows:
        key = (bucket, n_x, n_y, sigma)
        by_cell.setdefault(key, []).append(ok)
        by_cell_err.setdefault(key, []).append(max_err)
        t = tertile_of[(bucket, idx)]
        by_cell_tertile.setdefault((bucket, t, n_x, n_y, sigma), []).append(ok)

    decode_rate = {}
    for key, oks in by_cell.items():
        k, n = sum(oks), len(oks)
        p, lo, hi = wilson(k, n)
        decode_rate[key] = {"k": k, "n": n, "p": p, "ci_lo": lo, "ci_hi": hi}

    tertile_rate = {}
    for key, oks in by_cell_tertile.items():
        k, n = sum(oks), len(oks)
        p, lo, hi = wilson(k, n)
        tertile_rate[key] = {"k": k, "n": n, "p": p, "ci_lo": lo, "ci_hi": hi}

    err_stats = {}
    for key, errs in by_cell_err.items():
        err_stats[key] = {"mean_max_err": float(np.mean(errs)), "median_max_err": float(np.median(errs))}

    # ridge_curvature 분포: 버킷별, 프리셋별 (px, d_m0 단위)
    curv_dist_bucket = {}
    for b in buckets:
        curvs = np.array([curv_of[(b, i)] for i in range(len(harvest["buckets"][b]["samples"]))])
        dm0s = np.array([dm0_of[(b, i)] for i in range(len(harvest["buckets"][b]["samples"]))])
        ratios = curvs / dm0s
        curv_dist_bucket[b] = {
            "n": len(curvs),
            "px": {"mean": float(curvs.mean()), "median": float(np.median(curvs)),
                   "p25": float(np.percentile(curvs, 25)), "p75": float(np.percentile(curvs, 75))},
            "d_m0_units": {"mean": float(ratios.mean()), "median": float(np.median(ratios)),
                           "p25": float(np.percentile(ratios, 25)), "p75": float(np.percentile(ratios, 75))},
        }

    curv_dist_preset = {}
    for b in buckets:
        n_samples = len(harvest["buckets"][b]["samples"])
        presets = sorted(set(preset_of[(b, i)] for i in range(n_samples)))
        for preset in presets:
            idxs = [i for i in range(n_samples) if preset_of[(b, i)] == preset]
            curvs = np.array([curv_of[(b, i)] for i in idxs])
            dm0s = np.array([dm0_of[(b, i)] for i in idxs])
            ratios = curvs / dm0s
            curv_dist_preset[(b, preset)] = {
                "n": len(curvs),
                "px": {"mean": float(curvs.mean()), "median": float(np.median(curvs))},
                "d_m0_units": {"mean": float(ratios.mean()), "median": float(np.median(ratios))},
            }

    def keystr(k):
        return "|".join(str(x) for x in k)

    return {
        "tertile_edges_ridge_curvature_px": tertile_edges,
        "decode_rate": {keystr(k): v for k, v in decode_rate.items()},
        "tertile_rate": {keystr(k): v for k, v in tertile_rate.items()},
        "err_stats": {keystr(k): v for k, v in err_stats.items()},
        "ridge_curvature_dist_by_bucket": curv_dist_bucket,
        "ridge_curvature_dist_by_preset": {keystr(k): v for k, v in curv_dist_preset.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx-best", type=int, default=8,
                    help="실험(1) 이 가리키는 최적 n_x. 모호하면 8(기본값)을 쓰라고 지시받음")
    args = ap.parse_args()

    n_x_list = sorted(set([6, args.nx_best]))
    print(f"n_x_list = {n_x_list}", flush=True)

    with open(HARVEST_PATH, encoding="utf-8") as fh:
        harvest = json.load(fh)

    tasks = []
    for bucket in sorted(harvest["buckets"]):
        samples = harvest["buckets"][bucket]["samples"]
        for idx, samp in enumerate(samples):
            tasks.append((bucket, idx, samp["recipe"], n_x_list))
    print(f"total samples (tasks) = {len(tasks)}, cells/sample = "
          f"{len(n_x_list) * len(N_Y_LIST) * len(SIGMA_LIST)}", flush=True)

    all_rows = []
    done = 0
    with mp.Pool(processes=min(16, mp.cpu_count()), initializer=_worker_init) as pool:
        for rows in pool.imap_unordered(process_sample, tasks, chunksize=1):
            all_rows.extend(rows)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"progress {done}/{len(tasks)} samples", flush=True)

    print("aggregating...", flush=True)
    agg = aggregate(all_rows, harvest, n_x_list)

    out = {
        "n_x_list": n_x_list,
        "n_y_list": N_Y_LIST,
        "sigma_list": SIGMA_LIST,
        "rows": all_rows,  # (bucket, sample_idx, n_x, n_y, sigma, ok, max_err, med_err)
        "aggregate": agg,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
