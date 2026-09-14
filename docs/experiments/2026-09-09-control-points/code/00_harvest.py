"""00 -- n_x 와 무관하게 고정한 표본 400개/버킷을 수확한다.

표본 IN 규칙 (n_x 를 전혀 안 씀):
    decode(obs) is None  AND  decode(remap_G(obs, g, u_lo, u_hi, (H_OBS, w_flat))) is not None

버킷당 rng 는 루프 밖에서 1개만 만든다 -- 스파이크 09/10 은 루프 안에서 재시딩해서
무효화됐다. 여기서도 절대 안 한다.

부수로, 첫 디코딩이 실패한 모든 표본에 대해 기존 6x3 rectify 도 디코딩되는지
같이 기록한다. G remap 과 기존 제어점 정의가 완전히 포개지지 않는다는 것을
정직하게 남기기 위해서다 (설계 문서 참고).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import decode, lam_eff, remap_G, ridge_curvature  # noqa: E402

from scripts.label_recipes import rectify  # noqa: E402
from wemeet.data.synthesis import BUCKETS, H_OBS, build, draw_recipe, recipe_to_dict  # noqa: E402

SEED = 2026
N_KEEP = 400
RENDER_CAP = 20000

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "00_harvest.json")


def harvest_bucket(bucket: str) -> dict:
    rng = np.random.default_rng(SEED)  # 루프 밖, 버킷당 1개
    samples = []
    n_first_ok = 0
    n_fail_g_ok = 0
    n_fail_g_fail = 0
    n_fail_g_fail_but_old_ok = 0
    renders = 0

    while len(samples) < N_KEEP and renders < RENDER_CAP:
        recipe = draw_recipe(rng, bucket, renders)
        sample = build(recipe)
        renders += 1

        if decode(sample.obs) is not None:
            n_first_ok += 1
            if renders % 500 == 0:
                print(f"[{bucket}] renders={renders} kept={len(samples)}", flush=True)
            continue

        out_shape = (H_OBS, sample.w_flat)
        fixed_g = remap_G(sample.obs, sample.g, sample.u_lo, sample.u_hi, out_shape)
        g_ok = decode(fixed_g) is not None

        old_fixed = rectify(sample.obs, sample.dst_norm, sample.src_norm, out_shape)
        old_ok = decode(old_fixed) is not None

        if not g_ok:
            n_fail_g_fail += 1
            if old_ok:
                n_fail_g_fail_but_old_ok += 1
        else:
            n_fail_g_ok += 1
            lam, amp = lam_eff(sample.g, sample.u_lo, sample.u_hi)
            curv = ridge_curvature(sample.g, sample.u_lo, sample.u_hi)
            samples.append({
                "recipe": recipe_to_dict(recipe),
                "bucket": bucket,
                "m_min": sample.m_min,
                "sat_ratio": sample.sat_ratio,
                "w_flat": sample.w_flat,
                "obs_shape": list(sample.obs.shape),
                "lam_eff": lam,
                "lam_amp": amp,
                "ridge_curvature": curv,
                "old_rectify_ok": old_ok,
            })

        if renders % 500 == 0:
            print(f"[{bucket}] renders={renders} kept={len(samples)}", flush=True)

    return {
        "bucket": bucket,
        "seed": SEED,
        "renders": renders,
        "hit_cap": renders >= RENDER_CAP and len(samples) < N_KEEP,
        "n_first_ok": n_first_ok,
        "n_fail_g_ok": n_fail_g_ok,
        "n_fail_g_fail": n_fail_g_fail,
        "n_fail_g_fail_but_old_ok": n_fail_g_fail_but_old_ok,
        "n_kept": len(samples),
        "samples": samples,
    }


def main():
    result = {"seed": SEED, "n_keep_target": N_KEEP, "render_cap": RENDER_CAP, "buckets": {}}
    for bucket in sorted(BUCKETS):
        print(f"=== bucket {bucket} ===", flush=True)
        result["buckets"][bucket] = harvest_bucket(bucket)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)

    print("\n=== summary ===")
    for bucket, bd in result["buckets"].items():
        print(f"{bucket}: renders={bd['renders']} hit_cap={bd['hit_cap']} "
              f"first_ok={bd['n_first_ok']} fail_g_ok={bd['n_fail_g_ok']} "
              f"fail_g_fail={bd['n_fail_g_fail']} kept={bd['n_kept']} "
              f"mismatch(old_rectify_ok_but_G_fail)={bd['n_fail_g_fail_but_old_ok']}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
