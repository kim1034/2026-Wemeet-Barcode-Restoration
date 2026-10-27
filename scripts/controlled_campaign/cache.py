"""Populate the exact existing render cache on CPU; no training or changed recipes."""

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.controlled_geometry.common import configure, digest, validate_lock, write_json
from scripts.controlled_geometry.data import RenderCache, load_split

_CACHE = None


def initialize(root, fingerprint):
    global _CACHE
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configure(42, threads=1)
    _CACHE = RenderCache(root, fingerprint)


def render(record):
    _CACHE.get(record)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lock = validate_lock(args.protocol)
    if args.output.exists():
        raise FileExistsError(args.output)
    records = load_split(lock["split_manifest"])["train"]
    started = time.monotonic()
    with ProcessPoolExecutor(
        args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(lock["cache_root"], lock["renderer_fingerprint"]),
    ) as pool:
        for count, _ in enumerate(pool.map(render, records, chunksize=32), 1):
            if count % 1000 == 0:
                print(
                    f"cache {count}/{len(records)}; {time.monotonic() - started:.1f}s", flush=True
                )
    write_json(
        args.output,
        {
            "status": "passed",
            "train_samples": len(records),
            "elapsed_seconds": time.monotonic() - started,
            "workers": args.workers,
            "protocol_sha256": digest(args.protocol),
            "gpu_used": False,
            "cache_root": lock["cache_root"],
        },
    )
    print(f"cache complete: {len(records)}", flush=True)


if __name__ == "__main__":
    main()
