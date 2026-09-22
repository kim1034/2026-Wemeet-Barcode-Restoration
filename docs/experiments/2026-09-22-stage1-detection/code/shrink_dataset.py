#!/usr/bin/env python3
"""
데이터셋 해상도 축소 (업로드 용량·CPU 부담 줄이기)

1920x1080 → 1280x720 (기본). 가로세로를 같은 비율로 줄이므로
  - YOLO 라벨(0~1 정규화 좌표)은 그대로 써도 되고
  - 막대 영역 박스의 5:3 / 6:4 비율도 그대로 유지된다.
학습 시 imgsz=1280 이면 모델이 보는 화질은 원본과 동일하다 (어차피 긴 변을 1280으로 줄여 입력하므로).

사용법 (내 PC에서 실행 → 결과 폴더만 업로드)
  python shrink_dataset.py --src dataset --dst dataset_720
  python shrink_dataset.py --src dataset --dst dataset_720 --width 1280 --quality 90 --workers 8
"""

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


def resize_one(job):
    src, dst, width, height, quality = job
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return 0
    if (img.shape[1], img.shape[0]) != (width, height):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return dst.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--dst", default="dataset_720")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    jobs, n_lbl = [], 0
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in sorted((src / "images" / split).glob("*.jpg")):
            jobs.append((p, dst / "images" / split / p.name, args.width, args.height, args.quality))
        for p in sorted((src / "labels" / split).glob("*.txt")):  # 라벨은 정규화 좌표라 그대로 복사
            shutil.copy(p, dst / "labels" / split / p.name)
            n_lbl += 1
    print(f"이미지 {len(jobs):,}장 · 라벨 {n_lbl:,}개 → {args.width}x{args.height}")

    t0, total = time.perf_counter(), 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, size in enumerate(ex.map(resize_one, jobs, chunksize=16), 1):
            total += size
            if i % 1000 == 0:
                el = time.perf_counter() - t0
                print(f"  {i:,}/{len(jobs):,} · {el / 60:.1f}분 · 남은 시간 약 {el / i * (len(jobs) - i) / 60:.1f}분",
                      flush=True)

    # data.yaml 은 경로만 새 폴더로 바꿔 복사 (학습 스크립트가 다시 절대경로로 고쳐줌)
    yml = src / "data.yaml"
    if yml.exists():
        lines = [f"path: {dst.resolve().as_posix()}" if l.strip().startswith("path:") else l
                 for l in yml.read_text(encoding="utf-8").splitlines()]
        (dst / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for meta in src.glob("meta_*.csv"):
        shutil.copy(meta, dst / meta.name)

    before = sum(p.stat().st_size for p in (src / "images").rglob("*.jpg"))
    print(f"\n✅ 완료 · {(time.perf_counter() - t0) / 60:.1f}분")
    print(f"   이미지 용량 {before / 1e9:.2f} GB → {total / 1e9:.2f} GB ({total / max(before, 1):.0%})")
    print(f"   업로드할 폴더: {dst.resolve()}")


if __name__ == "__main__":
    main()
