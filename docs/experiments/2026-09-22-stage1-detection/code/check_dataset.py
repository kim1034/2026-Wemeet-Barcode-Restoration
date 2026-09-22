#!/usr/bin/env python3
"""
생성된 데이터셋 검사 (학습 전에 1번 실행)
  - 이미지/라벨 짝, 개수, 클래스 균형
  - 라벨이 이미지당 정확히 1줄인지
  - 박스 비율이 5:3 / 6:4 인지, 10% 패딩 박스가 이미지 안인지
  - 랜덤 16장에 박스를 그려 check_grid.jpg 로 저장 (눈으로 확인)

사용법:  python check_dataset.py --data dataset
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RATIO = {0: 5 / 3, 1: 3 / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    args = ap.parse_args()
    root = Path(args.data)

    errors, count, classes, pairs = [], Counter(), Counter(), []
    for split in ("train", "val"):
        imgs = {p.stem for p in (root / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        for s in sorted(imgs ^ lbls):
            errors.append(f"{split}/{s}: 이미지-라벨 짝 없음")
        for stem in sorted(imgs & lbls):
            lines = [l for l in (root / "labels" / split / f"{stem}.txt").read_text().splitlines() if l.strip()]
            count[split] += 1
            if len(lines) != 1:
                errors.append(f"{split}/{stem}: 라벨 {len(lines)}줄 (1줄이어야 함)")
                continue
            v = lines[0].split()
            cls, pts = int(v[0]), np.array(v[1:], float).reshape(4, 2) * [1920, 1080]
            classes[(split, cls)] += 1
            w, h = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[3] - pts[0])
            if abs(w / h - RATIO[cls]) / RATIO[cls] > 1e-3:
                errors.append(f"{split}/{stem}: 비율 {w / h:.4f}")
            c = pts.mean(0)
            pad = (pts - c) * 1.2 + c
            if (pad < 0).any() or (pad[:, 0] > 1920).any() or (pad[:, 1] > 1080).any():
                errors.append(f"{split}/{stem}: 패딩 박스가 이미지 밖")
            pairs.append((split, stem, pts, pad))

    print(f"이미지: train {count['train']:,} / val {count['val']:,}")
    print("클래스(split, 0=5:3, 1=6:4):", dict(sorted(classes.items())))
    print("✅ 문제 없음" if not errors else f"❌ 문제 {len(errors)}건\n  " + "\n  ".join(errors[:20]))

    cells = []
    for split, stem, pts, pad in random.sample(pairs, min(16, len(pairs))):
        img = cv2.imread(str(root / "images" / split / f"{stem}.jpg"))
        cv2.polylines(img, [pts.astype(np.int32)], True, (255, 255, 0), 4, cv2.LINE_AA)
        cv2.polylines(img, [pad.astype(np.int32)], True, (0, 220, 255), 3, cv2.LINE_AA)
        cell = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(cell, f"{split}/{stem}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cells.append(cell)
    cells += [np.zeros_like(cells[0])] * (-len(cells) % 4)
    grid = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)])
    cv2.imwrite(str(root / "check_grid.jpg"), grid)
    print(f"💾 {root / 'check_grid.jpg'} (하늘색 = 막대 영역 박스, 노랑 = 10% 패딩)")


if __name__ == "__main__":
    main()
