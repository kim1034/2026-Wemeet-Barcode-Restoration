#!/usr/bin/env python3
"""합성 val 에서 검출 CLAHE on/off 의 mAP·P·R 을 비교한다.

val 이미지를 `enhance_for_detection`(배포 코드 그대로)으로 바꾼 사본을 만들고,
원본 val 과 사본 val 에 같은 가중치로 `model.val()` 을 돌린다. 라벨은 그대로 복사한다.

사용법::

    python clahe_val_map.py --val-images <dataset_720>/images/val \
        --weights downloads/detection/v2-yolo-obb/best.pt \
        --work-dir downloads/clahe-val --imgsz 1280

val-images 옆에 `../../labels/val` 이 있는 YOLO 구조를 가정한다. 사본은 work-dir 에
만들고, 이미 다 만들어져 있으면 다시 만들지 않는다.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from wemeet.ai.detection.enhance import enhance_for_detection


def imread(path: Path) -> np.ndarray | None:
    # cv2.imread 는 Windows 에서 비ASCII 경로를 못 읽는다
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite(path: Path, image: np.ndarray) -> None:
    # 재압축 손실이 결과를 흐리지 않게 품질을 높게 둔다
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    assert ok
    buf.tofile(str(path))


def write_yaml(root: Path) -> Path:
    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root.as_posix()}\ntrain: images/val\nval: images/val\n"
        "names:\n  0: barcode_5x3\n  1: barcode_6x4\n",
        encoding="utf-8",
    )
    return yaml


def mirror(val_images: Path, root: Path, enhance: bool) -> Path:
    """val 을 work-dir 로 옮겨 담는다. enhance=True 면 CLAHE 를 건 사본이다.

    원본도 같은 jpg 품질로 다시 저장한다 — 사본만 재압축하면 CLAHE 효과와
    재압축 효과가 섞인다.
    """
    val_labels = val_images.parent.parent / "labels" / val_images.name
    out_img = root / "images" / "val"
    out_lbl = root / "labels" / "val"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in val_images.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if len(list(out_img.glob("*.jpg"))) < len(sources):
        for path in sources:
            image = imread(path)
            imwrite(out_img / f"{path.stem}.jpg", enhance_for_detection(image) if enhance else image)
            label = val_labels / f"{path.stem}.txt"
            if label.exists():
                shutil.copy(label, out_lbl / label.name)
    return write_yaml(root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1280])
    args = ap.parse_args()

    from ultralytics import YOLO

    yamls = {
        "off": mirror(args.val_images, args.work_dir / "off", enhance=False),
        "on": mirror(args.val_images, args.work_dir / "on", enhance=True),
    }
    model = YOLO(str(args.weights))

    for imgsz in args.imgsz:
        rows = {}
        for tag, yaml in yamls.items():
            # v2 는 single_cls 로 학습했다 — 규격은 박스 모양으로 판정한다 (geometry.py)
            box = model.val(
                data=str(yaml), imgsz=imgsz, single_cls=True, batch=2, workers=0,
                device=0, half=True, plots=False, verbose=False,
            ).box
            rows[tag] = (box.map50, box.map, box.mp, box.mr)
        print(f"\nimgsz={imgsz}          mAP50   mAP50-95  P       R")
        for tag, (m50, m, p, r) in rows.items():
            print(f"  CLAHE {tag:<4}       {m50:.4f}  {m:.4f}    {p:.4f}  {r:.4f}")
        d = [on - off for on, off in zip(rows["on"], rows["off"])]
        print(f"  on - off         {d[0]:+.4f}  {d[1]:+.4f}   {d[2]:+.4f} {d[3]:+.4f}")


if __name__ == "__main__":
    main()
