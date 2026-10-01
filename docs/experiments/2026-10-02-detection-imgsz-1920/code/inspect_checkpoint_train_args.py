#!/usr/bin/env python3
"""배포된 가중치 파일에 ultralytics 가 직접 박아놓은 학습 설정을 확인한다.

문서(README)나 학습 스크립트의 argparse 기본값은 바뀔 수 있지만, 체크포인트에
저장된 `train_args` 는 그 가중치를 실제로 만든 학습 실행의 기록 그 자체다.

사용법::

    python inspect_checkpoint_train_args.py
"""

import torch

from wemeet.ai.detection.yolo_obb import resolve_weights


def main() -> None:
    path = resolve_weights()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args", {})
    print(f"weights: {path}")
    print(f"imgsz (실제 학습값): {train_args.get('imgsz')}")
    print(f"batch: {train_args.get('batch')}  epochs: {train_args.get('epochs')}")
    print(f"scale 증강: {train_args.get('scale')}  (학습 중 이 비율로 imgsz 를 흔들며 봤다)")
    print(f"data: {train_args.get('data')}")


if __name__ == "__main__":
    main()
