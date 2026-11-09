#!/usr/bin/env python3
"""
MIG 에서 AutoBatch 대신 배치 크기를 직접 잰다.

MIG 1g.10gb 는 NVML 권한이 없어서, AutoBatch 가 배치를 늘려 보다가 메모리가 넘치면
OOM 대신 'NVML_SUCCESS == r INTERNAL ASSERT FAILED' 로 죽고, 기본값 16 으로 학습하다 또 죽는다
(11/06 1920 s 스모크). 그래서 배치마다 새 프로세스에서 forward+backward 를 돌려 최대 메모리를 잰다.

  python probe_batch.py yolo11s-obb.pt --imgsz 1920          # → "추천 배치 N" 출력
"""

import argparse
import subprocess
import sys

CHILD = r"""
import sys, torch
from ultralytics import YOLO
w, imgsz, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = YOLO(w).model.cuda().train()
for p in m.parameters():
    p.requires_grad_(True)
x = torch.rand(b, 3, imgsz, imgsz, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    y = m(x)
def flat(o):  # 학습 모드 출력은 버전에 따라 텐서·리스트·dict
    if torch.is_tensor(o):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from flat(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from flat(v)
sum(t.float().sum() for t in flat(y)).backward()
torch.cuda.synchronize()
print(f"PEAK {torch.cuda.max_memory_reserved() / 1e9:.3f} {torch.cuda.get_device_properties(0).total_memory / 1e9:.3f}")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--target", type=float, default=0.60, help="총 메모리 대비 목표 사용률 (AutoBatch 기본과 같음)")
    ap.add_argument("--max", type=int, default=32)
    args = ap.parse_args()

    best, b = 0, 1
    while b <= args.max:
        r = subprocess.run([sys.executable, "-c", CHILD, args.weights, str(args.imgsz), str(b)],
                           capture_output=True, text=True)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PEAK")), None)
        if not line:
            print(f"batch {b:2d}: 실패 (메모리 초과로 판단)")
            break
        peak, total = map(float, line.split()[1:])
        ok = peak <= total * args.target
        print(f"batch {b:2d}: 최대 {peak:.2f} / {total:.2f} GB ({peak / total:.0%}) {'OK' if ok else '목표 초과'}")
        if not ok:
            break
        best, b = b, b + 1
    print(f"추천 배치 {best}")
    sys.exit(0 if best else 1)


if __name__ == "__main__":
    main()
