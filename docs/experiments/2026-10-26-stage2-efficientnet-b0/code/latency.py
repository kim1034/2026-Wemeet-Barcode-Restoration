"""CPU 지연 측정: ResNet18-C3 vs EfficientNet-B0-C3 (가이드 4장).

두 모델을 **같은 컴퓨터에서, 다른 프로그램을 끈 상태로** 잰다.
- `torch.set_num_threads(1)`, batch 1, 입력 `torch.zeros(1, 3, 160, 384)`
- warm-up 20회 후 200회 측정해서 p50 / p95 (ms)
- CPU 이름도 같이 적는다

지연은 가중치 값과 무관하므로 기본은 무작위 초기화 모델로 잰다. `--resnet-checkpoint`,
`--efficientnet-checkpoint` 를 주면 학습된 가중치를 올려서 잰다.
모델 forward 만 잰다. 전처리·TPS 보정·디코딩 시간은 포함하지 않는다.

실행 (저장소 루트에서):
    PYTHONPATH=. uv run python docs/experiments/2026-10-26-stage2-efficientnet-b0/code/latency.py --wandb
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from model import EfficientNetB0C3, count_params

from wemeet.ai.geometry.model import ResNet18C3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--resnet-checkpoint", type=Path, default=None)
    parser.add_argument("--efficientnet-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/latency-efb0-vs-resnet18.json"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--offline-wandb", action="store_true")
    parser.add_argument("--run-name", default="latency-efb0-vs-resnet18")
    parser.add_argument("--smoke", action="store_true", help="시험 실행. W&B tag 에 smoke 를 붙인다")
    return parser


def cpu_name() -> str:
    """OS 마다 CPU 이름을 얻는 방법이 달라서 순서대로 시도한다."""
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if out.stdout.strip():
                return out.stdout.strip()
        if system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
            )
            if out.stdout.strip():
                return out.stdout.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _load(model: torch.nn.Module, checkpoint: Path | None) -> None:
    if checkpoint is None:
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)


def measure(model: torch.nn.Module, warmup: int, iters: int) -> dict:
    model.eval()
    x = torch.zeros(1, 3, 160, 384)
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(iters):
            started = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - started) * 1000.0)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def main() -> None:
    args = _parser().parse_args()
    torch.set_num_threads(args.threads)
    resnet = ResNet18C3(pretrained=False)
    efficientnet = EfficientNetB0C3(pretrained=False)
    _load(resnet, args.resnet_checkpoint)
    _load(efficientnet, args.efficientnet_checkpoint)

    env = {
        "cpu": cpu_name(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "batch": 1,
        "input": [1, 3, 160, 384],
        "warmup": args.warmup,
        "iters": args.iters,
    }
    results = {}
    for key, model in (("resnet18-c3", resnet), ("efficientnet-b0-c3", efficientnet)):
        results[key] = {"params": count_params(model), **measure(model, args.warmup, args.iters)}
        print(json.dumps({"model": key, **results[key]}, ensure_ascii=False))

    output = {"env": env, "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.wandb:
        import wandb

        tags = ["stage2", "backbone-compare", "efficientnet-b0-c3", "resnet18-c3", "latency"]
        if args.smoke:
            tags.append("smoke")
        with wandb.init(
            project="wemeet-barcode",
            entity="kim-1034",
            name=args.run_name,
            job_type="latency",
            tags=tags,
            mode="offline" if args.offline_wandb else "online",
            config=env,
        ) as run:
            metrics = {}
            for key, values in results.items():
                for name, value in values.items():
                    metrics[f"latency/{key}/{name}"] = value
            run.log(metrics)
            run.summary.update(metrics)
            print(f"W&B run: {run.url}")


if __name__ == "__main__":
    main()
