"""Small real-input checks before starting the complete, frozen campaign."""

import argparse
from pathlib import Path
from unittest.mock import patch

from scripts.controlled_geometry.common import configure, read_json, write_json
from scripts.mobilenet_followup import worker
from scripts.mobilenet_followup.core import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--attempt", type=int, default=1)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / "admission" / f"gpu{args.gpu}-attempt{args.attempt}"
    out.mkdir(parents=True, exist_ok=False)
    inputs = read_json(root / "inputs.json")
    configure(42, threads=1, device="cuda:0")
    backbone = "mobilenetv3-small" if args.gpu == 6 else "resnet18-c3"
    predictor = Predictor(inputs["checkpoints"][backbone], backbone)
    summaries = {}
    if args.gpu == 6:
        for dataset in ("validation", "benchmark"):
            samples = worker.crop_samples(inputs, dataset)
            selected = [samples[i] for i in [0, 1, 2, len(samples) // 2, len(samples) - 1]]
            task = {"kind": "crop", "dataset": dataset, "enhance": False}
            directory = out / dataset
            directory.mkdir()
            with patch.object(worker, "crop_samples", return_value=selected):
                summaries[dataset] = worker.evaluate(inputs, task, predictor, directory, args.gpu)
    samples = worker.scene_samples(inputs)[:12]
    for enhance in (False, True):
        task = {"kind": "scene", "dataset": "scene-v2", "imgsz": 1280, "enhance": enhance}
        directory = out / f"scene-{enhance}"
        directory.mkdir()
        with patch.object(worker, "scene_samples", return_value=samples):
            summaries[str(enhance)] = worker.evaluate(inputs, task, predictor, directory, args.gpu)
    write_json(out / "completion.json", {"status": "passed", "summaries": summaries})
    print({"gpu": args.gpu, "status": "passed", "output": str(out)})


if __name__ == "__main__":
    main()
