"""Actual dataset-image CPU checks; never a candidate-quality benchmark or training run."""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.controlled_geometry.common import (
    ROOT,
    configure,
    digest,
    read_json,
    validate_lock,
    write_json,
)
from scripts.controlled_geometry.data import benchmark_samples, preprocess
from scripts.controlled_geometry.evaluation import evaluate_samples
from wemeet.ai.geometry.controlled import BACKBONES, ControlledGeometry
from wemeet.ai.geometry.model import ResNet18C3, destination_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configure(42, threads=1)
    lock = validate_lock(args.protocol)
    samples = []
    for _, group in itertools.groupby(
        benchmark_samples(lock["data_root"]), key=lambda sample: sample[0].split("/")[1]
    ):
        samples.extend(itertools.islice(group, 2))
    image = torch.stack([preprocess(s[2]) for s in samples])
    details = {}
    for name in BACKBONES:
        model = ControlledGeometry(name, lock["pretrained"][name]["path"]).eval()
        with torch.inference_mode():
            feature = model.backbone(image)
            predicted = model(image)
        expected_channels = 256 if name == BACKBONES[0] else 48
        assert feature.shape == (6, expected_channels, 10, 24)
        assert torch.isfinite(feature).all()
        assert torch.equal(predicted, destination_grid().expand(6, -1, -1))
        summary, _ = evaluate_samples(
            model, samples, reference=read_json(lock["d0_manifest"])["benchmark"]
        )
        assert sum(summary["paths"]["crop_policy_model"][k] for k in "CWU") == 6
        details[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "feature_shape": list(feature.shape),
            "output_shape": list(predicted.shape),
            "pretrained_backbone_loaded": True,
            "identity_initialization_exact": True,
            "native_decode_path_checked": True,
        }
    weights = ROOT / "downloads/geometry/resnet18-c3/model_state_dict.pt"
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    legacy = ResNet18C3(pretrained=False).eval()
    new = ControlledGeometry(BACKBONES[0]).eval()
    legacy.load_state_dict(payload, strict=True)
    new.load_state_dict(payload, strict=True)
    with torch.inference_mode():
        a, b = legacy(image), new(image)
    maximum = float((a - b).abs().max())
    if maximum != 0 or not np.isfinite(maximum):
        raise AssertionError("Published ResNet parity failed")
    result = {
        "status": "passed",
        "scope": "technical CPU validation only; no model-quality claim",
        "protocol_sha256": digest(args.protocol),
        "samples": [s[0] for s in samples],
        "models": details,
        "legacy_weights_sha256": digest(weights),
        "legacy_resnet_max_abs_delta": maximum,
        "cuda_initialized": torch.cuda.is_initialized(),
        "full_training_started": False,
    }
    if result["cuda_initialized"]:
        raise AssertionError("CPU check initialized CUDA")
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
