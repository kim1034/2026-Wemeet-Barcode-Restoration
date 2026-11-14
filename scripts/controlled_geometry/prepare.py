"""CPU-only input freeze and 12-run matrix. Never starts training or a GPU job."""

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights

from scripts.controlled_geometry.common import (
    DATA_REVISION,
    PRESETS,
    PROTOCOL,
    ROOT,
    configure,
    digest,
    object_digest,
    package_versions,
    read_json,
    source_hashes,
    write_json,
)
from scripts.controlled_geometry.data import (
    RenderCache,
    benchmark_samples,
    load_split,
    validation_samples,
)
from scripts.controlled_geometry.evaluation import reference_record
from scripts.controlled_geometry.tracking import (
    configure_local_directories,
    log_artifact,
    probe,
    start_run,
)
from scripts.evaluate_stage2_v1 import observe_decode
from scripts.train_geometry import read_records
from wemeet.ai.geometry.controlled import BACKBONES
from wemeet.ai.geometry.model import destination_grid
from wemeet.schemas import DetectedBarcode, RectifiedBarcode
from wemeet.sw import decoding, pipeline
from wemeet.sw.rectify import _FLOW_STEP_PX


def verify_dataset(root):
    root = Path(root).resolve()
    published_root = root.parents[1]
    manifest = published_root / "sha256.json"
    files = {str(manifest): digest(manifest)}
    for relative, expected in read_json(manifest).items():
        if not relative.startswith("synthetic/v1/"):
            continue
        path = published_root / relative
        if digest(path) != expected:
            raise ValueError(f"Published dataset hash mismatch: {relative}")
        files[str(path)] = expected
    if len(files) < 3007:
        raise ValueError("Incomplete published dataset manifest")
    return files


def freeze_pretrained(directory, download=False):
    directory = Path(directory).resolve()
    torch.hub.set_dir(str(directory))
    result = {}
    for name, weights in zip(
        BACKBONES,
        (
            ResNet18_Weights.IMAGENET1K_V1,
            MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        ),
        strict=True,
    ):
        filename = Path(urlparse(weights.url).path).name
        path = directory / "checkpoints" / filename
        if not path.is_file():
            if not download:
                raise FileNotFoundError(
                    f"Pretrained weights absent; use --download-weights: {path}"
                )
            weights.get_state_dict(progress=False, check_hash=True)
        sha = digest(path)
        if not sha.startswith(filename.rsplit("-", 1)[1].split(".")[0]):
            raise ValueError(f"Pretrained checksum mismatch: {path}")
        result[name] = {"path": str(path), "sha256": sha, "enum": str(weights), "url": weights.url}
    return result


def freeze_d0(samples, training_recipes):
    result = {}
    dst = destination_grid().numpy()
    fields = set(next(iter(training_recipes.values())))
    recipe_hashes = {object_digest(r) for r in training_recipes.values()}
    for index, (sid, row, gray, src, sample_dst) in enumerate(samples):
        if sid in result or not np.isfinite(src).all() or src.shape != (48, 2):
            raise ValueError("Invalid or duplicate evaluation sample")
        if not np.allclose(sample_dst, dst, atol=1e-6):
            raise ValueError("Evaluation destination grid changed")
        if row["symbology"].lower() != "code128":
            raise ValueError("Evaluation is restricted to Code128")
        recipe = {key: row[key] for key in fields if key in row}
        if object_digest(recipe) in recipe_hashes:
            raise ValueError("Exact training recipe in evaluation")
        target = DetectedBarcode(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0, 1)
        _, decoded = observe_decode(RectifiedBarcode(gray, target, None))
        result[sid] = reference_record(row, gray, src, decoded)
        if (index + 1) % 250 == 0:
            print(json.dumps({"phase": "freeze_d0", "samples": index + 1, "last": sid}), flush=True)
    return result


def generate_configs(lock_path, output_root):
    lock = read_json(lock_path)
    configs = []
    for seed in (42, 43, 44):
        for preset, values in PRESETS.items():
            for model in BACKBONES:
                name = f"restoration-{model}-preset{preset}-s{seed}-v2"
                config = {
                    "name": name,
                    "model": model,
                    "preset": preset,
                    "seed": seed,
                    **values,
                    "weight_decay": 1e-4,
                    "validation_every": 5,
                    "num_workers": 1,
                    "cpu_threads": 2,
                    "precision": "fp32",
                    "scheduler": "none",
                    "protocol_lock": str(Path(lock_path).resolve()),
                    "protocol_lock_sha256": digest(lock_path),
                    "output": str(Path(output_root).resolve() / "training" / name),
                    "selection": lock["selection"],
                    "wandb_group": PROTOCOL,
                }
                configs.append(config)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT.parent / "data/barcode-datasets-a833a3a59441/synthetic/v1",
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "runs/restoration-resnet18-c3-v1/split.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/stage2-controlled-v2/preparation"
    )
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--wandb-online-check", action="store_true")
    args = parser.parse_args()
    # Hide devices before even checking their availability. No CUDA calls are required.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configure()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise FileExistsError("Preparation already frozen; choose a new output directory")
    sources = source_hashes()
    packages = package_versions()
    split = load_split(args.split)
    if (len(split["train"]), len(split["validation"])) != (27024, 2976):
        raise ValueError("Expected historical 27024/2976 split")
    print(json.dumps({"phase": "verify_published_data"}), flush=True)
    files = verify_dataset(args.data_root)
    regenerated = read_records(
        args.data_root / "recipes", {"target": 5000, "hard": 3500, "first_ok": 1500}, 0.1, 42
    )
    if split != dict(zip(("train", "validation"), regenerated, strict=True)):
        raise ValueError("Split recipes differ from the published selection")
    shutil.copyfile(args.split, output / "split.json")
    pretrained = freeze_pretrained(ROOT.parent / ".cache/torch/hub", args.download_weights)
    renderer = object_digest(
        {
            "sources": {k: v for k, v in sources.items() if k.startswith("wemeet/data/")},
            "packages": packages,
        }
    )
    cache_root = output.parent / "render-cache" / renderer
    cache = RenderCache(cache_root, renderer)
    training_recipes = {r["id"]: r["recipe"] for r in split["train"]}
    print(json.dumps({"phase": "freeze_validation_d0"}), flush=True)
    validation = freeze_d0(validation_samples(split["validation"], cache), training_recipes)
    print(json.dumps({"phase": "freeze_benchmark_d0"}), flush=True)
    benchmark = freeze_d0(benchmark_samples(args.data_root), training_recipes)
    write_json(output / "d0.json", {"validation": validation, "benchmark": benchmark})
    train_payloads = {r["recipe"]["text"] for r in split["train"]}
    audit = {
        "train_n": len(split["train"]),
        "validation_n": len(validation),
        "benchmark_n": len(benchmark),
        "published_files_verified": len(files) - 1,
        "train_validation_id_overlap": 0,
        "exact_train_eval_recipe_overlap": 0,
        "train_validation_payload_overlap": len(
            train_payloads & {r["gt_text"] for r in validation.values()}
        ),
        "benchmark_payloads_in_train": len(
            train_payloads & {r["gt_text"] for r in benchmark.values()}
        ),
        "d0_validation": dict(Counter(r["d0_status"] for r in validation.values())),
        "d0_benchmark": dict(Counter(r["d0_status"] for r in benchmark.values())),
        "group_independence": "not established; payload is only a proxy",
        "benchmark_role": "historical development benchmark, not a fresh final test",
    }
    write_json(output / "data_audit.json", audit)
    if sources != source_hashes():
        raise RuntimeError(
            "Sources changed during preparation; preserve this attempt and prepare again"
        )
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in sources:
            archive.write(ROOT / relative, relative)
    lock = {
        "protocol": PROTOCOL,
        "dataset_revision": DATA_REVISION,
        "data_root": str(args.data_root.resolve()),
        "dataset_files": files,
        "split_manifest": str(output / "split.json"),
        "split_sha256": digest(output / "split.json"),
        "sources": sources,
        "source_archive": str(output / "source.zip"),
        "source_archive_sha256": digest(output / "source.zip"),
        "packages": packages,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "reviewed_upstream_commit": "f449b9c34df3a42c5c324a0629b0dbee3308f1b3",
        "pretrained": pretrained,
        "presets": PRESETS,
        "renderer_fingerprint": renderer,
        "cache_root": str(cache_root),
        "d0_manifest": str(output / "d0.json"),
        "d0_manifest_sha256": digest(output / "d0.json"),
        "selection": (
            "validation crop_policy_model: RR_H max, FDR_H min, EPE mean min, earlier epoch"
        ),
        "evaluation_every_epochs": 5,
        "last_epoch_evaluated": True,
        "input_hw": [160, 384],
        "control_grid": [16, 3],
        "pretrained_policy": "ImageNet1K V1",
        "head": "C->256 adapter; spatial 256->128->64->2; identity + 0.5*tanh residual",
        "determinism": {
            "cpu": "strict deterministic algorithms",
            "cuda": "deterministic algorithms where available; warn-only for unsupported ops",
            "known_cuda_limitation": "PyTorch 2.8 AdaptiveAvgPool2d backward",
            "cuda_bitwise_replay_verified": False,
            "reference": (
                "https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html"
            ),
        },
        "decoder": {
            "pyzbar_available": decoding._pyzbar_decode is not None,
            "zxing_available": decoding._zxingcpp is not None,
        },
        "retry_scales": list(pipeline._RETRY_OUT_SCALES),
        "flow_step_px": _FLOW_STEP_PX,
        "mobile_original_provenance": {
            "run_url": "https://wandb.ai/kim-1034/wemeet-barcode/runs/rjujpri3",
            "historical_run_config_verified": True,
            "historical_implementation_available": False,
            "historical_checkpoint_available": False,
            "candidate_scope": "new shared-head candidate; not an exact historical reproduction",
        },
        "gpu_profile_status": "pending; no GPU allocation authorized in this preparation",
    }
    write_json(output / "protocol.json", lock)
    configs = generate_configs(output / "protocol.json", output.parent)
    for config in configs:
        write_json(output / "configs" / f"{config['name']}.json", config)
    write_json(output / "matrix.json", configs)
    configure_local_directories()
    wandb_status = probe()
    if args.wandb_online_check and wandb_status["online_read_verified"]:
        with start_run(
            "controlled-v2-cpu-preparation",
            {"protocol": PROTOCOL, "gpu_used": False},
            output,
            job_type="preparation",
        ) as run:
            run.log(
                {
                    "checks/input_freeze": 1,
                    "data/train_n": len(split["train"]),
                    "data/val_n": len(validation),
                }
            )
            log_artifact(
                run,
                "controlled-v2-input-freeze",
                "dataset",
                {
                    "protocol.json": output / "protocol.json",
                    "split.json": output / "split.json",
                    "data_audit.json": output / "data_audit.json",
                    "matrix.json": output / "matrix.json",
                    "d0.json": output / "d0.json",
                    "source.zip": output / "source.zip",
                },
            )
            wandb_status.update(online_write_verified=True, run_url=run.url)
    else:
        wandb_status["online_write_verified"] = False
    write_json(output / "wandb_status.json", wandb_status)
    write_json(
        output / "preparation.json",
        {
            "status": "inputs_prepared",
            "gpu_initialized": torch.cuda.is_initialized(),
            "full_training_started": False,
            "configs": len(configs),
            "audit": audit,
            "wandb": wandb_status,
            "gpu_profile": "pending",
            "protocol_sha256": digest(output / "protocol.json"),
        },
    )
    print(
        json.dumps(
            {
                "prepared": str(output),
                "runs_planned": len(configs),
                "gpu_initialized": torch.cuda.is_initialized(),
            }
        )
    )


if __name__ == "__main__":
    main()
