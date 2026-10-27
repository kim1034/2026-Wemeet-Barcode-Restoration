"""Pin published inputs and preserve the completed backbone campaign."""

import importlib.metadata
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from scripts.controlled_geometry.common import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/mobilenet-followup-20261027"
OLD = ROOT.parent / "2026-Wemeet-Barcode-Restoration"
PRIOR = OLD / "runs/stage2-controlled-v2"
CAMPAIGN = PRIOR / "campaign-gpu4567-20261026-resume-json"
DATA_REV = "41bd506fa5e9a5aac33f202d2f9ce98acc77d4f3"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "inputs.json").exists():
        raise RuntimeError("Inputs already prepared; verify and reuse, do not replace")
    data = OUT / "data"
    snapshot_download(
        "123metro/barcode-datasets",
        repo_type="dataset",
        revision=DATA_REV,
        allow_patterns=["benchmark/v2/*"],
        local_dir=data,
        max_workers=8,
    )
    weight_rev = HfApi().model_info("123metro/barcode-weights").sha
    detector = hf_hub_download(
        "123metro/barcode-weights",
        "detection/v2-yolo-obb/best.pt",
        revision=weight_rev,
        local_dir=OUT / "weights",
    )
    prior_lock = json.loads((PRIOR / "prepared/protocol.json").read_text())
    selected = {}
    preserved = {}
    for backbone in ("mobilenetv3-small", "resnet18-c3"):
        name = f"restoration-{backbone}-presetM-s42-v2"
        checkpoint = PRIOR / "training" / name / "best.pt"
        done = json.loads((checkpoint.parent / "completion.json").read_text())
        sha = digest(checkpoint)
        assert sha == done["selected/checkpoint_sha256"]
        selected[backbone] = {
            "path": str(checkpoint),
            "sha256": sha,
            "selected_epoch": done["selected/epoch"],
            "seed": 42,
            "preset": "M",
            "training_run": done["run_url"],
        }
        preserved[str(checkpoint)] = sha
    for base in [CAMPAIGN / "report", CAMPAIGN / "durable", PRIOR / "prepared"]:
        for p in base.rglob("*"):
            if p.is_file() and "wandb" not in p.parts and p.suffix not in (".log",):
                preserved[str(p)] = digest(p)
    for folder in [
        "scripts/controlled_geometry",
        "scripts/controlled_campaign",
        "scripts/controlled_resume_20261026",
        "scripts/controlled_durable_20261026",
    ]:
        for p in (OLD / folder).rglob("*.py"):
            preserved[str(p)] = digest(p)
    preserved[str(OLD / "wemeet/ai/geometry/controlled.py")] = digest(
        OLD / "wemeet/ai/geometry/controlled.py"
    )
    dataset_files = {str(p): digest(p) for p in (data / "benchmark/v2").rglob("*") if p.is_file()}
    rows = [json.loads(x) for x in (data / "benchmark/v2/manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 101 and len({x["id"] for x in rows[1:]}) == 100
    assert len(dataset_files) == 301
    inputs = {
        "protocol": "mobilenet-followup-20261027",
        "upstream_commit": "f449b9c34df3a42c5c324a0629b0dbee3308f1b3",
        "prior_campaign": str(CAMPAIGN),
        "prior_protocol": str(PRIOR / "prepared/protocol.json"),
        "prior_protocol_sha256": digest(PRIOR / "prepared/protocol.json"),
        "crop_data_root": prior_lock["data_root"],
        "split_manifest": prior_lock["split_manifest"],
        "cache_root": prior_lock["cache_root"],
        "renderer_fingerprint": prior_lock["renderer_fingerprint"],
        "scene_root": str(data / "benchmark/v2"),
        "scene_dataset_revision": DATA_REV,
        "scene_files": dataset_files,
        "checkpoints": selected,
        "detector": {"path": detector, "sha256": digest(detector), "hf_revision": weight_rev},
        "policies": ["early_exit", "always_rectify", "agreement"],
        "agreement_rule": (
            "D0 plus 3 distinct warp scales; at least 2 identical nonempty (text,format) "
            "observations and no conflicting nonempty result; no GT input"
        ),
        "decoder": "zxing-cpp only, native first-valid candidate; enhance false/true separately",
        "warp_scales": [1.0, 1.5, 3.0],
        "detector_sizes": [1280, 1600, 1920],
        "detector_clahe": True,
        "geometry_confidence": "availability only; never a calibrated score",
        "gpu_uuids": {
            "6": "GPU-e784131f-b03a-7638-3dd1-eb87207208fb",
            "7": "GPU-33fb8c54-a63e-6079-3ac3-792f7999fa11",
        },
        "timing": {"warmup": 50, "repeats": 3, "threads": 1, "serial_all_tasks": True},
        "selection": (
            "No new model/checkpoint selection. Policy tradeoffs reported separately; "
            "benchmark-v2 is a development regression set."
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torchvision",
                "numpy",
                "opencv-python",
                "zxing-cpp",
                "wandb",
                "ultralytics",
            ]
        },
    }
    write_json(OUT / "preservation_before.json", preserved)
    write_json(OUT / "inputs.json", inputs)
    write_json(
        OUT / "authorization.json",
        {
            "request": (
                "GPU 6/7: all four follow-up experiments, W&B records, "
                "combined Notion-ready and GitHub-shareable documentation"
            ),
            "gpu_indices": [6, 7],
            "training": False,
            "original_outputs": "preserve",
        },
    )
    old_report = ROOT / "docs/experiments/2026-10-26-stage2-controlled-v2"
    for name in ["REPORT_ko.md", "results.csv"]:
        shutil.copy2(CAMPAIGN / "report" / name, old_report / name)
    print(
        json.dumps(
            {
                "scene_files": len(dataset_files),
                "preserved": len(preserved),
                "inputs_sha256": digest(OUT / "inputs.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
