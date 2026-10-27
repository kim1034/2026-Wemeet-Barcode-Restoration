"""One explicit controlled run. Default invocation validates only; no GPU allocation.

Use --execute only after CPU preparation and GPU profiling are approved.
No service, queue, automatic retry, or multi-GPU distributed training is created.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.controlled_geometry.common import (
    capture_rng,
    configure,
    digest,
    read_json,
    restore_rng,
    save_checkpoint,
    selection_key,
    validate_lock,
    write_json,
)
from scripts.controlled_geometry.data import (
    GeometryDataset,
    RenderCache,
    load_split,
    validation_samples,
)
from scripts.controlled_geometry.evaluation import evaluate_samples, selection_metrics
from scripts.controlled_geometry.tracking import (
    configure_local_directories,
    log_artifact,
    start_run,
)
from wemeet.ai.geometry.controlled import ControlledGeometry


def run_epoch(model, loader, device, beta, optimizer=None):
    model.train(optimizer is not None)
    total_loss, count, steps = 0.0, 0, 0
    errors = []
    for images, targets, sizes, _ in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(images)
            loss = F.smooth_l1_loss(pred, targets, beta=beta)
            if not torch.isfinite(loss) or not torch.isfinite(pred).all():
                raise ValueError("Nonfinite loss/prediction")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise ValueError("Nonfinite gradient")
                optimizer.step()
                steps += 1
        n = images.shape[0]
        count += n
        total_loss += float(loss.detach().cpu()) * n
        difference = pred.detach().cpu().double() - targets.detach().cpu().double()
        errors.append((difference * (sizes.double() - 1).unsqueeze(1)).numpy())
    if not count:
        raise ValueError("Empty data loader")
    error = np.concatenate(errors)
    epe = np.linalg.norm(error, axis=2).ravel()
    return {
        "loss": total_loss / count,
        "samples": count,
        "optimizer_steps": steps,
        "epe_mean_px": float(epe.mean()),
        "epe_p95_px": float(np.quantile(epe, 0.95)),
        "epe_rmse_px": float(np.sqrt(np.mean(epe**2))),
        "mae_x_px": float(np.abs(error[:, :, 0]).mean()),
        "mae_y_px": float(np.abs(error[:, :, 1]).mean()),
    }


def make_checkpoint(model, optimizer, generator, epoch, config, best, steps, device):
    return {
        "format": "controlled-v2",
        "epoch": epoch,
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(generator, device),
        "best": best,
        "optimizer_steps": steps,
    }


def restore_checkpoint(payload, config, model, optimizer, generator, device="cpu"):
    if payload["format"] != "controlled-v2" or payload["config"] != config:
        raise ValueError("Resume configuration mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"], generator, device)
    return payload["epoch"], payload["best"], payload["optimizer_steps"]


def validate_config(config, verify_data=True):
    if digest(config["protocol_lock"]) != config["protocol_lock_sha256"]:
        raise ValueError("Protocol lock changed")
    lock = validate_lock(config["protocol_lock"], verify_data=verify_data)
    if config["model"] not in lock["pretrained"] or config["seed"] not in (42, 43, 44):
        raise ValueError("Unregistered model or seed")
    if config["preset"] not in lock["presets"]:
        raise ValueError("Unregistered preset")
    for key, value in lock["presets"][config["preset"]].items():
        if config[key] != value:
            raise ValueError(f"Changed preset field: {key}")
    if config["weight_decay"] != 1e-4 or config["validation_every"] != 5:
        raise ValueError("Changed optimizer or selection schedule")
    fixed = {
        "precision": "fp32",
        "scheduler": "none",
        "num_workers": 1,
        "cpu_threads": 2,
        "selection": lock["selection"],
        "wandb_group": lock["protocol"],
        "name": f"restoration-{config['model']}-preset{config['preset']}-s{config['seed']}-v2",
    }
    for key, value in fixed.items():
        if config[key] != value:
            raise ValueError(f"Changed locked setting: {key}")
    return lock


def execute(config, lock, device, resume=False):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    if device.startswith("cuda") and torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU per process")
    configure(config["seed"], config["cpu_threads"], device=device)
    split = load_split(lock["split_manifest"])
    cache = RenderCache(lock["cache_root"], lock["renderer_fingerprint"])
    output = Path(config["output"])
    if resume:
        payload = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        saved_run = read_json(output / "run.json")
        if payload["config"] != config or saved_run["config_sha256"] != digest_config(config):
            raise ValueError("Resume config does not match existing run")
    else:
        output.mkdir(parents=True, exist_ok=False)
        payload, saved_run = None, None
    configure_local_directories()
    with start_run(
        config["name"],
        config,
        output,
        mode="online",
        job_type="train",
        run_id=saved_run["id"] if saved_run else None,
        resume=resume,
    ) as run:
        write_json(
            output / "run.json",
            {"id": run.id, "url": run.url, "config_sha256": digest_config(config)},
        )
        write_json(output / "config.json", config)
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        # The artifact contains exact split/protocol/source provenance, not credentials.
        log_artifact(
            run,
            f"{config['name']}-inputs",
            "dataset",
            {
                "protocol.json": config["protocol_lock"],
                "split.json": lock["split_manifest"],
                "config.json": output / "config.json",
                "d0.json": lock["d0_manifest"],
                "source.zip": lock["source_archive"],
            },
        )
        model = ControlledGeometry(config["model"], lock["pretrained"][config["model"]]["path"]).to(
            device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        generator = torch.Generator().manual_seed(config["seed"])
        train_loader = DataLoader(
            GeometryDataset(split["train"], cache),
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=config["num_workers"],
            pin_memory=device.startswith("cuda"),
            **({"multiprocessing_context": "spawn"} if config["num_workers"] else {}),
        )
        val_loader = DataLoader(
            GeometryDataset(split["validation"], cache),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            **({"multiprocessing_context": "spawn"} if config["num_workers"] else {}),
        )
        epoch, best, steps = 0, None, 0
        if payload is not None:
            epoch, best, steps = restore_checkpoint(
                payload, config, model, optimizer, generator, device
            )
            if "metrics" in payload:
                record_metrics(output, payload["metrics"])
                if run.step <= epoch:
                    run.log(payload["metrics"], step=epoch)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        reference = read_json(lock["d0_manifest"])["validation"]
        started = time.perf_counter()
        for epoch in range(epoch + 1, config["epochs"] + 1):
            tick = time.perf_counter()
            train = run_epoch(model, train_loader, device, config["smooth_l1_beta"], optimizer)
            val = run_epoch(model, val_loader, device, config["smooth_l1_beta"])
            steps += train["optimizer_steps"]
            metrics = {f"train/{key}": value for key, value in train.items()}
            metrics.update({f"val/{key}": value for key, value in val.items()})
            if epoch % config["validation_every"] == 0 or epoch == config["epochs"]:
                summary, _ = evaluate_samples(
                    model,
                    validation_samples(split["validation"], cache),
                    device,
                    output=next_validation_path(output, epoch),
                    reference=reference,
                )
                metrics.update(selection_metrics(summary))
                key = selection_key(summary, epoch)
                if best is None or key < tuple(best["key"]):
                    best = {
                        "key": key,
                        "epoch": epoch,
                        "summary": summary,
                        "model": {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        },
                    }
            metrics.update(
                epoch=epoch,
                optimizer_steps=steps,
                samples_seen=epoch * len(split["train"]),
                lr=config["lr"],
                epoch_seconds=time.perf_counter() - tick,
                session_seconds=time.perf_counter() - started,
            )
            checkpoint = make_checkpoint(
                model, optimizer, generator, epoch, config, best, steps, device
            )
            checkpoint["metrics"] = metrics
            save_checkpoint(output / "last.pt", checkpoint)
            # The local log survives a W&B/network failure after checkpoint save.
            record_metrics(output, metrics)
            run.log(metrics, step=epoch)
            print(
                json.dumps({"epoch": epoch, "epoch_seconds": metrics["epoch_seconds"]}), flush=True
            )
        if best is None:
            raise ValueError("No eligible validation checkpoint")
        save_checkpoint(
            output / "best.pt",
            {
                "format": "controlled-v2-selected",
                "model": best["model"],
                "epoch": best["epoch"],
                "config": config,
                "selection": best["summary"],
                "selection_key": best["key"],
            },
        )
        summary_values = selection_metrics(best["summary"], "selected/val")
        summary_values.update(
            {
                "selected/epoch": best["epoch"],
                "selected/checkpoint_sha256": digest(output / "best.pt"),
                "cost/parameters": sum(p.numel() for p in model.parameters()),
                "cost/peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.startswith("cuda")
                else None,
            }
        )
        run.summary.update(summary_values)
        log_artifact(
            run,
            f"{config['name']}-model",
            "model",
            {
                "best.pt": output / "best.pt",
                "last.pt": output / "last.pt",
                "config.json": output / "config.json",
                "metrics.jsonl": output / "metrics.jsonl",
            },
            metadata={
                "selected_epoch": best["epoch"],
                "protocol_lock_sha256": config["protocol_lock_sha256"],
            },
        )
        write_json(
            output / "completion.json", {"status": "trained", "run_url": run.url, **summary_values}
        )


def next_validation_path(output, epoch):
    path = Path(output) / f"validation/epoch-{epoch:04d}"
    attempt = 1
    while path.exists():
        attempt += 1
        path = Path(output) / f"validation/epoch-{epoch:04d}-attempt-{attempt:02d}"
    return path


def record_metrics(output, metrics):
    path = Path(output) / "metrics.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    existing = [r for r in rows if r["epoch"] == metrics["epoch"]]
    if existing:
        if len(existing) != 1 or existing[0] != metrics:
            raise ValueError("Conflicting metrics for an already recorded epoch")
        return
    # Atomic rewrite prevents a truncated last JSONL line after interruption.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in [*rows, metrics])
    )
    temporary.replace(path)


def digest_config(config):
    from scripts.controlled_geometry.common import object_digest

    return object_digest(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly start one training run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    lock = validate_config(config)
    if not args.execute:
        print(
            json.dumps({"status": "validated_only", "name": config["name"], "gpu_allocated": False})
        )
        return
    execute(config, lock, args.device, args.resume)


if __name__ == "__main__":
    main()
