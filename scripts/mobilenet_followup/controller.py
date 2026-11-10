"""One durable writer; two evaluation workers, then exclusive serial timing."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.controlled_campaign.hardware import idle, snapshot
from scripts.controlled_geometry.common import digest, read_json, write_json


def task_matrix():
    result = []
    for enhance in (False, True):
        mode = "clahe" if enhance else "legacy"
        for dataset in ("validation", "benchmark"):
            result.append(
                {
                    "id": f"crop-{dataset}-{mode}",
                    "kind": "crop",
                    "dataset": dataset,
                    "model": "mobilenetv3-small",
                    "enhance": enhance,
                }
            )
        for model, size in [
            ("resnet18-c3", 1280),
            ("mobilenetv3-small", 1280),
            ("mobilenetv3-small", 1600),
            ("mobilenetv3-small", 1920),
        ]:
            result.append(
                {
                    "id": f"scene-{model}-{size}-{mode}",
                    "kind": "scene",
                    "dataset": "scene-v2",
                    "model": model,
                    "imgsz": size,
                    "enhance": enhance,
                }
            )
    for accuracy in list(result):
        if accuracy["dataset"] == "validation":
            continue
        result.append(
            {
                **accuracy,
                "id": accuracy["id"] + "-timing",
                "kind": accuracy["kind"] + "_timing",
                "accuracy_task": accuracy["id"],
            }
        )
    return result


def audit_job(root, task):
    done_path = root / "jobs" / task["id"] / "completion.json"
    if not done_path.exists():
        return None
    done = read_json(done_path)
    if done["task"] != task or done["status"] != "complete":
        raise ValueError("Task completion identity mismatch")
    out = Path(done["output"])
    for name, expected in done["files"].items():
        if digest(out / name) != expected:
            raise ValueError("Task output was changed")
    expected_n = (
        2976 if task["dataset"] == "validation" else 1500 if task["dataset"] == "benchmark" else 100
    )
    summary = read_json(out / "summary.json")
    if summary["n"] != expected_n:
        raise ValueError("Partial population cannot complete task")
    if "timing" not in task["kind"]:
        records = [json.loads(x) for x in (out / "per_sample.jsonl").read_text().splitlines()]
        if len(records) != expected_n or len({r["id"] for r in records}) != expected_n:
            raise ValueError("Missing or duplicate samples")
        for record in records:
            for policy, value in record["decisions"].items():
                actual = (
                    "U"
                    if value["text"] is None
                    else "C"
                    if value["text"] == record["gt_text"] and value["format"] == "Code128"
                    else "W"
                )
                if record["statuses"][policy] != actual:
                    raise ValueError("Raw prediction/GT grade mismatch")
        for path, metrics in summary["paths"].items():
            for label in ("C", "W", "U"):
                if metrics[label] != sum(r["statuses"][path] == label for r in records):
                    raise ValueError("Aggregate arithmetic mismatch")
    else:
        for metrics in summary["paths"].values():
            if len(metrics["passes"]) != 3 or not metrics["decision_parity"]:
                raise ValueError("Incomplete timing")
            if any(len(p["raw_ms"]) != expected_n for p in metrics["passes"]):
                raise ValueError("Timing population mismatch")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = read_json(root / "inputs.json")
        frozen = read_json(root / "execution_lock.json")
        for filename, expected in frozen.items():
            if digest(filename) != expected:
                raise ValueError(f"Frozen execution changed: {filename}")
        tasks = read_json(root / "tasks.json")
        if tasks != task_matrix():
            raise ValueError("Task matrix changed")
        state = {
            "status": "running",
            "completed": [],
            "total": len(tasks),
            "lanes": {},
            "errors": [],
        }
        import threading

        mutex = threading.Lock()

        def save():
            state["updated_at"] = time.time()
            write_json(root / "progress.json", state)

        def run_task(task, gpu):
            done = audit_job(root, task)
            if done:
                with mutex:
                    state["completed"].append(task["id"])
                    save()
                return
            directory = root / "jobs" / task["id"]
            directory.mkdir(parents=True, exist_ok=True)
            attempts = sorted(directory.glob("attempt-*"))
            attempt = int(attempts[-1].name.split("-")[-1]) if attempts else 1
            if attempts and not (attempts[-1] / "results.done.json").exists():
                attempt += 1
            for retry in range(3):
                while not idle(snapshot(), gpu, inputs["gpu_uuids"][str(gpu)]):
                    with mutex:
                        state["lanes"][str(gpu)] = {
                            "task": task["id"],
                            "status": "waiting_for_idle_gpu",
                        }
                        save()
                    time.sleep(20)
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=inputs["gpu_uuids"][str(gpu)],
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    CUBLAS_WORKSPACE_CONFIG=":4096:8",
                )
                logfile = directory / f"worker-{attempt:02d}-{int(time.time())}.log"
                with logfile.open("w") as stream:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "scripts.mobilenet_followup.worker",
                            "--root",
                            str(root),
                            "--task",
                            task["id"],
                            "--gpu",
                            str(gpu),
                            "--attempt",
                            str(attempt),
                        ],
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                    with mutex:
                        state["lanes"][str(gpu)] = {
                            "task": task["id"],
                            "pid": process.pid,
                            "attempt": attempt,
                        }
                        save()
                    started = time.monotonic()
                    timeout = 7200 if "timing" in task["kind"] else 3600
                    while process.poll() is None:
                        if time.monotonic() - started > timeout:
                            process.terminate()
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            break
                        with mutex:
                            save()
                        time.sleep(5)
                if process.returncode == 0 and audit_job(root, task):
                    with mutex:
                        state["completed"].append(task["id"])
                        state["lanes"].pop(str(gpu), None)
                        save()
                    return
                with mutex:
                    state["errors"].append(
                        {
                            "task": task["id"],
                            "attempt": attempt,
                            "exit": process.returncode,
                            "log": str(logfile),
                        }
                    )
                    save()
                if not (directory / f"attempt-{attempt:02d}" / "results.done.json").exists():
                    attempt += 1
                time.sleep(30 * (retry + 1))
            raise RuntimeError(f"Task failed after preserved attempts: {task['id']}")

        accuracy = [t for t in tasks if "timing" not in t["kind"]]

        def lane(gpu, assigned):
            for task in assigned:
                run_task(task, gpu)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, 6, accuracy[::2]), pool.submit(lane, 7, accuracy[1::2])]
            for future in futures:
                future.result()
        for i, task in enumerate(t for t in tasks if "timing" in t["kind"]):
            run_task(task, 6 if i % 2 == 0 else 7)
        state["status"] = "evaluations_complete"
        save()
        from scripts.mobilenet_followup.report import finalize

        finalize(root)
        state["status"] = "complete"
        save()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        # systemd does not repeatedly retry an exhausted or invalid experiment.
        sys.exit(78)
