"""Observe physical GPU identity and processes without allocating a CUDA context."""

import csv
import io
import os
import subprocess
from datetime import UTC, datetime


def utc_now():
    return datetime.now(UTC).isoformat()


def query(kind, fields):
    result = subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    return [
        dict(zip(fields, (v.strip() for v in row), strict=True))
        for row in csv.reader(io.StringIO(result))
        if row
    ]


def snapshot():
    return {
        "at_utc": utc_now(),
        "gpus": query(
            "gpu", ["index", "uuid", "name", "memory.total", "memory.used", "utilization.gpu"]
        ),
        "processes": query("compute-apps", ["pid", "gpu_uuid", "used_gpu_memory"]),
        "host_load": list(os.getloadavg()),
    }


def assigned_gpu(observation, index, uuid):
    matches = [g for g in observation["gpus"] if int(g["index"]) == index]
    if len(matches) != 1 or matches[0]["uuid"] != uuid:
        raise RuntimeError("Physical GPU index/UUID changed")
    return matches[0]


def idle(observation, index, uuid):
    gpu = assigned_gpu(observation, index, uuid)
    occupied = [p for p in observation["processes"] if p["gpu_uuid"] == uuid]
    return not occupied and float(gpu["memory.used"]) < 64 and float(gpu["utilization.gpu"]) < 5


def prove_own_cuda(index, uuid):
    observation = snapshot()
    assigned_gpu(observation, index, uuid)
    own = [p for p in observation["processes"] if int(p["pid"]) == os.getpid()]
    if len(own) != 1 or own[0]["gpu_uuid"] != uuid:
        raise RuntimeError("Actual CUDA PID/UUID observation did not match the assigned GPU")
    others = [p for p in observation["processes"] if p["gpu_uuid"] == uuid and p not in own]
    if others:
        raise RuntimeError("Concurrent GPU workload invalidates the profile")
    return {"pid": os.getpid(), "gpu_index": index, "gpu_uuid": uuid, "observation": observation}
