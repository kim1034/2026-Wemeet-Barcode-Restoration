"""Seal the reviewed source and inputs before enabling the durable controller."""

import importlib.metadata
from pathlib import Path

from scripts.controlled_campaign.hardware import idle, snapshot
from scripts.controlled_geometry.common import digest, read_json, write_json
from scripts.mobilenet_followup.controller import task_matrix


def main():
    repo = Path(__file__).resolve().parents[2]
    root = repo / "runs/mobilenet-followup-20261027"
    if (root / "execution_lock.json").exists():
        raise RuntimeError("Existing execution lock must not be overwritten")
    inputs = read_json(root / "inputs.json")
    for name, expected in inputs["packages"].items():
        assert importlib.metadata.version(name) == expected
    for gpu in (6, 7):
        assert (
            read_json(root / f"admission/gpu{gpu}-attempt1/completion.json")["status"] == "passed"
        )
        assert idle(snapshot(), gpu, inputs["gpu_uuids"][str(gpu)])
    for filename, expected in read_json(root / "preservation_before.json").items():
        assert digest(filename) == expected
    for filename, expected in inputs["scene_files"].items():
        assert digest(filename) == expected
    write_json(root / "tasks.json", task_matrix())
    files = sorted((repo / "scripts").rglob("*.py")) + sorted((repo / "wemeet").rglob("*.py"))
    files += sorted((repo / "tests").rglob("*.py"))
    files += [root / n for n in ("inputs.json", "tasks.json", "preservation_before.json")]
    write_json(root / "execution_lock.json", {str(p): digest(p) for p in files})
    print({"frozen_files": len(files), "tasks": len(task_matrix())})


if __name__ == "__main__":
    main()
