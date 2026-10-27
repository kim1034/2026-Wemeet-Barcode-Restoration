"""Package the reviewed local commit and documents; publish artifacts to W&B."""

import argparse
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import wandb

from scripts.controlled_geometry.common import digest, read_json, write_json

BASE = "f449b9c34df3a42c5c324a0629b0dbee3308f1b3"
BRANCH = "ai/mobilenet-followup-20261027"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    assert read_json(root / "completion.json")["status"] == "complete"
    assert read_json(root / "delivery/completion.json")["status"] == "complete"
    assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == BRANCH
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert commit != BASE
    output = root / "share"
    output.mkdir(exist_ok=True)
    if (output / "completion.json").exists():
        raise RuntimeError("Share package already completed; preserve it")
    folders = [
        repo / "docs/experiments/2026-10-26-stage2-controlled-v2",
        repo / "docs/experiments/2026-10-27-mobilenet-followup",
    ]
    if (output / "manifest.json").exists():
        previous = read_json(output / "manifest.json")
        assert previous["commit"] == commit
        for name, expected in previous["files"].items():
            assert digest(output / name) == expected
    else:
        if any(output.iterdir()):
            raise RuntimeError("Incomplete package preserved; use a separate reviewed revision")
        archive = output / "notion-ready.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for folder in folders:
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        z.write(path, str(path.relative_to(folder.parent)))
            z.writestr(
                "START_HERE.txt",
                "Notion: 2026-10-27-mobilenet-followup/NOTION_ko.md\n"
                "Decision: 2026-10-27-mobilenet-followup/DECISION_ko.md\n"
                "Includes figures, CSV results, failure records, and reproduction notes.\n",
            )
        with (output / "github-changes.patch").open("wb") as stream:
            subprocess.run(
                ["git", "format-patch", "--binary", "--full-index", "--stdout", BASE + "..HEAD"],
                stdout=stream,
                check=True,
            )
        subprocess.run(
            ["git", "bundle", "create", str(output / "github-branch.bundle"), BASE + ".." + BRANCH],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar.gz",
                "--output=" + str(output / "source.tar.gz"),
                "HEAD",
            ],
            check=True,
        )
        write_json(
            output / "manifest.json",
            {
                "base": BASE,
                "commit": commit,
                "branch": BRANCH,
                "remote_github_publication": False,
                "reason": "connected account has pull but no push",
                "notion_document_prepared": True,
                "notion_remote_publication": False,
                "files": {
                    p.name: digest(p)
                    for p in output.iterdir()
                    if p.is_file() and p.name not in ("manifest.json", "completion.json")
                },
            },
        )
    run = wandb.init(
        entity="kim-1034",
        project="wemeet-barcode",
        group="mobilenet-followup-20261027",
        name="notion-and-github-share-package",
        job_type="report",
        dir=str(root),
        mode="online",
        config={"commit": commit, "base": BASE, "branch": BRANCH},
    )
    artifact = wandb.Artifact("mobilenet-followup-share-package", type="report")
    for folder in folders:
        artifact.add_dir(str(folder), name="docs/" + folder.name)
    for path in output.iterdir():
        if path.is_file():
            artifact.add_file(str(path), name="share/" + path.name)
    for path in (root / "delivery").glob("*.json"):
        artifact.add_file(str(path), name="audit/" + path.name)
    preserved = read_json(root / "native_bridge_amendment/preserved_before.json")
    for name, expected in preserved["failed_files"].items():
        path = Path(name)
        assert digest(path) == expected
        artifact.add_file(str(path), name="preserved_failures/" + str(path.relative_to(root)))
    for name in ("prepare.log", "prepare-attempt02.log"):
        artifact.add_file(str(root / name), name="preserved_failures/" + name)
    dependency_fix = root / "delivery_dependency_fix_20261027"
    if dependency_fix.exists():
        for name, expected in read_json(dependency_fix / "preserved.json")["files"].items():
            assert digest(name) == expected
        artifact.add_dir(str(dependency_fix), name="preserved_failures/reporting_dependency_fix")
    run.log_artifact(artifact).wait()
    url = run.url
    run.summary.update({"prepared": True, "github_pushed": False, "notion_posted": False})
    run.finish()
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "wandb": url,
            "commit": commit,
            "branch": BRANCH,
            "completed_at": datetime.now(UTC).isoformat(),
            "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print({"wandb": url, "commit": commit, "output": str(output)})


if __name__ == "__main__":
    main()
