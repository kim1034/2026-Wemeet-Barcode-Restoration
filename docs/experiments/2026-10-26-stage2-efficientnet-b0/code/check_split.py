"""split.json 이 ResNet 기준선 실행의 것과 같은지 확인한다 (가이드 4장 실행 순서 3).

ResNet 쪽 split.json 은 둘 중 하나로 가져온다.
- `--reference` : 이미 가지고 있는 ResNet split.json 경로 (gpu06 의 runs/restoration-resnet18-c3-v1/)
- `--from-wandb`: ResNet 학습 실행(kim-1034/wemeet-barcode/p9eghdt2)이 올린 모델 artifact 에서
                  split.json **한 파일만** 내려받는다 (가중치는 받지 않는다)

실행 (저장소 루트에서):
    PYTHONPATH=. uv run python docs/experiments/2026-10-26-stage2-efficientnet-b0/code/check_split.py \
        --candidate runs/restoration-efficientnet-b0-c3-v1/split.json --from-wandb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RESNET_TRAIN_RUN = "kim-1034/wemeet-barcode/p9eghdt2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True, help="EfficientNet 의 split.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", type=Path, help="ResNet 의 split.json")
    group.add_argument("--from-wandb", action="store_true")
    parser.add_argument("--download-dir", type=Path, default=Path("runs/resnet18-c3-split"))
    return parser


def _fetch_from_wandb(download_dir: Path) -> Path:
    import wandb

    run = wandb.Api().run(RESNET_TRAIN_RUN)
    for artifact in run.logged_artifacts():
        if artifact.type != "model":
            continue
        try:
            entry = artifact.get_entry("split.json")
        except KeyError:
            continue
        download_dir.mkdir(parents=True, exist_ok=True)
        path = Path(entry.download(root=str(download_dir)))
        print(f"reference: {artifact.qualified_name} → {path}")
        return path
    raise RuntimeError(f"{RESNET_TRAIN_RUN} 의 model artifact 에서 split.json 을 찾지 못했습니다")


def _summary(path: Path) -> tuple[dict, str]:
    """내용과 정규형 sha256.

    원본 `_recipe()` 가 set 순서로 키를 넣기 때문에 같은 split 이라도 파일 바이트는
    실행마다 다르다. 그래서 파일 해시가 아니라 키를 정렬한 정규형의 해시를 낸다.
    """
    data = json.loads(path.read_bytes())
    canonical = json.dumps(data, sort_keys=True).encode()
    return data, hashlib.sha256(canonical).hexdigest()


def main() -> int:
    args = _parser().parse_args()
    reference = args.reference or _fetch_from_wandb(args.download_dir)
    cand, cand_sha = _summary(args.candidate)
    ref, ref_sha = _summary(reference)
    print(f"candidate sha256 (정규형): {cand_sha}")
    print(f"reference sha256 (정규형): {ref_sha}")
    same = True
    for part in ("train", "validation"):
        c_ids = [r["id"] for r in cand[part]]
        r_ids = [r["id"] for r in ref[part]]
        ids_same = c_ids == r_ids
        records_same = cand[part] == ref[part]
        print(
            f"{part}: candidate {len(c_ids)} / reference {len(r_ids)}, "
            f"id 순서 일치={ids_same}, 레코드 전체 일치={records_same}"
        )
        if not ids_same:
            only_c = sorted(set(c_ids) - set(r_ids))[:5]
            only_r = sorted(set(r_ids) - set(c_ids))[:5]
            print(f"  candidate 에만: {only_c}  reference 에만: {only_r}")
        same &= ids_same and records_same
    print("결과:", "같음 ✅" if same else "다름 ❌ — 원인을 README 에 적고 공유한다")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
