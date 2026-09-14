#!/usr/bin/env bash
# 제어점 격자 개수 실험 재현.
#
# 스파이크와 달리 이 실험은 리포 환경(wemeet 패키지)을 그대로 씁니다 --
# 측정 대상이 합성 파이프라인 본체이기 때문입니다. `uv sync` 가 먼저 필요합니다.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # 리포 루트

export PYTHONIOENCODING=utf-8          # 한글 출력이 깨지지 않게
CODE=docs/experiments/2026-09-09-control-points/code

run() { echo "=== $1 ==="; uv run python -u "$CODE/$1" "${@:2}"; }

# 00 이 먼저여야 합니다 -- 01·02 가 00_harvest.json 의 레시피에서 같은 표본을
# 다시 만듭니다 (recipe_from_dict + build).
run 00_harvest.py          # 표본 수확: 1차 실패 & G remap 성공, 버킷당 400
run 01_nx_sigma.py         # 실험 ①  n_x x 좌표오차 sigma
run 02_ny.py --nx-best 8   # 실험 ②  n_y.  8 은 ① 의 최적값이 sigma 에 따라
                           #             움직여 하나로 안 정해지기에 쓴 폴백값
run 03_g_resolution.py     # 표본 정의 타당성: 상한이 G 의 이산화 산물인가
run 05_grid.py             # 실험 ③  n_x x n_y x sigma 전수 (96셀, 약 4시간)
                           #   01·02 는 각각 n_y·n_x 단면만 봤다. 확정값은 여기서 나온다

# 04 는 위와 같이 돌리지 마십시오.
echo
echo "비용 측정은 반드시 단독으로:"
echo "    PYTHONIOENCODING=utf-8 uv run python $CODE/04_cost.py"
echo "  병렬로 재면 같은 조건이 3.4ms -> 22.8ms 로 7배 부풀려집니다 (인계 문서 실측)."
