#!/usr/bin/env bash
# Stage 2 합성 스파이크 재현.  리포 환경(pyproject.toml)을 건드리지 않는다.
set -euo pipefail
cd "$(dirname "$0")/code"

DEPS=(--with numpy --with opencv-python-headless --with python-barcode
      --with pillow --with zxing-cpp)

run() { echo "=== $1 ==="; uv run --no-project --python 3.12 "${DEPS[@]}" python -u "$1"; }

run common.py            # 표본판 — §2 수식이 만드는 그림 (results.json)
run 02_direction.py    # cumsum(1/m) 이 해석해와 맞는가
run 03_flip_and_nx.py  # §8.5 방향 함정 · 제어점 개수
run 04_dm0_grid.py     # 모듈 폭 x 각도 (기울기 예산 켬)
run 05_true_limits.py  # 예산 끄고 진짜 한계
run 06_section3.py     # 행별 정규화 · 정반사 · 구간 비율
run 07_knobs.py        # 목표 구간을 넓히는 손잡이
run 08_locality.py     # 날카로움 x 각도
run 09_targeted.py     # 겨냥 샘플링
run 10_resolution.py   # 해상도 구간별  (난수 버그 있음 — 12 참조)
run 11_sine_limit_cospsi.py   # 물결 상한 A/λ · cos(ψ) 보정
run 12_rerun_rng_fixed.py     # 09·10 의 난수 버그 수정 재측정  <- 이 숫자를 쓸 것
run 13_augment_G.py           # 밀집 대응장 G 방식 증강
run 14_coord_consistency.py   # 좌표 정합성 (펴진 이미지 비교)
run 15_budget_and_optics.py   # 접힘 tanα · 옥타브 · clip 의 법선각 오차 · 조립 순서
run 16_perf_and_contract.py   # 재현성 · 생성 속도 · tps_flow 시간 · 리사이즈 불변성
run 17_eval_set.py            # 평가 세트 실현성 · confidence 신호
run 18_redo.py                # 15·16 의 셋업 결함 재실험  <- 이 숫자를 쓸 것

# 주의: 타이밍(16·18)은 반드시 단독 실행으로 잴 것.
#       병렬로 돌리면 같은 조건이 3.4ms -> 22.8ms 로 7배 부풀려진다.

echo; echo "시각 보고서를 만들려면:  uv run ... python make_page.py  (results*.json 필요)"
