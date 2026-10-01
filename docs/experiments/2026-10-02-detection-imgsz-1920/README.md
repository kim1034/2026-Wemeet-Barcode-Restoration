# Stage 1 탐지 입력 크기 — `DEFAULT_IMGSZ` 960 → 1920 · 2026-10-02

**결론: 960으로 축소하면 저해상도 막대(1.6~2.2px)가 1px 밑으로 뭉개져 탐지 자체가
실패한다.** 재학습 없이 추론 `imgsz` 만 **1920(원본, 축소 없음)** 으로 올리면 같은
가중치로 탐지 69→99/100, 전체 판독(조건 D) 52→89/100 으로 개선된다. 원인 진단은
[`eval/benchmark-v2` 기록](https://github.com/kim1034/2026-Wemeet-Barcode-Restoration/tree/eval/benchmark-v2/docs/experiments/2026-09-30-e2e-benchmark-v2)
과 같지만, 그 기록이 제안한 **1280~1600** 이 아니라 **1920** 을 이 기록은 제안한다 — 지금은
베타 단계라 컨베이어에 바코드가 **한 번에 한 장만** 지나가고 동시 처리 부하가 없어, 속도보다
정확도를 우선해도 되는 시기라고 판단했다. GPU 운영 서버가 정해지면 — 또는 베타를 넘어 여러
장이 동시에 들어오는 정식 운영으로 갈 때는 — 반드시 재검증해야 한다 (「한계」 참고).

> 전부 합성 데이터이고, **GPU 없는 PC(CPU)** 에서 쟀다. 시간은 운영 예산과 1:1로
> 비교하지 않는다 (`docs/benchmark/README.md` 와 같은 원칙).

## 배경

[`wemeet/ai/detection/yolo_obb.py`](../../../wemeet/ai/detection/yolo_obb.py) 의
`DEFAULT_IMGSZ` 주석은 원래 "데이터셋이 1280x720 이라 960 이면 화질 손실이 거의 없다"
였다. 그런데 실제 배포된 가중치(`123metro/barcode-weights`,
`detection/v2-yolo-obb/best.pt`) 를 열어 ultralytics 가 체크포인트에 직접 기록한
`train_args` 를 확인하면:

```
imgsz: 960
scale: 0.5
```

**학습도 960으로 했다** (`docs/experiments/2026-09-22-stage1-detection/code/train_detection.py`
의 `--imgsz` argparse 기본값은 1280이지만, 실제 실행한 명령은 `--imgsz 960` 으로
덮어썼다 — 문서·기본값과 실제 학습 기록이 다를 수 있다는 걸 보여주는 사례라 체크포인트를
직접 열어 확인했다). 재현: [`code/inspect_checkpoint_train_args.py`](code/inspect_checkpoint_train_args.py).

즉 960은 추측이 아니라 실제 학습값과 일치한다. 그런데도 저해상도 구간에서 탐지가
크게 실패하는 걸 아래에서 실측했다 — **"학습값과 같다"가 "실전에서 맞는 값"을
보장하지 않는다**는 사례다.

## 결과 1 — imgsz별 탐지 성공률 (benchmark-v1, low 버킷 34장)

| imgsz | 탐지 성공 |
|---|---|
| 960 | 14/34 (41%) |
| 1280 | 21/34 (62%) |
| 1920 | 32/34 (94%) |

재현: [`code/sweep_imgsz_accuracy.py`](code/sweep_imgsz_accuracy.py)
(`--data-root downloads/benchmark-v1 --imgsz 960 1280 1920`).

## 결과 2 — 버킷별 탐지 성공률 (benchmark-v2, 100장)

막대 폭(`d_m0`) 실측: low 1.60~2.17px(평균 1.89) · mid 2.59~3.45px(평균 3.04) ·
high 4.07~5.95px(평균 5.18) — `docs/decisions/0004` 가 말하는 "모듈 1px 이하는
복원 불가" 구간에 low 가 가장 가깝다.

| 버킷 | 전체 | 960 (이전) | 1920 (현재) |
|---|---|---|---|
| low | 34 | 17/34 (50%) | **34/34 (100%)** |
| mid | 34 | 25/34 (74%) | **34/34 (100%)** |
| high | 32 | 27/32 (84%) | 31/32 (97%) |
| **전체** | **100** | **69/100 (69%)** | **99/100 (99%)** |

**어느 버킷도 나빠지지 않았다** — low 를 고치다 mid/high 를 깨뜨리는 일은 없었다.
1920 에서도 1장은 여전히 실패한다 (high 버킷) — 해상도가 아닌 다른 원인(심한 손상 등)
으로 추정되며 따로 확인이 필요하다.

재현: 위와 같은 스크립트, `--data-root downloads/hf/benchmark/v2 --imgsz 960 1920`.

## 결과 3 — 전체 판독 성공률, A vs D (benchmark-v2, imgsz=1920 반영 후)

`docs/benchmark/README.md` 의 조건 정의를 따랐다 (A = 원본 사진 + pyzbar 단독,
D = `run()` 전체).

| | A: 원본 + pyzbar 단독 | D: 우리 파이프라인 전체 |
|---|---|---|
| 판독 성공 | 52/100 (52%) | **89/100 (89%)** |
| 오독(틀린 번호) | 0 | 1 |
| 실패 | 48 | 10 |
| 시간 p50 / p95 (CPU) | 151ms / 209ms | 334ms / 895ms |

재현: [`code/pipeline_vs_pyzbar.py`](code/pipeline_vs_pyzbar.py).

## 결과 4 — 속도 (CPU)

탐지만 (`code/detect_latency.py`, benchmark-v2 100장):

| | avg | p50 | p95 | min~max |
|---|---|---|---|---|
| 960 | 78ms | 75ms | 104ms | 61~119ms |
| 1920 | 345ms | 336ms | 468ms | 248~527ms |

(1920/960)² ≈ 4배와 거의 일치하는 4.4~4.5배 증가 — 예상한 그대로다.

전체 파이프라인 (`code/pipeline_latency.py`, imgsz=1920, benchmark-v2 100장):

```
avg=546ms  p50=438ms  p95=1011ms  min=234ms  max=5173ms
  300ms 초과: 87/100장    500ms 초과: 26/100장
  700ms 초과: 18/100장    900ms 초과: 8/100장
```

p50 은 탐지만 쟀을 때(336ms)와 거의 같다 — **보통 경우 늘어난 시간은 거의 전부 탐지
때문**이다. 하지만 p95(1011ms)는 탐지만(468ms)보다 훨씬 크다 — 어렵게 휘거나 손상된
사진일수록 기하추정·보정·재시도(최대 3회)가 추가로 지연을 쌓는다. max=5173ms 는
나머지 분포와 너무 떨어져 있어 이 PC의 다른 프로세스 간섭 등 측정 잡음일 가능성이 있다.
**같은 조건으로 다시 재도 p50=334ms, p95=895ms 로 수치가 달랐다** (결과 3) — CPU
측정 자체의 변동폭이 크다는 뜻이며, 절대값보다 "탐지 외 단계도 무시 못할 지연을
만든다"는 경향으로만 받아들여야 한다.

## 수용장(receptive field) 미스매치는 왜 반대로 작용했나

학습(960)과 다른 크기(1920)로 추론하면, 모델의 고정된 수용장이 물체를 보는 상대적
비율이 학습 때와 달라진다는 지적이 있었다 — 이론적으로 타당하고, 실제로 그 방향의
효과는 존재한다 (960 에서는 바코드가 모델 입력에서 더 작게, 1920 에서는 더 크게 보여
수용장 대비 비율이 달라진다). 그런데도 1920 이 압도적으로 더 잘 나온 이유:

1. **정보 손실이 더 치명적이다.** 막대가 1.6~2.2px인데 960 으로 줄이면 0.8~1.1px —
   1px 밑으로 내려간 정보는 어떤 수용장 크기로도 복원이 안 된다. 1920 은 그 정보를
   보존한다.
2. **YOLO는 단일 수용장이 아니다.** P3/P4/P5 다중 스케일 헤드가 있어 물체가 상대적으로
   커 보여도 더 넓은 수용장을 가진 head 가 받을 수 있다 — 미스매치를 어느 정도 흡수한다.
3. **이론과 실측이 다르면 실측을 따른다.** 결과 1·2가 같은 가중치·같은 사진으로 직접
   측정한 값이다.

다만 1920 에서도 100% 가 아니라 99% 인 점, 오독이 0→1 로 생긴 점은 이 미스매치가
완전히 사라진 게 아니라 정보 보존 효과에 눌려 있을 뿐이라는 신호로 보인다.

## `eval/benchmark-v2` 팀원 기록과의 비교

원인 진단("입력 축소가 원인")은 동일하다. 다만 그 기록은:

- **1280~1600** 을 제안한다 — "입력 크기별 시간을 GPU에서 재고 예산(110ms) 안의
  최대값을 고른다"는 기준
- **오독 증가를 이미 알고 있고, 디코딩 검증(4단계: 후보 불일치 시 거부, 번호 형식
  검사)을 같이 넣을 것을 권한다** — imgsz 변경 단독으로 내보내는 것을 권하지 않는다

이 기록이 1920을 미는 이유는 수치가 달라서가 아니라 **전제가 다르기 때문**이다 —
지금은 베타라 동시 처리 부하가 없다는 점(위 결론 참고)이 속도 손해를 감수할 여지를
만든다. 디코딩 검증은 이 기록의 범위 밖이며, 별도로 진행해야 한다 (「한계」).

## 한계

- **CPU 전용 측정이다.** GPU 환경(RTX4050 기준 Stage 1 보고값 4.1ms)에서는 숫자가
  완전히 달라진다 — 운영 서버가 확정되면 반드시 재측정해야 한다.
- **측정 자체의 변동폭이 크다** (결과 4, p50 334ms vs 438ms). 추세로만 쓴다.
- **베타(순차 1장 처리) 가정에 기반한 권고다.** 여러 장이 동시에 들어오는 정식
  운영에서는 처리량(큐 적체) 문제가 별도로 생길 수 있다 — 이 기록은 다루지 않는다.
- **오독 증가(0→1)에 대한 대응(디코딩 검증)이 아직 코드에 없다.** `eval/benchmark-v2`
  가 제안한 4단계 수정(후보 불일치 거부, 형식 검사)을 별도 작업으로 뒤따라야 한다.
- 표본이 작다 (100~132장). 1~2장 차이는 우연 범위일 수 있다.
- zbar 콘솔에 뜨는 "pdf417 Assertion failed" 경고는 다른 심볼로지 탐색 실패일 뿐
  결과에 영향 없다.

## 권고

1. **지금 당장**: `DEFAULT_IMGSZ` 를 960 → 1920 으로 올린다 (베타 범위, 재학습 없음).
2. **병행(SW)**: `eval/benchmark-v2` 가 제안한 디코딩 검증(후보 불일치 거부, 번호
   형식 검사)을 별도로 진행 — 오독 리스크를 상쇄한다.
3. **정식 운영 전**: 운영 서버(GPU/CPU)가 확정되면 이 값을 GPU 기준으로 재검증하고,
   여러 장 동시 처리 시나리오의 처리량도 같이 확인한다.

## 재현

```bash
uv run python docs/experiments/2026-10-02-detection-imgsz-1920/code/inspect_checkpoint_train_args.py
uv run python docs/experiments/2026-10-02-detection-imgsz-1920/code/sweep_imgsz_accuracy.py \
    --data-root downloads/benchmark-v1 --imgsz 960 1280 1920
uv run python docs/experiments/2026-10-02-detection-imgsz-1920/code/sweep_imgsz_accuracy.py \
    --data-root downloads/hf/benchmark/v2 --imgsz 960 1920
uv run python docs/experiments/2026-10-02-detection-imgsz-1920/code/pipeline_vs_pyzbar.py \
    --data-root downloads/hf/benchmark/v2
uv run python docs/experiments/2026-10-02-detection-imgsz-1920/code/detect_latency.py \
    --data-root downloads/hf/benchmark/v2
uv run python docs/experiments/2026-10-02-detection-imgsz-1920/code/pipeline_latency.py \
    --data-root downloads/hf/benchmark/v2
```

`downloads/hf/benchmark/v2` 는 `docs/benchmark/README.md` 의 안내대로 받는다:

```bash
hf download 123metro/barcode-datasets --repo-type dataset --include "benchmark/v2/*" --local-dir downloads/hf
```
