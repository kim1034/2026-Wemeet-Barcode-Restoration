# E2E 벤치마크 기록

**고정된 사진 100장**으로 파이프라인을 돌릴 때마다 결과를 `runs.jsonl` 에 한 줄씩 쌓는다.
마지막 결과보고서는 이 기록으로 만든다.

## 보고서가 답할 두 질문

1. **단계를 붙일수록 얼마나 좋아지나** — 같은 100장에 네 조건
2. **우리 서비스 없이 디코딩만 할 때 vs 우리 서비스** — 조건 A vs D

| 조건 | 무엇 |
|---|---|
| A | 원본 사진 전체를 그대로 `decode()` — **우리 서비스 없이** |
| B | + 1단계 검출 크롭 (1차 디코딩) |
| C | + 2·3단계 기하 보정 1회 (배율 1.0) |
| D | + 배율 재시도 = `run()` 전체 — **우리 서비스** |

판독 성공 = 정답 번호와 **완전 일치**. 틀린 번호는 `misread`(오독)로 따로 센다.

## 세트 — `benchmark-v1`

| 구간 | 뜻 | 장수 |
|---|---|---|
| `first_ok` | 크롭을 그대로 디코딩하면 읽힘 | 15 |
| `target` | 그대로는 안 읽히고, 정답 제어점으로 펴면 읽힘 | 80 |
| `hard` | 정답 제어점으로 펴도 안 읽힘 | 5 |

해상도 low/mid/high 에 고르게 나눴다 (`scripts/make_benchmark.py` 의 `PER_BAND`).
seed 2026 — 학습(42)·평가(7)와 다른 시드다.

> **이 구성은 현장 비율이 아니라 팀이 정한 것이다.** 보고서의 수치는 "이 100장에서"의
> 수치이고, 목표 구간이 80% 라 A 가 낮게 나오는 것은 구성 탓이 크다. 보고서에 구성을
> 같이 적는다 (`docs/decisions/0005` 문제 2).

이미지는 HF `123metro/barcode-datasets` 의 `benchmark/v1/` 에 있다. **다시 만들지 않는다**
— 세트가 바뀌면 이전 기록과 비교할 수 없다.

```bash
hf download 123metro/barcode-datasets --repo-type dataset --include "benchmark/v1/*" --local-dir downloads/hf
# downloads/hf/benchmark/v1 을 --data-root 로 준다
```

## 돌리는 법

```bash
uv run python -m scripts.run_benchmark --data-root downloads/hf/benchmark/v1 --note "무엇을 바꿨나"
```

- `--note` 는 필수다. 나중에 "이 줄에서 왜 올랐나"를 알 수 있는 건 이것뿐이다
- 기록 한 줄에 커밋 해시와 `dirty` 가 들어간다. **`dirty: true` 인 줄은 재현할 수 없다** —
  보고서에 쓸 실행은 커밋한 뒤에 돌린다
- 시험 삼아 돌릴 때는 `--no-log`
- 샘플별 결과는 `runs/benchmark/` 에 남는다 (git 제외)
- `latency_ms_D` 는 돌린 PC 의 시간이다. 줄마다 다른 PC 면 비교하지 않는다
