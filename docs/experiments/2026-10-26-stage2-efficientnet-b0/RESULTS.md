# 측정표 전문 — Stage 2 EfficientNet-B0-C3 vs ResNet18-C3

실행: 2026-10-28 ~ 2026-10-29. 데이터 revision `a833a3a5944148cb9815ccbd1f05aced50f6c464`
(`synthetic/v1`), seed 42, split은 ResNet 실행과 동일함을 `check_split.py`로 확인
(`runs/restoration-efficientnet-b0-c3-v1/split.json` == `runs/resnet18-c3-split/split.json`).

## 1. 핵심 표 (README 6장)

| 모델 | params | 정답 / 오독 / 실패 (1,500장) | 구제율 | 오독률 | low / mid / high 구제율 | val 좌표 오차 mean / P95 (px) | CPU p50 / p95 (1스레드, ms) | best epoch | W&B |
|---|---:|---|---:|---:|---|---|---|---:|---|
| ResNet18-C3 (재평가) | 3,217,538 | 1,067 / 7 / 426 | 71.13% | 0.47% | 67.6% / 73.4% / 72.4% | 1.3723 / 3.7989 | 66.96 / 82.37 | 48 | [학습](https://wandb.ai/kim-1034/wemeet-barcode/runs/p9eghdt2) · [재평가](https://wandb.ai/kim-1034/wemeet-barcode/runs/7i4f4lbw) |
| EfficientNet-B0-C3 | 1,249,698 | 1,180 / 11 / 309 | 78.67% | 0.73% | 74.6% / 81.0% / 80.4% | 0.9122 / 2.6063 | 21.21 / 32.80 | 48 | [학습](https://wandb.ai/kim-1034/wemeet-barcode/runs/3clzz2wx) · [평가](https://wandb.ai/kim-1034/wemeet-barcode/runs/fh6sgirp) |
| **차이 (EfficientNet − ResNet)** | **−61.2%** (−1,967,840) | ok +113 / 오독 +4 / 실패 −117 | **+7.53%p** | **+0.27%p** | +7.0 / +7.6 / +8.0 %p | −0.4601 / −1.1926 | **−45.75 / −49.57** (약 3.2배 빠름) | 동일 | |

- 구제율·오독률·실패율은 평가 JSON(`ok`/`misread`/`decode_failed`, 1,500장 중 비율)에서 그대로 계산했다.
- ResNet18-C3 재평가 숫자(1,067/7)는 완료 체크리스트에 적힌 기존 기준값과 일치한다 — 두 모델이 같은 평가 경로를 탔다는 확인.
- val 좌표 오차는 각 모델의 **best epoch(48) 체크포인트** 기준 W&B `best/val_control_point_mean_px` · `best/val_control_point_p95_px`다. ResNet 값은 [2026-09-21 보고서](../2026-09-21-stage2-resnet18-c3/README.md#결과)에서 가져왔다(같은 trainer, 배치 단위 평균 집계 방식).
- params는 `sum(p.numel() for p in model.parameters())` — backbone + adapter + head 전체(평가 스크립트 출력 그대로).
- CPU 지연은 같은 컴퓨터, 같은 측정 코드(`latency.py`, threads=1, batch=1, warm-up 20회, 측정 200회, 입력 `(1,3,160,384)`)로 쟀다. CPU: `Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz`.

## 2. 버킷별 상세 (평가 1,500장 = low/mid/high 각 500장)

| 모델 | 버킷 | n | 정답 | 오독 | 디코드 실패 | 구제율 | 오독률 | 실패율 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18-C3 | low | 500 | 338 | 2 | 160 | 67.6% | 0.4% | 32.0% |
| ResNet18-C3 | mid | 500 | 367 | 1 | 132 | 73.4% | 0.2% | 26.4% |
| ResNet18-C3 | high | 500 | 362 | 4 | 134 | 72.4% | 0.8% | 26.8% |
| EfficientNet-B0-C3 | low | 500 | 373 | 5 | 122 | 74.6% | 1.0% | 24.4% |
| EfficientNet-B0-C3 | mid | 500 | 405 | 3 | 92 | 81.0% | 0.6% | 18.4% |
| EfficientNet-B0-C3 | high | 500 | 402 | 3 | 95 | 80.4% | 0.6% | 19.0% |

- 두 모델 모두 **low 버킷이 가장 약하다** (해당 버킷의 원래 의도와 달리 high가 아니라 low가 최저 구제율). 버킷 간 격차는 EfficientNet에서도 그대로 유지된다(평균 ~6~7%p 낮음).
- 세 버킷 모두 EfficientNet이 ResNet보다 +7.0~+8.0%p 높다 — 특정 버킷에 치우친 개선이 아니라 전체적으로 균일하게 나타났다.
- 오독은 ResNet 1건(mid), 4건(high) → EfficientNet 5건(low), 3건(mid), 3건(high)로 low 버킷에서 집중적으로 늘었다.

## 3. 학습 곡선 요약 (W&B epoch별 값에서)

| 모델 | best epoch | best val loss | best val mean px | best val RMSE px | best val P95 px | 총 학습 시간 | GPU |
|---|---:|---:|---:|---:|---:|---|---|
| ResNet18-C3 | 48 | 0.0010065 | 1.3723 | 2.7068 | 3.7989 | 17h 40m 20s | A100-PCIE-40GB ×1 |
| EfficientNet-B0-C3 | 48 | 0.00037953 | 0.9122 | 1.2560 | 2.6063 | 17h 28m 54s | A100 80GB PCIe ×1 |

- epoch 50(마지막)에서는 EfficientNet val mean/P95가 0.9451/2.6540px로 48 epoch보다 살짝 높아져(ResNet과 같은 경향으로) `best.pt`(48 epoch)를 평가·지연 측정에 사용했다.
- GPU 모델명(40GB vs 80GB PCIe)이 다르지만 같은 A100 세대이고 둘 다 1장만 사용했다. 학습 시간 차이(약 11분)는 이 차이로 보지 않는다.

## 4. 재현 기록

| 단계 | 산출물 | 비고 |
|---|---|---|
| 짧은 시험 실행 | `runs/smoke-efficientnet-b0-c3/` | 1 epoch, 100/70/30 표본, 평가 20장/버킷. 끝까지 돎 확인용, 표에는 반영 안 함 |
| split 확인 | `runs/restoration-efficientnet-b0-c3-v1/split.json` | `runs/resnet18-c3-split/split.json`과 완전히 동일 (`check_split.py`) |
| 본 학습 | `runs/restoration-efficientnet-b0-c3-v1/best.pt`, `last.pt` | W&B `restoration-efficientnet-b0-c3-v1` ([3clzz2wx](https://wandb.ai/kim-1034/wemeet-barcode/runs/3clzz2wx)) |
| 평가 (EfficientNet) | `runs/restoration-efficientnet-b0-c3-v1/eval.json` | W&B [fh6sgirp](https://wandb.ai/kim-1034/wemeet-barcode/runs/fh6sgirp). 이전 시도([fxhdf5q9](https://wandb.ai/kim-1034/wemeet-barcode/runs/fxhdf5q9), ok=1,171)는 최종본이 아니라 폐기 |
| 평가 (ResNet 재평가) | `runs/restoration-efficientnet-b0-c3-v1/resnet18-c3-reeval.json` | W&B [7i4f4lbw](https://wandb.ai/kim-1034/wemeet-barcode/runs/7i4f4lbw). 이전 시도([3lmsnd89](https://wandb.ai/kim-1034/wemeet-barcode/runs/3lmsnd89), ok=1,059)는 최종본이 아니라 폐기 |
| 지연 측정 | `runs/restoration-efficientnet-b0-c3-v1/latency.json` | W&B [rfi3fhx2](https://wandb.ai/kim-1034/wemeet-barcode/runs/rfi3fhx2) |

이전 평가 시도(`fxhdf5q9`, `3lmsnd89`)가 최종 숫자와 다른 이유는 재확인하지 않았다 — 평가 스크립트
재실행 중 체크포인트 로딩 조건이 달랐을 가능성이 있다. 최종 숫자는 로컬 `eval.json` /
`resnet18-c3-reeval.json` 파일 및 가장 마지막 W&B run과 모두 일치하는 쪽을 채택했다.

## 5. 결론 요약

7장 기준 읽기: [README.md 「결과」](README.md#결과) 참고.
