# MobileNet·ResNet 동일 조건 비교 결과

12개 학습, 24개 선택 체크포인트 평가, 12개 paired 비교 완료.
수치는 3개 seed의 평균 ± 표준편차이며 비율의 단위는 %다.

| 모델 | 조건 | 데이터 | 정답률 | 오독률 | RR_H | EPE(px) |
|---|---|---|---:|---:|---:|---:|
| resnet18-c3 | R | benchmark | 84.7556 ± 0.6842 | 2.8667 ± 0.2906 | 84.4550 ± 0.6977 | 2.5355 ± 0.0902 |
| resnet18-c3 | R | validation | 63.8105 ± 0.2352 | 1.8705 ± 0.1940 | 56.7990 ± 0.2808 | 2.3886 ± 0.0650 |
| resnet18-c3 | M | benchmark | 86.0444 ± 0.3791 | 2.6444 ± 0.1388 | 85.7693 ± 0.3866 | 2.4459 ± 0.1187 |
| resnet18-c3 | M | validation | 64.3593 ± 0.2936 | 1.8593 ± 0.0513 | 57.4542 ± 0.3505 | 2.2759 ± 0.1133 |
| mobilenetv3-small | R | benchmark | 88.2889 ± 0.4286 | 2.5778 ± 0.2341 | 88.0580 ± 0.4371 | 2.1908 ± 0.0204 |
| mobilenetv3-small | R | validation | 65.2778 ± 0.1851 | 1.9489 ± 0.1465 | 58.5506 ± 0.2209 | 2.1518 ± 0.0378 |
| mobilenetv3-small | M | benchmark | 88.4667 ± 0.9018 | 2.5556 ± 0.3791 | 88.2393 ± 0.9196 | 2.0620 ± 0.0199 |
| mobilenetv3-small | M | validation | 66.1178 ± 0.2858 | 1.8593 ± 0.0388 | 59.5534 ± 0.3412 | 2.0184 ± 0.0370 |

R: lr 1e-4 / batch 8 / 50 epoch / beta 0.02.
M: lr 3e-4 / batch 64 / 100 epoch / beta 1/154.

같은 조건 내에서 모델을 비교한다. 기존 합성 개발셋이며 새 독립 test가 아니다.
MobileNet은 공통 head를 적용한 새 구현이다. GPU 재개의 비트 단위 일치는 보장하지 않는다.
세부 결과와 W&B 링크는 results.csv, paired 구간은 comparisons/에 있다.
시간 측정은 summary.json에 있다.

> 공유본 안내: 위 내용은 기존 완료 보고서다. 시간 측정 요약은 [LATENCY_ko.md](LATENCY_ko.md), 원시 반복 시간과 당시 summary.json은 [기존 W&B 결과의 artifact](https://wandb.ai/kim-1034/wemeet-barcode/runs/n13l9dno)에서 확인한다.
