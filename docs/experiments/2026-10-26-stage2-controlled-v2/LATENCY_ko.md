# 기존 네 조건의 시간 측정

seed42 체크포인트, A100, 50회 warmup 후 benchmark 1,500개를 3회 반복했다.
아래 값은 반복별 p50/p95의 중앙값(ms)이다. crop 경로는 탐지(Stage 1)를 제외한다.

| 모델·조건 | Stage 2 p50 / p95 | crop 정책 p50 / p95 | W&B |
|---|---:|---:|---|
| resnet18-c3 R | 2.627 / 3.172 | 29.540 / 168.933 | [run](https://wandb.ai/kim-1034/wemeet-barcode/runs/fxelrsmp) |
| resnet18-c3 M | 2.629 / 3.201 | 28.481 / 161.445 | [run](https://wandb.ai/kim-1034/wemeet-barcode/runs/qeibu49e) |
| mobilenetv3-small R | 4.042 / 4.642 | 29.313 / 153.426 | [run](https://wandb.ai/kim-1034/wemeet-barcode/runs/t1zgj93a) |
| mobilenetv3-small M | 4.155 / 4.883 | 29.184 / 139.255 | [run](https://wandb.ai/kim-1034/wemeet-barcode/runs/ua4zdwkk) |

이 A100 측정에서 MobileNet의 Stage 2 단독 지연시간은 ResNet보다 길었다.
따라서 MobileNet을 경량 모델이라는 이유만으로 더 빠르다고 해석하지 않는다.
MobileNet 채택 근거는 공통 조건에서의 정답률·복원 성능이며, 최종 처리시간은 전체 경로로 따로 확인한다.
이번 후속 실험은 최신 소스의 디코더 CLAHE off/on과 세 정책을 분리했으므로, 예전 crop 수치와 새 수치를 그대로 합쳐 평균내지 않는다.
