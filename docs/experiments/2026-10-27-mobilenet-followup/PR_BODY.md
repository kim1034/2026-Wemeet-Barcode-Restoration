기존 ResNet/MobileNet의 서로 다른 학습 조건을 공통 head와 동일 데이터에서 R/M 두 preset, seed42/43/44로 교차 비교한 결과를 기록합니다. 선택된 MobileNet M에 대해 판독 정책, 전체 장면 경로, 탐지 해상도, 실패 분석을 추가했습니다.

오독 억제 우선 후보는 imgsz 1280 · CLAHE on · 항상 보정이며, 합성 장면 100개에서 정답 81개/오독 0개, 평가 경로 p95 256.50ms입니다. 정답률 우선 후보는 imgsz 1600 · CLAHE on · 현재 조기 종료, 정답 90개/오독 4개, p95 284.70ms입니다. 두 설정을 모두 운영 검증 후보로 유지합니다. 추가 학습이나 제품 기본 정책 변경은 포함하지 않습니다.

- 정확도 12개 및 직렬 시간 측정 10개, W&B 저장 파일 대조 완료.
- 원시 판독/분모/집계/hash 검사, 장면 800건 native 파이프라인 대조 완료.
- pytest 342 passed/11 skipped, ruff 및 import-linter 통과.
- 합성 개발셋에서 선택한 운영 후보이며 별도 실촬영 holdout 검증이 필요합니다.

읽을 문서: `docs/experiments/2026-10-27-mobilenet-followup/DECISION_ko.md`, `NOTION_ko.md`, `REPORT_ko.md`, `EXECUTION_ko.md`.
