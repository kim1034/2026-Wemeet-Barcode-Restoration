# MobileNet 후속 실험과 기존 네 조건 비교

기존 ResNet/MobileNet × R/M의 네 조건 비교와 선택된 MobileNet 후속 평가를 공유하는 폴더다.
결과 생성 전에는 표의 값이나 완료를 추정해서 적지 않는다. 완료 보고서는 원시 기록을 검사한 뒤 생성된다.

## 읽는 순서

1. `DECISION_ko.md`: 오독 억제와 정답률 우선의 두 운영 검증 후보. 두 설정을 모두 유지한다.
2. `NOTION_ko.md`: Notion에 가져오거나 붙여 넣을 통합 문서.
3. `REPORT_ko.md`: 기존 네 조건, 정책·해상도·디코더 비교, 실제 지연시간, 실험 한계.
4. `results.csv`, `timing.csv`: 전체 수치와 W&B run 링크.
5. `failure_summary.csv`, `failure_cases.csv`: 실패 유형 및 모든 오독/무응답 샘플.
6. `paired_comparisons.json`: 동일 입력의 paired 변화와 payload 그룹 bootstrap 구간.
7. `provenance.json`, `validation.json`: 입력 버전, 가중치 hash, 전체 집계 검증.
8. `FAILURE_ANALYSIS_ko.md`, `EXECUTION_ko.md`: 실패 사례 해석과 실행 중 수정 기록.

기존 3-seed 결과는 [이전 결과 폴더](../2026-10-26-stage2-controlled-v2/REPORT_ko.md)에 보존한다.
기존 모델별 비교와 후속 단일 모델 정책 비교는 반복 횟수와 분모가 달라 구분해서 읽는다.

## 고정한 조건

- 기반 upstream: `f449b9c34df3a42c5c324a0629b0dbee3308f1b3`.
- MobileNetV3-Small M/seed42/epoch90, ResNet18-C3 M/seed42/epoch90.
- 모델 재학습 없이 선택 체크포인트를 엄격하게 로딩한다.
- GPU 6·7 각각 하나의 프로세스만 사용하며 실제 NVIDIA UUID/PID를 기록한다.
- crop validation 2,976개, crop benchmark 1,500개, scene benchmark/v2 100개.
- 정책: 현재 조기 종료, 항상 보정, 판독 일치 확인.
- 디코더: zxing-cpp, CLAHE 재시도 off/on을 별도 비교. 탐지 전 CLAHE는 공통 적용.
- MobileNet 탐지 imgsz 1280/1600/1920, ResNet 대조군 imgsz1280.
- 정확도 12개 작업을 병렬로 실행한 뒤 시간 측정 10개를 직렬 실행한다.
- 시간 측정은 50회 warmup + 전체 입력 3회 반복, CPU thread 1, FP32/TF32 off.

시간은 평가용 정책 실행기 기준이며 판독값·형식 추적과 이미지 hash 계산을 포함한다.
제품 UI/카메라 캡처/파일 읽기/모델 최초 로딩은 포함하지 않는다.
최종 서비스의 지연시간을 주장하려면 해당 서비스에서 별도 측정해야 한다.

정확도 감사에서는 D0와 모든 warp를 관찰하고 세 정책의 결정을 계산한다.
시간 측정에서는 각 정책에 필요한 경로만 실제 실행한다.

- 현재 정책: D0가 어떤 문자열이든 반환하면 종료하고, 아니면 모델과 warp 재시도를 실행한다.
- 항상 보정: 사용하지 않는 D0 판독을 생략하고 모델→warp 재시도를 실행한다.
- 일치 확인: D0와 세 warp를 모두 판독한 다음 일치 여부를 판단한다.

`scripts/mobilenet_followup/core.py`의 판독 함수는 GT를 전달받지 않는다.
오독은 정답과 다른 유효 문자열을 반환한 경우이며, 무응답과 별도 집계한다.
일치 확인은 서로 다른 이미지에서 같은 판독을 두 번 이상 얻고 상충 결과가 없을 때 채택한다.
같은 오독이 반복되는 경우까지 제거한다는 보장은 없다.
`RR_H`의 분모는 D0가 정답이 아닌 입력이다. CLAHE off/on에서는 이 모집단도 달라지므로,
두 모드의 RR_H 차이를 동일한 샘플 집합의 개선량으로 읽지 않는다.

## 코드 구조와 실행 기록

- `prepare.py`: 공개 HF 데이터·탐지 가중치 revision 고정, 기존 결과 보존 manifest.
- `admission.py`: 실제 두 GPU의 소규모 수치/판독 재현과 native pipeline 대조.
- `freeze.py`: 입력 및 실행 코드 hash 고정.
- `controller.py`: 단일 writer lock, 두 평가 worker, 순차 시간 측정, 최종 집계.
- `worker.py`: 원시 추론·모든 warp 판독·전이·시간·GPU 관찰·W&B artifact.
- `report.py`: 표본 ID/GT, 판독 정책, 정답/오독/무응답 산술, 전이, 시간 quantile 재검사.

이 실행의 로컬 기록은 `runs/mobilenet-followup-20261027/`에 있다.
`inputs.json`과 `execution_lock.json`은 실행 당시 서버의 절대 경로를 포함한다.
`jobs/<task>/attempt-XX/`는 부분 실패도 보존한다. 검증을 통과한 작업만 `completion.json`이 생긴다.
계산이 완료된 뒤 W&B 업로드만 실패하면 같은 결과를 검사해 다시 업로드하며 추론을 반복하지 않는다.
최종 `completion.json`은 22개 작업의 완료·집계 검증과 통합 W&B artifact 업로드 후 생성된다.

운영 service: `wemeet-mobilenet-followup-gpu67-20261027.service`.
SSH나 대화 연결과 독립적인 user systemd service이며, 사용자 lingering이 활성화되어 있다.
일시적인 프로세스 종료 시 완료 결과를 검사하고 이어간다. 검증 오류나 반복 실패는 보존하고 중단한다.
다른 사용자의 GPU 프로세스는 종료하지 않는다. 완료한 서비스를 다시 실행하지 않는다.

## 재현 준비

새 환경에서는 이 보고서의 가중치 hash와 HF revision을 먼저 확인한다.
현재 서버의 구성은 Python 3.12.3, torch 2.8.0+cu126, torchvision 0.23.0+cu126,
NumPy 2.5.1, OpenCV 5.0.0.93, zxing-cpp 3.1.1, wandb 0.28.1, ultralytics 8.4.190이다.
그림 생성에는 matplotlib 3.11.2, python-dateutil 2.9.0.post0, six 1.17.0을 사용했다.
`pyproject.toml`이나 기존 공용 환경은 변경하지 않았고, 추가 패키지는 별도 실험 환경에 설치했다.

1. 기존 [W&B 통합 결과](https://wandb.ai/kim-1034/wemeet-barcode/runs/n13l9dno)와
   [MobileNet 학습](https://wandb.ai/kim-1034/wemeet-barcode/runs/ypolniyl),
   [ResNet 학습](https://wandb.ai/kim-1034/wemeet-barcode/runs/btfvvkwr)의 artifacts를 받는다.
   원래 split/protocol, 선택 checkpoint, 기존 crop 원시 평가 기록이 재현 검사에 필요하다.
2. `123metro/barcode-datasets`의 crop revision
   `a833a3a5944148cb9815ccbd1f05aced50f6c464`, scene revision
   `41bd506fa5e9a5aac33f202d2f9ce98acc77d4f3`을 사용한다.
   `123metro/barcode-weights`의 탐지 가중치 revision/hash는 `provenance.json`에 있다.
3. 새 출력 폴더와 해당 서버의 물리 GPU UUID를 명시한 새 inputs를 작성한다.
   기존 완료 폴더나 seal을 수정하지 않는다. 원본 프로토콜 및 체크포인트 hash는 유지한다.
4. `admission.py`의 실제 데이터 검사에 통과한 뒤 새 실행 lock과 task matrix를 고정한다.
5. native 판독 대조 수정까지 포함한 `python -m scripts.mobilenet_followup.native_bridge controller
   --root <새 출력 폴더>`를 사용한다. amendment 파일의 기존 실행 경로도 새 manifest에 맞춰
   명시적으로 준비해야 한다. 완료한 원본 캠페인 service를 재시작하지 않는다.

서버용 `prepare.py`는 기존 캠페인이 옆의 `2026-Wemeet-Barcode-Restoration` 디렉터리에 있는
배치를 사용한다. 경로를 자동 추정하는 범용 설치 프로그램은 아니다.
새 서버에서는 경로를 명시적으로 옮기고 새 실행 manifest를 만들어야 한다.
기존 4조건 학습 코드도 함께 보관했지만, 원래 학습을 재현할 때는 당시 W&B source archive와
프로토콜을 사용한다. 최신 디코더를 넣고 과거 실험과 같은 실행이라고 부르면 안 된다.

코드 검사는 다음 세 가지다. 실제 GPU 검사는 별도로 기록된다.

```bash
python -m ruff check .
python -m ruff format --check .
lint-imports --no-cache
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

## Notion·GitHub 공유

Notion은 `NOTION_ko.md`를 Markdown으로 가져오면 된다. CSV는 필요하면 별도 표로 가져온다.
W&B 링크는 각 표의 실제 run으로 연결한다.

GitHub branch는 `ai/mobilenet-followup-20261027`이다. 모델 가중치, 원본 데이터,
캐시, 인증 정보, W&B 로컬 디렉터리는 커밋하지 않는다. 대용량 원시 trace는 W&B artifact로 공유한다.
`main`에 직접 push하거나 실험 코드를 제품의 기본 판독 정책으로 바로 적용하지 않는다.

이는 합성 개발셋 평가다. 현장 데이터의 독립 test나 배포 승인을 대신하지 않는다.
