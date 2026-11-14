> 이 문서는 2026-10-26 당시의 준비·실행 기록을 보존한 것입니다. 최종 완료 결과는 [REPORT_ko.md](REPORT_ko.md), 후속 실험까지 합친 결과는 [통합 보고서](../2026-10-27-mobilenet-followup/REPORT_ko.md)를 참조하세요. 아래의 준비 중·실행 전 상태는 현재 상태가 아닙니다.

# GPU 4–7 실행

2026-10-26 사용자 지시: “4번부터 gpu 4-7번 이렇게 4개 써서 진행해.”
이 지시로 GPU 프로파일링, 12개 학습, 선택 체크포인트 평가와 비교 보고서를 진행한다.
이전 CPU 준비 단계의 정지 범위는 여기까지 확장됐다.

| 물리 GPU | 모델 | 조건 | 순서 |
|---|---|---|---|
| 4 | ResNet18-C3 | R: lr 1e-4, batch 8, 50 epoch, beta 0.02 | seed 42 → 43 → 44 |
| 5 | ResNet18-C3 | M: lr 3e-4, batch 64, 100 epoch, beta 1/154 | seed 42 → 43 → 44 |
| 6 | MobileNetV3-Small | R | seed 42 → 43 → 44 |
| 7 | MobileNetV3-Small | M | seed 42 → 43 → 44 |

준비된 `prepared/protocol.json`과 12개 설정은 그대로 사용한다. 모델·학습·평가
소스도 준비 때의 hash를 유지한다. 실행 관리 코드는 `scripts/controlled_campaign/`에
분리했으며 실행 manifest로 고정한다. 각 프로세스에는 배정 GPU UUID 한 개만 노출한다.

## 진행

1. 학습 이미지 27,024개의 기존 렌더링 캐시를 CPU 8 worker로 완성했다(약 193초).
2. 네 조건에서 실제 GPU 학습 배치와 검증 배치를 측정했다. 실제 PID/UUID 관찰,
   유한 loss/gradient, CPU/GPU feature·출력 비교 및 고정 D0 기반 판독 검사가 통과했다.
   프로파일의 임시 가중치는 저장하지 않으며 본 학습은 원래 초기화에서 시작한다.
3. 네 GPU가 각자 세 seed를 순서대로 학습하고, 각 선택 checkpoint를 benchmark
   1,500장과 validation 2,976장으로 평가한다. 결과는 각각 W&B에 기록한다.
4. 모든 학습이 끝나면 seed 42의 선택 checkpoint 네 개를 순차적으로 시간 측정한다.
   50회 warm-up, benchmark 전체 3회 반복이며 Stage 1 시간은 제외한다.
5. 같은 preset·seed·dataset의 결과를 sample ID로 맞춰 paired 비교한다.
   3개 seed의 평균·표준편차, 원시 결과 CSV, 한국어 보고서와 W&B Artifact를 생성한다.

실행 서비스는 `wemeet-controlled-v2-gpu4567-20261026.service` 하나이며 터미널과
독립적으로 동작한다. `writer.lock`으로 중복 실행을 차단한다. 기존 출력이 발견되거나
예상치 못한 오류가 생긴 조건은 보존 후 보류하며, 다른 GPU의 조건은 계속 진행한다.
실패한 작업을 처음부터 자동 재실행하지 않는다. 서비스 재시작 전 실제 진행 상태와
남은 프로세스를 점검해야 한다. 완료 여부는 서비스 활성 상태가 아니라 실제 출력으로 판정한다.

## 초기 실측

| GPU | 최대 할당 텐서 메모리(GiB) | 판독 평가 제외 epoch 추정(초) | 학습 1회 추정(시간) |
|---|---:|---:|---:|
| 4 | 0.260 | 108.3 | 2.15 |
| 5 | 2.340 | 71.2 | 3.31 |
| 6 | 0.186 | 113.7 | 2.34 |
| 7 | 1.098 | 71.3 | 3.32 |

메모리는 PyTorch 최대 할당 텐서 기준이며 드라이버·예약 메모리 전체와 다르다.
출력의 CPU/GPU 최대 절대 차이는 네 조건 모두 5.96e-8이었다(실제 이미지 6장).
PyTorch의 adaptive pooling CUDA 역전파 결정성 경고는 원래 준비 조건대로 기록했다.
GPU 실행의 비트 단위 재현성을 주장하지 않는다.

초기 예상은 첫 네 학습 약 3–4시간, 12개 학습과 평가·시간 측정·집계 전체 약
10–14시간이다. 캐시가 준비된 짧은 배치 측정에서 외삽했으며, 학습된 예측에 따라
판독 시간도 달라진다. 실제 epoch 시간과 5 epoch마다의 판독 평가 시간으로 보정한다.

본 학습은 2026-10-26 09:43 UTC(18:43 KST)에 시작했다. 네 조건 모두 첫 epoch의
train 27,024 / validation 2,976 샘플 처리와 정상 checkpoint 저장을 확인했다.
실제 첫 epoch는 GPU 4/5/6/7 각각 109.5/105.6/118.0/102.7초였다.
이 실측을 반영한 예상은 **첫 네 학습 약 4–5시간, 전체 작업 약 13–16시간**이다.
첫 epoch 기준 추정이며 실제 5 epoch 판독 평가를 거치면 다시 보정할 수 있다.
근거는 실행 루트의 `actual_launch_verified.json`, `runtime_eta.json`에 기록한다.

## 확인 위치

저장 루트: `runs/stage2-controlled-v2/campaign-gpu4567-20261026/`

- `progress.json`: GPU별 PID, W&B 링크, 마지막 완료 epoch, 완료 개수와 오류.
- `events.jsonl`: 시작·종료와 5초 간격 실제 GPU 관찰.
- `profiles/`: GPU 사전 검증; 각 결과의 `run_url`은 W&B 기록.
- `evaluations/`: 12개 선택 checkpoint의 24개 판독 평가.
- `latency/`: seed 42 선택 checkpoint의 순차 시간 측정.
- `report/REPORT_ko.md`, `report/results.csv`, `report/summary.json`: 최종 산출물.
- `report/completion.json`: 전체 집계 완료 증거와 최종 W&B 링크. 생성 전에는 미완료다.

학습 checkpoint는 기존 설정대로 `runs/stage2-controlled-v2/training/`에 저장한다.
W&B project는 `kim-1034/wemeet-barcode`, group은 `stage2-backbone-controlled-v2`다.
MobileNet은 공통 head를 적용한 새 구현이다. 기존 benchmark는 합성 개발 데이터이며
새 독립 test가 아니다. 같은 preset 안에서 두 백본을 비교한다.

## 19:43 KST 저장 오류 수정 후 재개

첫 네 작업은 epoch5 전체 판독 평가 결과를 JSON으로 저장하면서 `numpy.bool_`
직렬화 오류로 중단됐다. 네 epoch4 체크포인트는 정상 저장돼 있었으며, 원래
서비스는 18:57 KST에 종료됐다. GPU 프로파일은 판독 결과의 파일 저장까지
검사하지 않아 이 오류를 사전에 잡지 못했다.

`scripts/controlled_resume_20261026/`에서 참·거짓 값의 JSON 표현만 보정했다.
원래 소스·프로토콜·조건은 유지하며 추가 실행 코드와 해시를 별도로 기록한다.
원래 캠페인의 진행 기록과 로그는 그대로 두고, 네 체크포인트와 메트릭·W&B 로컬
기록 등 119개 파일을 별도 보존했다. CPU 회귀 검사 270개가 통과했고 10개는
기존 fixture 부재로 건너뛰었다. 실제 네 체크포인트 각각 12개 샘플의 평가·저장·
재읽기 결과가 동일함을 확인했다.

19:43 KST에 같은 단일 서비스를 재개했다. 네 작업 모두 저장된 epoch4의 가중치,
옵티마이저 및 난수 상태를 복원하고 동일 W&B run ID로 이어간다. 실패한 epoch5의
빈 파일을 유지하고 새 평가는 `epoch-0005-attempt-02`에 저장한다. 새 원시 결과로
전체 2,976개 validation과 선정 기준을 검증한 뒤 이후 학습을 계속한다.

현재 실행 루트는 `runs/stage2-controlled-v2/campaign-gpu4567-20261026-resume-json/`이다.
상위 `CURRENT_STATE.json`이 이 경로를 가리킨다. 원래 실행 루트는 중단 당시
상태로 보존돼 있으므로 현재 진행률은 새 `progress.json`으로 확인한다.
`revision.json`, `preserved/`, `actual_gpu_resume_verified.json`에 수정·보존·실제
GPU 재개 증거가 있다. `first_full_validation_verified.json`은 네 작업의 epoch5
전체 결과 검증이 실제 완료된 경우에만 생성된다. 이후 평가·집계도 새 실행 루트에
저장하며, 모든 W&B 실행과 결과 artifact에 수정 내역을 함께 기록한다.

19:49:57 KST에 네 작업 모두 epoch5 체크포인트와 validation 2,976개 전체 결과
검증을 통과했다. 원시 결과에서 재계산한 요약, 고정 sample ID·GT·D0, 유한 가중치,
기존 epoch1–4 메트릭의 보존을 확인했다. 각 epoch5는 전체 판독 평가를 포함해
330.1–344.7초 걸렸다. 19:50 기준 남은 전체 작업의 점 추정은 약 13.2시간이며,
범위는 13–16시간(10/27 약 09–12시 KST)이다. 이후 판독 비용과 서버 부하에 따라 달라진다.

## 전체 완료까지 자동 복구·감시

사용자의 후속 지시 “전체 평가와 집계 완료 될 때까지 끝까지 진행되게 해. 중간에 끊기지 않게!”에
따라 `scripts/controlled_durable_20261026/`을 추가했다. 기존 모델·학습·평가 조건과
고정된 두 실행 계층은 유지한다. 현재 정상 학습의 PID와 시작 시각을 보존한 상태로
같은 서비스에 `Restart=on-failure`, 60초 재시작과 새 재개 명령을 등록했다.

추가 CPU 감시 서비스 `wemeet-controlled-v2-supervision-20261026.service`는 20초마다
실제 진행을 확인한다. 기존 실행기의 작업 보류나 서비스 중단 시 마지막 정상 checkpoint와
완료된 단계에서 이어가며, 중단된 미완료 결과는 별도 보존한다. 완료된 학습·평가는 재사용한다.
일시적 W&B 연결·업로드 오류도 재시도한다. 재시도 간격은 60초부터 최대 10분이다.

최종 12개 학습, 24개 평가, 4개 시간 측정, 12개 비교와 3개 seed 집계를 원시 결과에서
검증하고 W&B 최종 보고·독립 검증까지 업로드한 뒤 종료한다. 실제 완료 증거는 현재
실행 루트의 `durable/completion.json`이며 감시 상태는 `durable/status.json`이다.
완료 검증 후 두 서비스를 자동 시작 대상에서 해제한다.

전체 검사 304개 통과, 기존 fixture 관련 10개 건너뜀. 별도 CPU systemd 장애 시험은
exit17 후 자동 재시작 1회, 저장 파일 보존, 최종 exit0을 확인했다. 실제 학습에는 장애를
주입하지 않았다. 코드·승인·검사 근거는 `durable/manifest.json`, `durable/source.zip`,
`durable/deployment.json`과 `durable/README_ko.md`에 있다.
