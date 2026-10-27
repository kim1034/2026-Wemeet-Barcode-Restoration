# 실행 중 확인한 사항

실험 입력과 코드의 기본 hash 목록은 실행 시작 전에 고정했다.
기존 학습 결과와 원시 평가, 완료 보고서는 그대로 보존했다.

## 다운로드

HF 데이터의 최초 다운로드에서 Xet 파일 생성 충돌(`File exists`)이 발생했다.
실패 로그를 보존하고 Xet을 끈 일반 다운로드로 다시 받았다.
scene revision을 고정한 상태에서 301개 파일, 고유한 장면 ID 100개를 확인했다.
데이터 revision이나 실험 모집단을 바꾸지 않았다.

## native 파이프라인 대조 설정 수정

`wemeet.sw.pipeline`은 `decode` 함수를 import할 때 참조를 보관한다.
평가 코드에서 `decoding.decode`의 CLAHE 설정을 바꾸는 것만으로는 이 별도 참조가 바뀌지 않았다.
따라서 legacy(off) 평가값을 native 파이프라인의 기본 on 설정과 대조하면서,
1920 조건의 native 일치 검사가 실패했다. 모델 오류나 데이터 손상은 아니었다.

별도 `native_bridge.py`에서 검증용 native 파이프라인에도 동일한 decoder 참조를 전달했다.
주 평가의 추론, 픽셀, 판독 정책, 데이터, 체크포인트는 변경하지 않았다.
off/on 참조와 원복을 검사하는 CPU 테스트 2개, GPU7의 실제 1920 장면 12개 대조가 통과했다.

수정 전 기본 코드와 seal은 그대로 유지했다. 완료한 3개 작업의 완료 파일을 다시 검사해 재사용했다.
실패한 1920의 부분 trace 3개와 로그도 별도로 남겼다. 동일한 단일 controller service가 후속 작업을 진행했다.
진행 중이던 validation worker가 끝난 후 controller를 교체하여 그 평가를 중단하거나 반복하지 않았다.

- [입력 및 원래 실행 코드 W&B](https://wandb.ai/kim-1034/wemeet-barcode/runs/cxmq8ezi)
- [설정 대조 수정 및 GPU 확인 W&B](https://wandb.ai/kim-1034/wemeet-barcode/runs/dhjq8dwv)
- 수정 seal: `3509326ea1ddcb9b84e151499eb12b2addec35b340c8156148aef5f0e2095039`

## 수치 읽기

`summary.json`의 진단 필드 `wrong_early_return`은 D0와 최종 결과가 모두 W인 샘플 수다.
`always_rectify`나 `agreement`에서 이것만으로 같은 오독을 그대로 채택했다고 해석하면 안 된다.
실제 문자열까지 같은 오독인지는 `failure_summary.csv`의 `same_wrong_as_d0`로 확인한다.
`wrong_after_rectification`은 오독이 관찰된 단계이며, 모델/TPS의 원인을 확정한 분류가 아니다.

성공 여부는 service의 exit code만으로 판단하지 않는다. 각 작업의 표본 수·원시 예측·hash,
W&B 업로드, 전체 집계와 최종 완료 파일을 확인한다.

## 최종 보고서 환경 보완

22개 평가·시간 작업이 모두 끝나고 장면 800건의 CPU 대조와 W&B 파일 검증을 통과한 뒤,
그래프 생성에서 `python-dateutil` 누락으로 보고서 후처리가 한 차례 실패했다.
최초 실패 로그와 당시 검증 파일 5개를 `delivery_dependency_fix_20261027/`에 복사해 hash로 보존했다.
별도 실험 환경에 `python-dateutil==2.9.0.post0`, `six==1.17.0`을 추가하고
Matplotlib 그림 생성 smoke 검사를 통과한 뒤, 같은 CPU closeout service를 이어서 실행했다.
추론 패키지와 측정 결과는 바꾸지 않았고 GPU 평가를 재실행하지 않았다.
보존된 실패 파일과 의존성 보완 기록도 최종 W&B 공유 artifact에 포함한다.
