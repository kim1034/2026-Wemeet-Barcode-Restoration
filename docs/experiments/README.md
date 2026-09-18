# 실험 기록

측정하고 버린 코드와 그 결과를 남기는 곳입니다. **승격된 코드는 `wemeet/` 에 있습니다.**

여기 있는 것은 "설계를 확정하기 전에 진짜 되는지 재본" 기록입니다.
숫자가 어떤 조건에서 나왔는지가 나중에 반드시 문제가 되기 때문에,
결론만 옮기지 않고 재현 가능한 형태로 함께 둡니다.

| 실험 | 무엇을 쟀나 |
|---|---|
| [2026-08-28-synthesis-spike](2026-08-28-synthesis-spike/README.md) | Stage 2 합성 데이터 — 왜곡 수식이 실제로 어떤 그림을 만드는가, 정답 제어점으로 되펴면 읽히는가. 설계 문서 §4·§8·§15 의 오류 3건과 목표 구간의 해상도 의존성 |
| [2026-09-09-control-points](2026-09-09-control-points/README.md) | 제어점 격자 개수 확정 — `6×3` → **`16×3`**. 표본 정의를 `n_x` 에서 떼어내 순환을 끊고, 좌표 오차 σ 를 축으로 넣어 상한 곡선을 처음 쟀다. 최적 격자가 σ 의 함수임을 보이고 최대 후회로 확정. `n_y` 도 σ 에 대해 뒤집힌다 (`16×7` 은 σ=0 에서 1위, σ=4 에서 14.3%). 확정값은 3단계 20ms 예산을 17% 넘는다 |
| [2026-09-15-stage2-backbone](2026-09-15-stage2-backbone/README.md) | Stage 2 v1 기준 백본 `ResNet18-C3`와 `MobileNetV3-Small`·`EfficientNet-B0` 3-way 비교 설계. 같은 spatial Geometry Head·`16×3` 제어점·TPS 경로에서 구제율, pixel error, latency, model cost를 비교한다 |

## 규칙

- 폴더 이름은 `YYYY-MM-DD-주제`
- `README.md` 에 **결론과 한계**를, `RESULTS.md` 에 **측정표 전문**을
- `run.sh` 로 재현되게. 리포 환경(`pyproject.toml`)은 건드리지 않는다
- 결론이 설계 문서나 `decisions/` 를 뒤집으면 **그 문서에 정정을 추가하고 여기로 링크**한다.
  원문은 지우지 않는다 (`decisions/README.md` 의 원칙)
- 스파이크 코드는 `ruff` 검사에서 제외된다 (`pyproject.toml` 참고)
