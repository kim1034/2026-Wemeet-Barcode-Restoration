# Weights & Biases 운영

학습 실험을 기록하는 곳이다.

## 왜 필요한가

기획서 4.6.3은 "모델 버전별 성능 이력을 관리"하라고 하는데, 4.4의 도구 목록에는 Git/GitHub만 있었다. 이 문서가 그 빈칸을 채운다.

이 프로젝트는 **모델을 정해놓고 시작하는 게 아니라 후보를 비교해서 고르는** 구조다 (`docs/decisions/0002`, `0003`). 그러려면 학습을 여러 번 돌리고 결과를 나란히 놓고 봐야 한다. 손으로 적으면 8명이 각자 다른 곳에 적는다.

- 학생 무료
- 학습 코드에 3줄 추가하면 정확도·손실 그래프가 자동으로 쌓인다
- 8명이 같은 대시보드를 본다
- 어떤 설정으로 돌렸는지가 함께 기록되어 재현이 가능하다

## HF Hub와 역할을 나눈다

둘 다 파일을 저장할 수 있어서 헷갈리기 쉽다. **역할을 명확히 나눈다.**

| | 저장하는 것 |
|---|---|
| **W&B** | 지표, 그래프, 하이퍼파라미터, 학습 곡선 |
| **HF Hub** | 데이터셋 파일, 모델 가중치 파일 |

W&B의 Artifacts 기능으로 가중치도 올릴 수 있지만 쓰지 않는다. 같은 파일이 두 곳에 있으면 어느 게 최신인지 모르게 된다. **가중치는 HF에만 둔다.**

W&B 기록에는 HF 가중치 경로를 남겨서 둘을 연결한다.

```python
wandb.init(config={"weights_uri": "metro-wemeet/barcode-weights@detection/v2-yolo-obb"})
```

---

## 1. 준비 절차

### 팀장이 하는 일 (한 번)

1. wandb.ai 가입
2. **학생 계정 신청** — 학교 이메일로 인증하면 팀 기능이 무료로 열린다
3. 팀(Team) 생성 — 이름 `metro-wemeet` 권장
4. 팀 → Members → 팀원 초대 (최소한 AI파트 전원)

### 학습을 돌리는 사람이 하는 일 (각자 한 번)

1. wandb.ai 가입, 팀 초대 수락
2. Settings → **API keys** 에서 키 복사
3. 로컬에 저장

```bash
echo 'WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxx' >> .env
```

또는 한 번 로그인해두면 이후 자동으로 쓰인다.

```bash
uv run wandb login
```

**API 키는 절대 커밋하지 않는다.** `.env`는 `.gitignore`에 포함되어 있다.

---

## 2. 프로젝트와 실행 이름 규약

### 프로젝트

프로젝트 하나만 쓴다: **`wemeet-barcode`**

Stage별로 프로젝트를 나누지 않는다. 나누면 탐지와 복원의 추론 시간을 합쳐서 500ms 예산을 확인할 때 두 곳을 오가야 한다.

### 실행(run) 이름

```
<stage>-<model>-v<n>
```

예시:

```
detection-traditional-v1
detection-yolo-obb-v1
detection-yolo-obb-v2
restoration-unet-v1
restoration-unet-v2
restoration-pix2pix-v1
```

이름만 보고 무슨 실험인지 알 수 있어야 한다. `test`, `실험1`, `final_final` 같은 이름은 쓰지 않는다.

### 태그

| 태그 | 의미 |
|---|---|
| `baseline` | 비교 기준선 (디코더 단독, 학습 없는 전통 영상처리 등) |
| `candidate` | 모델 선정 후보로 정식 비교에 들어가는 실행 |
| `final` | 최종 채택된 실행 |
| `failed` | 실패했지만 기록으로 남기는 실행 |

`docs/decisions/0002`, `0003`의 비교 표는 `candidate` 태그가 붙은 실행들로 채운다.

---

## 3. 반드시 기록할 것

### config — 무엇으로 돌렸는가

```python
import wandb

wandb.init(
    project="wemeet-barcode",
    entity="metro-wemeet",
    name="detection-yolo-obb-v2",
    tags=["candidate"],
    config={
        # 데이터 (가장 중요)
        "dataset_repo": "metro-wemeet/barcode-datasets",
        "dataset_revision": "synth-v1",     # HF 태그
        "train_size": 10000,
        "val_size": 2500,

        # 모델
        "model": "yolov8n-obb",
        "input_size": 640,
        "pretrained": True,

        # 학습
        "epochs": 100,
        "batch_size": 16,
        "lr": 0.001,
        "optimizer": "AdamW",
        # 복원 모델일 때만 쓴다 (탐지 모델에는 해당 없음)
        # "loss_weights": {"l1": 1.0, "edge": 0.5, "ssim": 0.3},

        # 환경
        "gpu": "RTX 4090",
        "seed": 42,
    },
)
```

**`dataset_revision`이 가장 중요하다.** 이게 없으면 두 실험의 성능 차이가 데이터 때문인지 하이퍼파라미터 때문인지 구분할 수 없다. 데이터가 계속 늘어나는 프로젝트라 특히 그렇다.

**`seed`도 반드시 남긴다.** 같은 설정으로 다시 돌렸는데 결과가 다르면 원인을 찾을 수 없다.

### 지표 — 결과가 어땠는가

```python
# 학습 중 매 epoch
wandb.log({"epoch": e, "train_loss": tl, "val_loss": vl, "val_ssim": s})

# 학습 종료 후 최종 지표
wandb.summary.update({
    # 정확도
    "mAP50": 0.912,
    "mAP50_95": 0.624,
    "angle_error_deg": 2.4,

    # 속도 — 500ms 예산 확인용
    "inference_ms_p50": 38.4,
    "inference_ms_p95": 51.2,

    # 최종 목표와 연결되는 지표
    "decode_success_rate": 0.87,
    "false_decode_rate": 0.0,

    # 연결 정보
    "weights_uri": "metro-wemeet/barcode-weights@detection/v2-yolo-obb",
})
```

**추론 시간을 반드시 기록한다.** 기획서 4.6.1에는 정확도 지표만 있었다. 정확해도 500ms 예산을 넘으면 탈락이므로, 정확도와 같은 자리에서 봐야 한다.

**`false_decode_rate`(오독률)도 기록한다.** 틀린 번호를 자신 있게 내놓는 것이 판독 실패보다 나쁘다. 물류에서는 잘못된 화물 정보가 흘러가기 때문이다.

### 이미지 — 눈으로 확인

복원 모델은 숫자만으로 판단하기 어렵다. 원본·복원·정답을 나란히 올린다.

```python
wandb.log({
    "samples": [
        wandb.Image(damaged, caption="훼손 원본"),
        wandb.Image(restored, caption="복원 결과"),
        wandb.Image(ground_truth, caption="정답"),
    ]
})
```

SSIM이 높은데 디코더가 못 읽는 경우가 생긴다. 그때 눈으로 봐야 원인이 보인다.

---

## 4. 팀 규약

**실패한 실험을 지우지 않는다.** `failed` 태그를 붙여 남긴다. 지우면 다른 팀원이 같은 실수를 반복한다. "학습률 0.1은 발산한다"는 것도 기록할 가치가 있는 정보다.

**한 번에 한 가지만 바꾼다.** 학습률과 배치 크기를 동시에 바꾸면 어느 쪽이 효과를 냈는지 알 수 없다.

**결정 문서와 연결한다.** `docs/decisions/0002`, `0003`의 비교 표에 W&B 실행 이름을 적어둔다. 그러면 11월 보고서를 쓸 때 근거를 바로 찾을 수 있다.

```markdown
| 후보 | mAP50 | 추론 ms | 라이선스 | W&B run |
|---|---|---|---|---|
| 전통 영상처리 | 0.72 | 12 | 제약 없음 | detection-traditional-v1 |
| YOLO-OBB | 0.91 | 38 | AGPL-3.0 | detection-yolo-obb-v2 |
```

**주간 회의 전에 대시보드를 본다.** 각자 말로 보고하는 대신 화면을 같이 본다. 기획서 4.5의 마일스톤별 멘토 코드 리뷰에서도 이 화면을 그대로 쓸 수 있다.

---

## 5. 체크리스트

**설정 (한 번)**
- [ ] 팀장 wandb.ai 가입 및 **학생 계정 인증**
- [ ] 팀 `metro-wemeet` 생성
- [ ] AI파트 전원 초대 및 수락
- [ ] 프로젝트 `wemeet-barcode` 생성
- [ ] 각자 API 키 발급, `.env`에 저장
- [ ] `.env`가 `.gitignore`에 포함되어 있는지 확인

**매 학습마다**
- [ ] 실행 이름이 `<stage>-<model>-v<n>` 형식인가
- [ ] 태그를 붙였는가 (`baseline` / `candidate` / `final` / `failed`)
- [ ] config에 `dataset_revision`과 `seed`가 있는가
- [ ] 추론 시간(p50/p95)을 기록했는가
- [ ] 오독률을 기록했는가
- [ ] 가중치를 HF에 올리고 `weights_uri`를 남겼는가

**모델 선정 시**
- [ ] `candidate` 실행들이 같은 `dataset_revision`으로 돌았는가
- [ ] 비교 표에 정확도·추론시간·라이선스가 모두 있는가
- [ ] `docs/decisions/`에 W&B 실행 이름을 적었는가
