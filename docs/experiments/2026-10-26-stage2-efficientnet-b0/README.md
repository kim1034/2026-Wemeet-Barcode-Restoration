# Stage 2 EfficientNet-B0 비교 실험 가이드 — 2026-10-26

- **상태**: 실행 전
- **담당**: (실행 담당자 이름 기입)
- **관련 문서**: [ResNet18-C3 기준선](../2026-09-21-stage2-resnet18-c3/README.md) ·
  [백본 비교 설계](../2026-09-15-stage2-backbone/README.md) ·
  [0003 기하 추정 모델 선정](../../decisions/0003-기하-추정-모델-선정.md)

## 0. 이 실험의 규칙

**저장소 코드(`wemeet/`, `scripts/`, `tests/`, `pyproject.toml`)는 고치지 않는다.**
실험 코드는 전부 이 폴더(`docs/experiments/2026-10-26-stage2-efficientnet-b0/`) 안에 둔다.
EfficientNet이 ResNet보다 좋다는 결과가 나오면, 그때 별도 PR로 본 코드에 반영한다.

## 1. 무엇을 하는 실험인가

Stage 2 모델은 바코드 크롭을 보고 **TPS 제어점 48개의 좌표(숫자 96개)** 를 예측한다.
지금 기준 모델은 ResNet18-C3(ResNet18을 `layer3`까지만 쓴 것)이다.

이번 실험은 **ResNet18-C3와 모든 조건을 똑같이 두고 백본만 EfficientNet-B0로 바꿨을 때**
결과가 좋아지는지 확인한다.

| 항목 | ResNet18-C3 (기준) | EfficientNet-B0-C3 (이번) |
|---|---|---|
| 백본 | `resnet18` 의 `layer3` 까지 | `efficientnet_b0().features[:6]` |
| pretrained | `ResNet18_Weights.IMAGENET1K_V1` | `EfficientNet_B0_Weights.IMAGENET1K_V1` |
| 백본 출력 | `B×256×10×24` | `B×112×10×24` |
| adapter | `Conv2d(256→256, 1×1)` | `Conv2d(112→256, 1×1)` |
| 백본 파라미터 | 2,782,784 | 851,808 |

출력 크기와 파라미터 수는 torchvision 0.29.0에서 입력 `(1, 3, 160, 384)`로 확인한 값이다(2026-10-26).

> MobileNetV3-Small 결과(구제율 77.2%)는 학습 조건(lr, batch, epoch, beta)이 달라서
> **이번 결과와 비교하지 않는다.** 비교 대상은 ResNet18-C3 하나다.

## 2. 데이터는 어디에 있나

| 무엇 | 위치 |
|---|---|
| 합성 데이터 (학습 + 평가) | Hugging Face **`123metro/barcode-datasets`** (public), 폴더 `synthetic/v1/` |
| 반드시 쓸 revision | **`a833a3a5944148cb9815ccbd1f05aced50f6c464`** (ResNet 기준선과 같은 데이터) |
| 데이터 설명서 | 저장소의 [`data/hf/README.md`](../../../data/hf/README.md) |
| ResNet 기준 가중치 | Hugging Face `123metro/barcode-weights`, 파일 `geometry/resnet18-c3/model_state_dict.pt` |

내려받기 (저장소 루트에서, 약 240MB):

```bash
uv sync
uv run hf download 123metro/barcode-datasets --repo-type dataset \
  --revision a833a3a5944148cb9815ccbd1f05aced50f6c464 \
  --include "synthetic/v1/*" \
  --local-dir ../data/barcode-datasets-a833a3a59441
```

받으면 이런 구조가 된다.

```text
../data/barcode-datasets-a833a3a59441/synthetic/v1/
├── recipes/train.{L,M,H}.jsonl.gz   학습용 "레시피". 학습 중에 이미지를 그때그때 그린다
└── eval/{low,mid,high}/             평가용 이미지 각 500장 + 정답(npz) + manifest.jsonl
```

**주의**
- `eval/`은 **평가에만** 쓴다. 학습에 섞으면 결과가 무효다.
- 학습 이미지는 레시피에서 CPU로 그려 만들기 때문에 CPU를 많이 쓴다. GPU 서버(gpu06) 접근이 필요하면 팀장에게 요청한다.

## 3. 무엇을 고정하나 — 하나라도 바뀌면 비교가 무효다

ResNet 기준선([run.sh](../2026-09-21-stage2-resnet18-c3/run.sh))과 똑같이 맞춘다.

| 항목 | 값 |
|---|---|
| 데이터 | 위 revision의 `synthetic/v1` |
| 표본 수 | 버킷마다 `target` 5,000 / `hard` 3,500 / `first_ok` 1,500, validation 비율 0.1 |
| seed | 42 |
| 입력 | 흑백 → `160×384` 강제 리사이즈 → 3채널 복제 → ImageNet 정규화 |
| Head | adapter 뒤의 구조, 출력층, 항등 초기화(`identity + 0.5*tanh(raw)`) |
| 학습 | AdamW, lr `1e-4`, weight decay `1e-4`, Smooth L1 beta `0.02`, batch 8, 50 epoch |
| checkpoint | validation loss가 가장 낮은 epoch (`best.pt`) |
| 평가 | `eval/low·mid·high` 각 500장, 정답 문자열과 정확히 일치해야 성공 |

**규칙: 위 값을 바꿔야 할 것 같으면 (메모리 부족 등) 바꾸지 말고 먼저 물어본다.**

## 4. 실험 코드 만드는 법

저장소의 함수를 **import 해서 재사용**한다. 그러면 데이터 처리·학습 루프·평가 방식이 자동으로
ResNet과 같아진다. 새로 짜는 것은 모델 클래스 하나와 실행 스크립트뿐이다.

```text
docs/experiments/2026-10-26-stage2-efficientnet-b0/
├── README.md          이 문서 (결론·한계를 여기에)
├── RESULTS.md         측정표 전문
├── run.sh             처음부터 끝까지 재현하는 명령
└── code/
    ├── model.py       EfficientNetB0C3 클래스
    ├── train.py       학습
    ├── evaluate.py    평가
    └── latency.py     CPU 지연 측정
```

### `code/model.py`

[`wemeet/ai/geometry/model.py`](../../../wemeet/ai/geometry/model.py)의 `ResNet18C3`를
**복사해서** 백본·adapter 두 줄만 바꾼다. 나머지(head, output, `_initialize_identity`, `forward`)는
글자 하나 바꾸지 않는다.

```python
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
self.backbone = efficientnet_b0(weights=weights).features[:6]   # 바뀐 줄
self.adapter = nn.Conv2d(112, 256, kernel_size=1)               # 바뀐 줄
```

`destination_grid`, `N_X`, `N_Y`는 `wemeet.ai.geometry.model`에서 import한다.

### `code/train.py`

[`scripts/train_geometry.py`](../../../scripts/train_geometry.py)의 `main()`을 복사하고
**모델 만드는 줄만** 바꾼다. 나머지 함수는 import한다.

```python
from scripts.train_geometry import RecipeDataset, _run_epoch, read_records, set_seed
model = EfficientNetB0C3(pretrained=True).to(device)   # 원래: ResNet18C3(...)
```

W&B 설정도 아래 5장처럼 이름을 바꾼다.

### `code/evaluate.py`

[`scripts/evaluate_geometry.py`](../../../scripts/evaluate_geometry.py)의 `main()`을 복사한다.
`_model()`만 EfficientNet을 불러오도록 바꾸고 나머지는 import한다.

```python
from scripts.evaluate_geometry import _decode_counts, _load_rows, _predict
```

### `code/latency.py`

두 모델을 **같은 컴퓨터에서, 다른 프로그램을 끈 상태로** 잰다.

- `torch.set_num_threads(1)`, batch 1, 입력 `torch.zeros(1, 3, 160, 384)`
- warm-up 20회 후 200회 측정해서 p50 / p95 (ms)
- CPU 이름도 같이 적는다

### 실행 순서

1. **짧게 돌려 보기**: `--epochs 1`, 표본 `100 / 70 / 30`, 평가는 버킷당 20장. 끝까지 돌면 다음으로 넘어간다.
2. **본 학습**: 3장의 값 그대로 50 epoch.
3. **split 확인**: 출력된 `split.json`이 ResNet 실행의 것과 같은지 확인한다(seed가 같으면 같아야 한다).
4. **평가**: EfficientNet `best.pt`를 평가한다. **ResNet 가중치도 같은 평가 코드로 한 번 더 평가한다.**
   기존 결과(정답 1,067 / 오독 7)와 같으면 평가 경로가 같다는 확인이 된다.
5. **지연 측정**: 두 모델 모두 측정한다.

저장소 루트에서 `uv run python -m docs.experiments...`가 안 되면, 저장소 루트를 `PYTHONPATH`에 넣고 실행한다.

```bash
PYTHONPATH=. uv run python docs/experiments/2026-10-26-stage2-efficientnet-b0/code/train.py ...
```

## 5. 결과는 어디에 기록하나

| 무엇 | 어디에 | 비고 |
|---|---|---|
| 학습 곡선·설정 | **W&B** 프로젝트 `wemeet-barcode` (entity `kim-1034`) | run 이름 `restoration-efficientnet-b0-c3-v1`, 평가는 `...-eval-v1` |
| 체크포인트(`best.pt`)·평가 JSON | 로컬 `runs/restoration-efficientnet-b0-c3-v1/` | **git에 올리지 않는다** (`runs/`는 제외 대상) |
| 측정표 전문 | 이 폴더 `RESULTS.md` | 6장의 표 |
| 결론과 한계 | 이 문서 맨 아래 「결과」 | 7장 기준으로 한 문단 |
| 실험 목록 | [`docs/experiments/README.md`](../README.md) 표에 한 줄 추가 | |

### W&B에 반드시 기록한다

**모든 실행(학습·평가·지연 측정)은 W&B에 기록한다.** 팀 전체가 같은 대시보드에서
ResNet 기준선과 나란히 비교하기 때문이다. 로컬 로그나 캡처만 남기면 기록하지 않은 것으로 본다.
로그인 방법은 [`docs/external/wandb.md`](../../external/wandb.md)를 따른다(`uv run wandb login`).

| 실행 | W&B 설정 | 기록할 것 |
|---|---|---|
| 짧은 시험 실행 | `name="smoke-efficientnet-b0-c3"`, tags에 `smoke` | 끝까지 도는지만 확인. 결과 표에는 쓰지 않는다 |
| 학습 | `project="wemeet-barcode"`, `entity="kim-1034"`, `job_type="train"`, `name="restoration-efficientnet-b0-c3-v1"` | **config**: 3장의 모든 값 + `model="EfficientNet-B0-C3"` + 데이터 revision + params 수. **매 epoch**: `train/*`, `val/*` (원본 `_run_epoch`가 내는 값 그대로) |
| 평가 (EfficientNet) | `job_type="eval"`, `name="restoration-efficientnet-b0-c3-eval-v1"` | `eval/rescue_rate`, `eval/misread_rate`, `eval/decode_failed_rate`, 버킷별 같은 값, 평가 JSON 파일 |
| 평가 (ResNet 재평가) | `job_type="eval"`, `name="restoration-resnet18-c3-reeval-efb0"` | 위와 같음 |
| 지연 측정 | `job_type="latency"`, `name="latency-efb0-vs-resnet18"` | 두 모델의 p50 / p95, 스레드 수, CPU 이름 |

- **tags**: 모든 실행에 `stage2`, `backbone-compare`, 그리고 모델 이름(`efficientnet-b0-c3` 또는 `resnet18-c3`)을 붙인다. W&B에서 이 tag로 필터하면 비교 화면이 바로 나온다.
- **W&B가 안 되는 환경**이면 `mode="offline"`으로 돌리고, 끝난 뒤 `uv run wandb sync wandb/offline-run-*`로 올린다. 올릴 때까지 완료가 아니다.
- **가중치 파일(`best.pt`)은 W&B에 올리지 않는다.** 팀 규칙상 가중치는 채택된 뒤 Hugging Face에만 둔다. 원본 스크립트의 `wandb.Artifact(...)` 중 모델 artifact 부분은 지운다.

> ⚠️ **원본 스크립트의 이름을 그대로 쓰면 안 된다.** 원본은 run 이름·tags·artifact 이름이
> `resnet18-c3`로 고정돼 있고 `latest` 별칭을 붙인다. 바꾸지 않으면 **ResNet 기록을 덮어쓰거나
> 섞인다.** 위 표의 이름으로 바꾼다.

실행이 끝나면 W&B run 링크를 6장 표의 W&B 칸에 붙인다.

가중치를 Hugging Face(`123metro/barcode-weights`)에 올리는 것은 **채택이 결정된 뒤**에 한다.

작업은 브랜치 `ai/efficientnet-b0-experiment`에서 하고, 문서와 `code/`만 PR로 올린다.

## 6. 기록할 표

| 모델 | params | 정답 / 오독 / 실패 (1,500장) | 구제율 | 오독률 | low / mid / high 구제율 | val 좌표 오차 mean / P95 (px) | CPU p50 / p95 (1스레드, ms) | best epoch | W&B |
|---|---:|---|---:|---:|---|---|---|---:|---|
| ResNet18-C3 (재평가) | | | | | | | | 48 | |
| EfficientNet-B0-C3 | | | | | | | | | |

- 정답·오독·실패는 평가 JSON의 `ok` / `misread` / `decode_failed`이다.
- 좌표 오차는 W&B의 `val/` 값이다. 두 모델이 같은 코드로 계산했으므로 비교할 수 있다.
- params는 `sum(p.numel() for p in model.parameters())`로 센다(백본 + adapter + head 전체).

## 7. 결과를 읽는 법

- **seed 1개 실험이다.** 1,500장에서 구제율이 2~3%p 차이 나는 정도는 우연일 수 있다.
  그 정도면 "차이 없음 또는 판단 보류"로 쓴다.
- **오독이 늘었으면 구제율이 올라도 좋아졌다고 쓰지 않는다.** 틀린 번호가 나가는 것이 못 읽는 것보다 나쁘다.
- **버킷별로 본다.** 지금까지 오차는 `high` 버킷에 몰려 있다. 어느 버킷이 달라졌는지 적는다.
- 이 평가는 정답 캔버스 크기를 쓰는 **합성 데이터** 평가다. 실제 사진 성능이라고 쓰지 않는다.

**본 코드에 반영할지는 결과를 공유한 뒤 같이 정한다.** 구제율이 확실히 높고(대략 3%p 이상)
오독이 늘지 않았다면 seed를 추가해 다시 확인하고, 그다음 반영 PR을 만든다.

## 8. 막히면

| 증상 | 확인할 것 |
|---|---|
| `ModuleNotFoundError: scripts` / `wemeet` | 저장소 루트에서 `PYTHONPATH=.`을 붙여 실행했는지 |
| 데이터 경로 오류 | `--data-root`가 `.../synthetic/v1`을 가리키는지 |
| 학습 초반 loss가 NaN | lr·beta가 3장과 같은지 확인하고 공유한다. 임의로 바꾸지 않는다 |
| GPU 메모리 부족 | batch를 줄이지 말고 먼저 공유한다 |
| W&B 로그인이 안 됨 | `uv run wandb login`. 안 되면 offline으로 돌리고 나중에 `wandb sync`. 설명은 [`docs/external/wandb.md`](../../external/wandb.md) |
| HF 다운로드가 안 됨 | public이라 로그인 없이 받아져야 한다. 안 되면 [`docs/external/huggingface.md`](../../external/huggingface.md) |

## 9. 완료 체크리스트

- [ ] 데이터를 지정 revision으로 받음
- [ ] 짧은 시험 실행 통과
- [ ] `split.json`이 ResNet 실행과 같음
- [ ] ResNet 재평가 숫자가 1,067 / 7과 일치함 (다르면 이유를 적음)
- [ ] 학습·평가(두 모델)·지연 측정이 모두 W&B에 올라가 있음 (offline이었다면 sync까지)
- [ ] 6장 표를 `RESULTS.md`에 채우고 W&B 링크를 연결함
- [ ] 이 문서 「결과」에 7장 기준으로 결론을 씀
- [ ] `docs/experiments/README.md`에 한 줄 추가
- [ ] 저장소 코드(`wemeet/`, `scripts/`, `tests/`)를 고치지 않았음

## 결과

(실험 후 작성)
