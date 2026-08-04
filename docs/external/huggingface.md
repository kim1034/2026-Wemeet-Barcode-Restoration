# Hugging Face Hub 운영

데이터셋과 모델 가중치를 저장하는 곳이다.

## 왜 HF Hub인가

Git 저장소에는 **스크립트만** 올리고 실제 파일은 전부 HF Hub에 둔다.

- 무료이고 용량 제약이 사실상 없다
- **git 기반이라 데이터셋 버전 관리가 된다** — "어떤 데이터로 학습했는지"를 나중에 되짚을 수 있다
- private 저장소도 무료다 (개인정보가 담긴 실촬영 데이터용)
- 이미 `barcodes-google-ocr`를 HF에서 가져오므로 도구가 하나로 통일된다

### 왜 Git LFS나 DVC가 아닌가

| 방법 | 문제 |
|---|---|
| Git LFS | 무료 한도가 저장 1GB, 대역폭 월 1GB. 수만 장이면 8명이 clone하는 첫 주에 초과되고 그 뒤 유료 |
| DVC | 원격 스토리지를 따로 붙여야 하고, `dvc pull`/`dvc push`를 git 명령과 혼동하기 쉽다 |

---

## 1. 저장소 구성

조직 하나 아래에 저장소 3개를 만든다. 조직 이름은 `metro-wemeet`을 권장한다 (생성 시 변경 가능).

| 저장소 | 종류 | 공개 | 내용 |
|---|---|---|---|
| `metro-wemeet/barcode-datasets` | Dataset | **public** | 오픈소스 데이터 정리본, 합성 데이터 |
| `metro-wemeet/barcode-field` | Dataset | **private** | 실촬영 송장 (개인정보 포함) |
| `metro-wemeet/barcode-weights` | Model | public | 학습된 가중치 |

### 왜 실촬영 데이터를 분리하는가

실제 송장에는 수취인 이름·주소·연락처가 인쇄되어 있다. **public 저장소에 올리면 개인정보 유출이다.** 저장소 자체를 분리해두면 실수로 섞이지 않는다.

접근 권한은 필요한 인원으로 제한한다. 8명 전원에게 줄 필요는 없다 — 촬영 담당자와 최종 평가 담당자만으로 충분하다.

---

## 2. 준비 절차

### 팀장이 하는 일 (한 번)

1. huggingface.co 가입
2. 우상단 프로필 → **New Organization** → 이름 `metro-wemeet`, 무료 플랜
3. 조직 → Settings → Members → 팀원 8명 초대 (HF 계정명 필요)
4. 조직 → **New Dataset** 로 `barcode-datasets` 생성 (public)
5. 조직 → **New Dataset** 로 `barcode-field` 생성 (**private**)
6. 조직 → **New Model** 로 `barcode-weights` 생성 (public)

### 팀원이 하는 일 (각자 한 번)

1. huggingface.co 가입 후 계정명을 팀장에게 전달
2. 조직 초대 수락
3. Settings → **Access Tokens** → New token
   - 종류: **Write**
   - 이름: 본인 이름
4. 발급된 토큰으로 로그인 — **파일에 적는 게 아니다**

```bash
uv run hf auth login
```

`Enter your token (input will not be visible):` 가 나오면 붙여넣고 Enter.
**화면에 아무것도 안 보이는 것이 정상이다.** 그다음 `Add token as git credential?` 은
`n` 으로 답한다 — 우리는 HF에 git push를 하지 않고 `hf upload` 를 쓴다.

확인:

```bash
uv run hf auth whoami   # 계정명과 소속 조직이 나오면 성공
```

토큰은 `~/.cache/huggingface/token` 에 저장된다 (Windows는 `C:\Users\<이름>\.cache\huggingface\token`).
**이 파일을 직접 만들거나 편집하지 않는다.** CLI가 관리한다. 한 번 로그인하면
이후 `hf` 명령과 파이썬 코드가 자동으로 이 토큰을 쓴다.

**토큰은 절대 커밋하지 않는다.** 실수로 커밋하면 HF에서 즉시 폐기(revoke)하고 재발급한다.

### `.env` 에 넣으면 되지 않나

**지금은 안 된다.** 저장소에 `.env` 를 읽는 코드가 없다 (`python-dotenv` 를 쓰지 않는다).
`.env` 에 `HF_TOKEN` 을 적어도 `huggingface_hub` 은 못 본다 — 실제로 확인한 결과다.

```
.env 에 HF_TOKEN 을 넣은 상태  →  get_token() == None
환경변수로 export 한 경우      →  토큰이 보인다
```

`.env.example` 은 **"어떤 키가 필요한지 적어둔 목록"** 이다. 2단계에서
`wemeet/data/download.py` 가 생길 때 `python-dotenv` 를 붙일지 결정한다.
그때까지는 위의 `hf auth login` 이 유일하게 동작하는 방법이다.

### 명령 이름 주의 — `hf` 이지 `huggingface-cli` 가 아니다

`huggingface-cli` 는 폐기됐다. 실행하면 이렇게 나온다.

```
Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
```

인터넷 문서나 블로그에는 아직 `huggingface-cli` 로 적힌 것이 많다. **`hf` 로 바꿔 읽으면 된다.**

---

## 3. 저장소 내부 구조

저장소 안의 폴더 구조도 규약으로 정한다. 정하지 않으면 몇 주 뒤에 아무도 뭐가 어디 있는지 모른다.

### `barcode-datasets` (public)

```
barcode-datasets/
├── README.md                    데이터셋 카드 (아래 참고)
├── raw/                         오픈소스 원본 정리본
│   ├── kaggle-damaged/
│   ├── kaggle-detection/
│   └── hf-google-ocr/
├── synthetic/                   역방향 합성 데이터
│   ├── v1/
│   │   ├── train/               훼손 이미지 + 정답 이미지 쌍
│   │   ├── val/
│   │   └── manifest.csv         쌍 목록, 훼손 종류, 강도
│   └── v2/
└── labels/                      OBB 어노테이션
    └── detection-obb/
```

### `barcode-field` (private)

```
barcode-field/
├── README.md
├── dev/                         중간 점검·도메인 갭 측정·디버깅용 (40%)
│   ├── flat/                    ① 평탄 상태 촬영
│   ├── damaged/                 ② 훼손 상태 촬영
│   └── manifest.csv             ID, 정답 번호, 촬영 조건
└── eval/                        최종 평가 전용 (60%)
    ├── flat/                    11월 2주차에 딱 한 번 연다
    ├── damaged/
    └── manifest.csv
```

**`dev`와 `eval`을 송장 단위로 나눈다.** 같은 송장의 다른 각도 사진이 양쪽에 들어가면 사실상 유출이다. 촬영 시점에 송장을 두 무더기로 갈라놓고 시작한다.

`eval`은 11월 2주차 최종 평가 전까지 열지 않는다. 도메인 갭이 커서 실촬영 데이터를 학습에 투입해야 하는 경우에도 `dev`만 쓴다. 미리 보면 그 결과를 보고 튜닝하게 되고, 그 순간 기획서 4.2.1이 요구한 독립성이 깨진다.

`manifest.csv` 형식:

```csv
id,ground_truth,symbology,damage_type,damage_level,lighting,note
F0001,8801234567890,CODE128,curvature,high,indoor_led,비닐 2겹
F0002,8801234567891,CODE128,glare,mid,indoor_led,
```

**`ground_truth`가 이 파일의 존재 이유다.** 평탄 상태에서 먼저 판독해 채워 넣는다. 이 값 없이는 평가가 불가능하다.

### `barcode-weights`

```
barcode-weights/
├── README.md
├── detection/
│   ├── v1-traditional-cv/       학습 없는 베이스라인
│   └── v2-yolo-obb/
│       ├── best.pt
│       └── config.yaml          학습에 쓴 설정 (재현용)
└── restoration/
    └── v1-unet/
        ├── best.pt
        └── config.yaml
```

가중치와 함께 **학습 설정을 반드시 같이 올린다.** 설정 없는 가중치는 재현이 불가능해서 반년 뒤에 쓸 수 없다.

---

## 4. 버전 관리 규약

### 커밋 메시지

Git 저장소와 같은 형식을 쓴다.

```
[데이터] 합성 v1 학습셋 10,000쌍 추가
[AI] 탐지 v2 YOLO-OBB 가중치 추가 (mAP50 0.91)
```

### 태그

데이터셋 버전이 바뀌면 태그를 붙인다. 그러면 학습 코드에서 특정 버전을 고정할 수 있다.

```bash
# HF 웹 UI 또는 API로 태그 생성
# 예: synth-v1, synth-v2, field-eval-v1
```

학습할 때 어떤 태그를 썼는지 **W&B config에 기록한다.** 이게 없으면 나중에 모델 성능 차이가 데이터 때문인지 하이퍼파라미터 때문인지 알 수 없다.

```python
wandb.init(config={"dataset_revision": "synth-v1", ...})
```

### 폴더 이름에 버전을 넣는 이유

`synthetic/v1/`, `synthetic/v2/` 처럼 폴더를 나눈다. 덮어쓰지 않는다. 덮어쓰면 이전 실험 결과를 재현할 수 없다.

---

## 5. 업로드와 다운로드

### 다운로드 (팀원 전원)

```bash
uv run python -m wemeet.data.download
```

`wemeet/data/download.py`가 HF에서 받아 `downloads/` 아래에 넣는다. 팀원은 이 명령만 알면 된다.

### 업로드 (데이터파트 / AI파트)

```bash
# 데이터셋
uv run hf upload metro-wemeet/barcode-datasets \
    ./downloads/synthetic/v1 synthetic/v1 --repo-type=dataset

# 가중치
uv run hf upload metro-wemeet/barcode-weights \
    ./runs/detection-v2 detection/v2-yolo-obb --repo-type=model
```

### 수만 장을 올릴 때

개별 jpg 수만 개를 그대로 올리면 업로드와 다운로드가 매우 느려진다. **tar 샤드나 parquet으로 묶어서** 올린다.

```bash
# 예: 1,000장씩 tar로 묶기
tar -cf synthetic/v1/train-0000.tar train/000*.png
```

합성 데이터는 스크립트로 언제든 다시 만들 수 있으므로, 굳이 다 올리지 않고 **합성 스크립트와 난수 시드만 올리는 것도 방법이다.** 용량이 문제가 되면 이 방식을 검토한다.

---

## 6. 데이터셋 카드 (README.md)

각 저장소의 `README.md`에 아래를 적는다. 기획서 2.2의 산출물 "검증 데이터셋"에 해당하는 문서다.

```markdown
# 난인식 바코드 데이터셋

## 구성
| 분할 | 장수 | 출처 | 훼손 종류 |
|---|---|---|---|

## 라벨 형식
## 훼손 부여 방법 (합성 데이터)
## 정답(GT) 확보 방법
## 사용 시 주의
- eval/ 분할은 학습에 사용하지 않는다
## 라이선스
- 원본 데이터셋별 라이선스 명시
```

**원본 데이터셋 라이선스를 반드시 확인해 기재한다.** Kaggle과 HF 데이터셋은 각각 라이선스가 다르고, 재배포 조건이 있는 경우가 있다.

---

## 7. 체크리스트

**설정 (한 번)**
- [ ] 조직 `metro-wemeet` 생성
- [ ] 팀원 8명 초대 및 수락
- [ ] `barcode-datasets` (public) 생성
- [ ] `barcode-field` (**private**) 생성, 접근 권한 최소화
- [ ] `barcode-weights` 생성
- [ ] 각자 write 토큰 발급, `.env`에 저장
- [ ] `.env`가 `.gitignore`에 포함되어 있는지 확인

**운영 규약**
- [ ] 저장소 내부 폴더 구조 팀원 공유
- [ ] `manifest.csv` 형식 팀원 공유
- [ ] 데이터셋 버전이 바뀌면 태그를 붙이고 W&B config에 기록
- [ ] 가중치는 학습 설정과 항상 같이 올림
- [ ] `eval/` 분할은 학습에 절대 투입하지 않음
- [ ] 실촬영 원본은 public 저장소에 올리지 않음
