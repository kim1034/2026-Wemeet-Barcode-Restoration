# Stage 1 검출 CLAHE — 기본값 off · 2026-11-02

**결론: `detect()` 의 CLAHE 전처리를 기본으로 끈다.** 실사진에 가까운 benchmark-v2 에서는
탐지가 나아지지 않았고(1920 기준 99/100 → 99/100), 합성 val 에서는 오히려 떨어졌다
(mAP50 −0.06, recall −0.07~−0.09). 그런데 매 프레임 23.6 ms 를 쓴다.
`enhance.py` 와 `use_clahe=True` 는 남겨둔다 — CLAHE 를 건 데이터로 다시 학습한 모델이
나오면 그때 켜서 다시 잰다.

> 가중치는 배포본 `123metro/barcode-weights` 의 `detection/v2-yolo-obb/best.pt`
> (sha256 `223ef1b2…`) 하나로 쟀다. 측정 장비는 RTX 4050 Laptop (전원 연결).

## 배경

CLAHE(`wemeet/ai/detection/enhance.py`) 는 저대비 바코드를 검출기가 더 잘 보라고
2026-09-22 에 들어왔고 기본값이 `True` 였다. 그런데 이 모델은 **원본 이미지로
학습**했다 — CLAHE 는 추론 때만 걸린다.

같은 방향의 측정이 이미 한 번 있었다. [`2026-09-30-e2e-benchmark-v2`](../2026-09-30-e2e-benchmark-v2/README.md)
에서 입력 960 기준 CLAHE on 66 / off 67 로 효과가 없었고, "끌지는 GPU 시간과 같이
판단" 하기로 남겨뒀다. 이 기록은 그 판단에 필요한 두 가지 — 현재 입력 크기(1280·1920)
에서의 탐지와 CLAHE 자체 시간 — 를 잰다.

## 결과 1 — benchmark-v2 탐지 성공 (100장, 1920×1080)

`detect()` 를 그대로 부르고 `use_clahe` 만 바꿨다. 탐지 성공 = `detect()` 가 None 이 아님
(2026-10-02 imgsz 기록과 같은 기준).

| imgsz | CLAHE | high (32) | low (34) | mid (34) | 전체 |
|---|---|---|---|---|---|
| 1280 | off | 32 | 28 | 31 | 91/100 |
| 1280 | on | 32 | 29 | 31 | 92/100 |
| 1920 | off | 31 | 34 | 34 | **99/100** |
| 1920 | on | 31 | 34 | 34 | **99/100** |

현재 `DEFAULT_IMGSZ`(1920) 에서는 한 장도 다르지 않다. 1280 의 1장 차이는 우연 범위다.
로그: [`results/benchmark_v2.log`](results/benchmark_v2.log)

## 결과 2 — 합성 val mAP (2,000장, 1280×720)

`docs/experiments/2026-09-22-stage1-detection` 의 val 과 같은 2,000장이다. 같은 val 을
`enhance_for_detection` 으로 바꾼 사본을 만들어 둘 다 `model.val(single_cls=True)` 로 쟀다.
원본도 같은 jpg 품질(98)로 다시 저장해서 재압축 효과가 섞이지 않게 했다.

| imgsz | CLAHE | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| 1280 | off | 0.9716 | 0.9398 | 0.9328 | 0.9443 |
| 1280 | on | 0.9098 | 0.8648 | 0.9572 | 0.8550 |
| | **on − off** | **−0.0618** | **−0.0750** | +0.0243 | **−0.0893** |
| 1920 | off | 0.9548 | 0.8756 | 0.8957 | 0.8974 |
| 1920 | on | 0.8866 | 0.7888 | 0.9292 | 0.8275 |
| | **on − off** | **−0.0682** | **−0.0867** | +0.0335 | **−0.0699** |

P 는 오르고 R 이 크게 떨어진다 — 학습 때 못 본 대비 분포라 확신도가 전반적으로
내려가 conf 0.25 밑으로 빠지는 바코드가 생긴다고 본다 (train/test 도메인 차이).
로그: [`results/val_map.log`](results/val_map.log)

## 결과 3 — CLAHE 자체 시간

benchmark-v2 1920×1080 100장, 이미지마다 3회 중앙값. OpenCV 5.0.0 기본 스레드(22).

```
p50 = 23.6 ms    p95 = 28.7 ms
```

CPU 전처리라 GPU 가 아무리 빨라도 줄지 않는다. 같은 장비에서 검출 추론은 약 13~20 ms
(결과 2 의 ultralytics val, FP16) 라, CLAHE 가 추론 자체와 맞먹는다.

## 두 결과가 다른 이유

합성 val 은 떨어지고 benchmark-v2 는 그대로다. benchmark-v2 장면은 대비가 이미 충분하고
배경이 단순해서 CLAHE 를 걸어도 확신도가 conf 0.25 를 넘는 쪽에 남는 것으로 보인다.
어느 쪽이든 **CLAHE 로 좋아진 데이터셋은 없다** — 9/30 기록(66 vs 67)까지 세 번 모두 그렇다.

## 다른 파트에 미치는 영향

`detect()` 기본값이 바뀌므로 기본값으로 부르는 쪽은 모두 영향을 받는다. git 충돌은 없다.

- **Stage 2 [`2026-10-27-mobilenet-followup`](../2026-10-27-mobilenet-followup/README.md)**
  — "탐지 전 CLAHE 공통 적용" 조건으로 쟀다. 그 운영 후보 수치(예: 1280 · 81/0/19)는
  CLAHE on 기준이므로, 이후 비교할 때 탐지 조건이 다르다는 점을 감안한다.
- **SW `sw/decode-first-stream` · `sw/barcode-track-capture`** — `detect()` 를 기본값으로
  부르므로 머지되면 CLAHE 가 꺼진다. 예전 동작이 필요하면 `detect(image, use_clahe=True)`.

## 한계

- **전부 합성 데이터다.** CLAHE 가 원래 노린 실제 웹캠 저대비·저조도 장면은 테스트셋이
  없어 재지 못했다. 그런 장면에서 off 가 손해일 가능성은 남아 있다 — 실촬영 테스트셋이
  생기면 `use_clahe=True` 와 같이 다시 잰다.
- benchmark-v2 의 성공 기준은 박스가 1개 이상 나오는가이고 위치 정확도는 보지 않는다.
- 디코딩용 CLAHE(`wemeet/sw/decoding`, 원본으로 못 읽을 때 재시도)는 별개이고 이 기록과
  무관하다.

## 곁가지 관찰 — imgsz

합성 val(1280×720) 에서는 1920 이 1280 보다 낮다(mAP50 0.9548 vs 0.9716). 원본보다 크게
늘려 넣기 때문으로 보인다. benchmark-v2(1920×1080) 에서는 1920 이 1280 보다 높다
(99 vs 91). 입력 크기는 카메라 해상도와 함께 정해야 한다는 뜻이고, 이 PR 범위 밖이다.

## 재현

```bash
uv pip install ultralytics
hf download 123metro/barcode-weights detection/v2-yolo-obb/best.pt --local-dir downloads
hf download 123metro/barcode-datasets --repo-type dataset --include "benchmark/v2/*" --local-dir downloads/hf

python docs/experiments/2026-11-02-detection-clahe-off/code/clahe_benchmark.py \
    --data-root downloads/hf/benchmark/v2 --imgsz 1280 1920

# 합성 val 은 2026-09-22 기록의 make_barcode_dataset.py → shrink_dataset.py 로 만든 dataset_720
python docs/experiments/2026-11-02-detection-clahe-off/code/clahe_val_map.py \
    --val-images dataset_720/images/val \
    --weights downloads/detection/v2-yolo-obb/best.pt \
    --work-dir downloads/clahe-val --imgsz 1280 1920
```
