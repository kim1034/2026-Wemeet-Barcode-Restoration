"""탐지 결과를 크롭으로 바꾸는 기하 계산.

모델 가중치와 무관한 순수 함수만 둔다. 학습 결과가 없어도 테스트할 수 있어야
회전 부호 같은 실수를 일찍 잡는다 — 부호를 틀리면 크롭이 반대로 기울고,
2단계 기하 추정이 통째로 망가진다.

규격 두 가지 (막대 영역 기준, 흰 종이 여백이 아니다):
    50x30 라벨 → 막대 영역 가로:세로 = 5:3
    60x40 라벨 → 막대 영역 가로:세로 = 6:4 (= 3:2)

**규격 판정을 모델의 클래스 출력으로 하지 않는다.** 두 규격은 막대 무늬가
같고 비율만 약 10% 다르다. 2클래스로 학습했을 때 Precision 0.845 / Recall
0.867 이었고, 단일 클래스로 바꾸고 박스 모양으로 판정하니 0.934 / 0.937 이
되었다 (docs/experiments/2026-09-22-stage1-detection).
"""

import cv2
import numpy as np

RATIO_5X3 = 5 / 3
RATIO_6X4 = 3 / 2
#: 두 규격의 중간값. 이보다 길쭉하면 5:3, 아니면 6:4 로 본다.
RATIO_THRESHOLD = (RATIO_5X3 + RATIO_6X4) / 2
#: 크롭할 때 각 변 바깥으로 붙이는 여백 비율. 가로·세로 모두 1.2 배가 되어 비율은 유지된다.
CROP_PAD = 0.10


def upright(cx: float, cy: float, w: float, h: float, angle_rad: float) -> tuple[float, ...]:
    """긴 변이 가로가 되도록 (w, h, angle) 을 정규화한다.

    모델은 같은 박스를 90도 돌려서 낼 수 있다. 그대로 두면 5:3 이 3:5 로
    읽혀서 규격 판정이 뒤집힌다.
    """
    if w < h:
        w, h = h, w
        angle_rad += np.pi / 2
    return cx, cy, w, h, angle_rad


def decide_ratio(w: float, h: float) -> tuple[float, float]:
    """박스 크기 → (판정한 규격 비율, 실제 비율).

    실제 비율도 함께 돌려주는 것은 의도한 것이다. 기준선 근처의 애매한 판정을
    호출부가 따로 다룰 수 있어야 한다 (RESULTS.md 「애매 구간」).
    """
    measured = w / max(h, 1e-6)
    return (RATIO_5X3 if measured >= RATIO_THRESHOLD else RATIO_6X4), measured


def fit_ratio(w: float, h: float, ratio: float) -> tuple[float, float]:
    """짧은 쪽만 늘려 비율을 맞춘다.

    **늘리기만 하고 줄이지 않는다.** 줄이면 막대가 잘려 디코딩이 불가능해지지만,
    늘리면 여백이 조금 더 붙을 뿐이다. 규격 판정이 틀려도 바코드 자체는
    온전히 남는다.
    """
    return (w, w / ratio) if w / max(h, 1e-6) >= ratio else (h * ratio, h)


def crop_upright(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_rad: float,
    pad: float = CROP_PAD,
) -> tuple[np.ndarray, float]:
    """박스를 수평으로 세워 여백과 함께 잘라낸다.

    반환: (크롭 BGR, 회전 보정한 각도 [도, 반시계])

    cv2.getRotationMatrix2D 는 양수 각도를 반시계로 해석한다(원점 좌상단).
    이미지 좌표계는 y 가 아래로 증가하므로, 화면에서 시계 방향으로
    angle_rad 만큼 기운 박스는 같은 크기의 양수 각도로 되돌려 세운다.
    그래서 여기 넘기는 각도가 곧 schemas.DetectedBarcode.angle_deg_ccw 다.
    """
    angle_deg_ccw = float(np.degrees(angle_rad))
    out_w = max(2, int(round(w * (1 + 2 * pad))))
    # 세로를 가로에서 역산한다. 각각 반올림하면 비율에 오차가 섞인다.
    out_h = max(2, int(round(out_w * h / max(w, 1e-6))))

    m = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg_ccw, 1.0)
    m[:, 2] += (out_w / 2 - cx, out_h / 2 - cy)
    crop = cv2.warpAffine(
        image_bgr,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, angle_deg_ccw
