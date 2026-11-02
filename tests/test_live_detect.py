"""실시간 뷰어(`scripts/live_detect.py`)의 그리기·라벨 함수 테스트. 카메라·모델 없이 돈다."""

from __future__ import annotations

import numpy as np

from scripts.live_detect import draw_hud, draw_obb, obb_label_line, parse_source
from wemeet.ai.detection import ObbDetection
from wemeet.ai.detection.geometry import RATIO_5X3, RATIO_6X4, obb_corners


def _found(cx=100.0, cy=60.0, w=50.0, h=30.0, angle=0.0, ratio=RATIO_5X3) -> ObbDetection:
    return ObbDetection(obb_corners(cx, cy, w, h, angle), cx, cy, w, h, angle, ratio, 0.87)


def test_라벨은_클래스와_정규화된_꼭짓점_8개다():
    parts = obb_label_line(_found(), width=200, height=120).split()
    assert parts[0] == "0"
    values = np.array(parts[1:], dtype=float).reshape(4, 2)
    assert np.allclose(
        sorted(map(tuple, values * [200, 120])), [(75, 45), (75, 75), (125, 45), (125, 75)]
    )


def test_6대4_는_클래스_1이다():
    assert obb_label_line(_found(ratio=RATIO_6X4), 200, 120).startswith("1 ")


def test_화면_밖으로_나간_꼭짓점은_0에서_1로_자른다():
    values = np.array(obb_label_line(_found(cx=5, cy=5), 200, 120).split()[1:], dtype=float)
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_박스를_그리면_박스_위치의_픽셀이_바뀐다():
    image = np.zeros((120, 200, 3), np.uint8)
    draw_obb(image, _found())
    assert image[45, 100].any()  # 위쪽 변 (y = 60 - 15)
    assert not image[60, 100].any()  # 박스 안쪽은 그대로


def test_상태줄을_그려도_크기와_형식이_유지된다():
    image = np.zeros((720, 1280, 3), np.uint8)
    out = draw_hud(image, ["cam 30.0 fps", "detect 8.3 ms"])
    assert out.shape == (720, 1280, 3) and out.dtype == np.uint8
    assert out[:80, :400].any()


def test_소스는_숫자면_카메라_번호_아니면_주소다():
    assert parse_source("0") == 0
    assert parse_source("rtsp://10.0.0.2/s1") == "rtsp://10.0.0.2/s1"
