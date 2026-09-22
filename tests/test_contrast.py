"""Stage 1 검출 입력용 CLAHE 대비 향상 테스트.

가중치 없이 도는 순수 cv2 연산만 검증한다.
"""

import cv2
import numpy as np

from wemeet.ai.detection.contrast import enhance_for_detection


def test_enhance_for_detection_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    image = (rng.normal(loc=128, scale=5, size=(64, 96, 3))).clip(100, 140).astype(np.uint8)

    enhanced = enhance_for_detection(image)

    assert enhanced.shape == image.shape
    assert enhanced.dtype == image.dtype


def test_enhance_for_detection_increases_local_contrast():
    # 좁은 밝기 범위(100~140)로 만든 저대비 이미지는 CLAHE 후 표준편차가 커져야 한다.
    rng = np.random.default_rng(0)
    image = (rng.normal(loc=128, scale=5, size=(64, 96, 3))).clip(100, 140).astype(np.uint8)

    enhanced = enhance_for_detection(image)

    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    assert enhanced_gray.std() > original_gray.std()
