import numpy as np
from wemeet.data.render import render_clean


def _narrowest_black_run(img: np.ndarray) -> int:
    """중간 행에서 검은 픽셀이 연속하는 가장 짧은 구간의 길이."""
    row = img[img.shape[0] // 2] < 128
    runs, current = [], 0
    for pixel in row:
        if pixel:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return min(runs)


def test_render_clean_shape_and_dtype():
    img = render_clean("WEMEET0001", module_px=4.0, height_px=220)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape[0] == 220


def test_module_width_matches_request():
    img = render_clean("WEMEET0001", module_px=4.0)
    assert abs(_narrowest_black_run(img) - 4.0) <= 0.5


def test_module_width_scales():
    narrow = render_clean("WEMEET0001", module_px=2.0)
    wide = render_clean("WEMEET0001", module_px=6.0)
    assert wide.shape[1] > narrow.shape[1] * 2.5
