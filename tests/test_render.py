import numpy as np
from wemeet.data.render import render_clean, _BASE_MODULE_PX


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


def test_base_module_width_exact():
    """기본 모듈 폭에서는 최좁은 검은 줄이 정확히 기본값."""
    img = render_clean("WEMEET0001", module_px=_BASE_MODULE_PX)
    narrowest = _narrowest_black_run(img)
    assert narrowest == _BASE_MODULE_PX, f"Expected {_BASE_MODULE_PX}, got {narrowest}"


def test_effective_module_width_fractional():
    """분수 모듈 폭에서 유효 모듈 폭은 ±0.05 px 이내."""
    # 기본 렌더에서 모듈 개수 M 계산
    base_img = render_clean("WEMEET0001", module_px=_BASE_MODULE_PX)
    M = round(base_img.shape[1] / _BASE_MODULE_PX)

    test_values = [1.6, 1.9, 2.5, 3.7, 4.9, 5.5]
    for module_px in test_values:
        img = render_clean("WEMEET0001", module_px=module_px)
        effective = img.shape[1] / M
        error = abs(effective - module_px)
        assert error <= 0.05, (
            f"module_px={module_px}: effective={effective:.4f}, error={error:.4f}"
        )


def test_module_px_1_0_does_not_crash():
    """module_px=1.0 에서도 렌더링 성공."""
    img = render_clean("WEMEET0001", module_px=1.0)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape[0] == 220


def test_module_px_1_2_does_not_crash():
    """module_px=1.2 에서도 렌더링 성공."""
    img = render_clean("WEMEET0001", module_px=1.2)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape[0] == 220


def test_module_width_scales():
    narrow = render_clean("WEMEET0001", module_px=2.0)
    wide = render_clean("WEMEET0001", module_px=6.0)
    assert wide.shape[1] > narrow.shape[1] * 2.5
