import numpy as np

from wemeet.data.render import _BASE_MODULE_PX, module_count, render_clean


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
    img = render_clean("WEMEET0001", module_px=4.0, height_px=200)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape[0] == 200


def test_base_module_width_exact():
    """기본 모듈 폭에서는 최좁은 검은 줄이 정확히 기본값."""
    img = render_clean("WEMEET0001", module_px=_BASE_MODULE_PX, height_px=200)
    narrowest = _narrowest_black_run(img)
    assert narrowest == _BASE_MODULE_PX, f"Expected {_BASE_MODULE_PX}, got {narrowest}"


def test_effective_module_width_fractional():
    """분수 모듈 폭에서 유효 모듈 폭은 ±0.05 px 이내."""
    # 기본 렌더에서 모듈 개수 M 계산
    base_img = render_clean("WEMEET0001", module_px=_BASE_MODULE_PX, height_px=200)
    M = round(base_img.shape[1] / _BASE_MODULE_PX)

    test_values = [1.6, 1.9, 2.5, 3.7, 4.9, 5.5]
    for module_px in test_values:
        img = render_clean("WEMEET0001", module_px=module_px, height_px=200)
        effective = img.shape[1] / M
        error = abs(effective - module_px)
        assert error <= 0.05, (
            f"module_px={module_px}: effective={effective:.4f}, error={error:.4f}"
        )


def test_module_px_1_0_does_not_crash():
    """module_px=1.0 에서도 렌더링 성공."""
    img = render_clean("WEMEET0001", module_px=1.0, height_px=200)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape[0] == 200


def test_module_px_1_2_does_not_crash():
    """module_px=1.2 에서도 렌더링 성공."""
    img = render_clean("WEMEET0001", module_px=1.2, height_px=200)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape[0] == 200


def test_module_width_scales():
    narrow = render_clean("WEMEET0001", module_px=2.0, height_px=200)
    wide = render_clean("WEMEET0001", module_px=6.0, height_px=200)
    assert wide.shape[1] > narrow.shape[1] * 2.5


def test_render_clean_has_no_white_padding():
    """첫 행과 끝 행이 둘 다 막대를 포함한다 — 렌더러 여백이 잘려 나갔다는 뜻."""
    img = render_clean("WEMEET0001", module_px=4.0, height_px=200)
    assert img[0].min() < 128, "첫 행이 전부 흰색이다 — 여백이 남아 있다"
    assert img[-1].min() < 128, "끝 행이 전부 흰색이다 — 여백이 남아 있다"


def test_vertical_resize_does_not_change_columns():
    """세로 크기를 바꿔도 가로 내용이 그대로다 — 여백이 빠졌다는 증거.

    여백이 있으면 리사이즈가 흰 행과 막대 행을 섞는 비율을 바꿔 열 프로파일이
    통째로 흔들린다 (실측 최대 172 계조). 제거 후에는 0.64 계조다.

    "모든 행이 정확히 같은가" 로 재면 안 된다 — cv2.resize 의 반올림 때문에
    크기에 따라 1 계조짜리 행이 두 종류 나오고, 그건 이 수정과 무관하다.
    """
    for module_px in (1.6, 2.0, 4.0, 6.0):
        short = render_clean("WEMEET0001", module_px, height_px=150)
        tall = render_clean("WEMEET0001", module_px, height_px=460)
        drift = float(np.abs(short.mean(axis=0) - tall.mean(axis=0)).max())
        assert drift < 2.0, f"module_px={module_px}: {drift:.2f} 계조"


def test_module_count_matches_rendered_width():
    """래스터화 없이 센 모듈 개수가 실제 렌더 폭과 일치한다."""
    text = "WEMEET0001"
    m = module_count(text)
    for module_px in (1.6, 2.0, 3.7, 6.0):
        img = render_clean(text, module_px, height_px=200)
        assert img.shape[1] == round(m * module_px), f"module_px={module_px}"


def test_module_count_is_constant_across_wrapped_texts():
    """WEMEET0000~9999 는 전부 같은 모듈 개수여야 버킷이 곧 해상도다 (설계 §2)."""
    counts = {module_count(f"WEMEET{i:04d}") for i in range(0, 10000, 13)}
    assert counts == {154}, counts
