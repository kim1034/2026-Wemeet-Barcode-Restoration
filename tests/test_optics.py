import numpy as np

from wemeet.data.optics import shade
from wemeet.data.surface import grad_crease


def test_specular_is_added_not_multiplied():
    """설계 §4. 곱하면 0 * 무엇 = 0 이라 검은 막대가 절대 안 지워진다."""
    h, w = 8, 301
    zx, zy = grad_crease(w, h, slope=1.5, w_c=4.0)
    black = np.zeros((h, w), dtype=np.uint8)
    img, _ = shade(black, zx, zy, light=(0.3, -0.2, 1.0), ks=0.9, p=40.0)
    assert img.max() > 60, "검은 바탕이 정반사로 밝아져야 한다 (더하기)"


def test_no_specular_leaves_black_black():
    h, w = 8, 301
    zx, zy = grad_crease(w, h, slope=1.5, w_c=4.0)
    black = np.zeros((h, w), dtype=np.uint8)
    img, _ = shade(black, zx, zy, light=(0.3, -0.2, 1.0), ks=0.0)
    assert img.max() == 0


def test_highlight_sits_on_the_crease():
    """설계 §10-8. 하이라이트가 곡률 최대점에서 5px 안. 무작위 마스크는 86.8px."""
    h, w = 8, 401
    zx, zy = grad_crease(w, h, slope=1.5, w_c=4.0, offset=0.0)
    black = np.zeros((h, w), dtype=np.uint8)
    img, _ = shade(black, zx, zy, light=(0.0, 0.0, 1.0), ks=0.9, p=60.0)

    highlight_col = int(np.argmax(img[h // 2]))
    curvature_col = int(np.argmax(np.abs(np.gradient(zx[h // 2]))))
    assert abs(highlight_col - curvature_col) <= 5


def test_saturation_ratio_is_zero_without_specular():
    h, w = 8, 101
    zx, zy = np.zeros((h, w)), np.zeros((h, w))
    white = np.full((h, w), 255, dtype=np.uint8)
    _, sat = shade(white, zx, zy, light=(0.0, 0.0, 1.0), ks=0.0)
    assert sat == 0.0, "ka + kd = 0.85 이므로 포화가 없어야 한다"


def test_saturation_ratio_rises_with_specular():
    h, w = 8, 101
    zx, zy = np.zeros((h, w)), np.zeros((h, w))
    white = np.full((h, w), 255, dtype=np.uint8)
    _, sat = shade(white, zx, zy, light=(0.0, 0.0, 1.0), ks=1.0, p=1.0)
    assert sat > 0.9


def test_flat_surface_uses_ambient_plus_full_diffuse():
    h, w = 4, 16
    zx, zy = np.zeros((h, w)), np.zeros((h, w))
    white = np.full((h, w), 255, dtype=np.uint8)
    img, _ = shade(white, zx, zy, light=(0.0, 0.0, 1.0), ks=0.0)
    assert abs(float(img.mean()) - 0.85 * 255) < 1.0
