"""wemeet/sw/rectify/ 의 apply_field() 검증.

합성 데이터(wemeet.data)는 쓰지 않는다. 깨끗한 바코드 픽스처와 손으로 만든 제어점만으로
입력·출력 형식, 좌표 방향, 실패 시 통과 동작을 본다.
"""

import cv2
import numpy as np
import pytest

from wemeet.schemas import DetectedBarcode, GeometryField, RectifiedBarcode
from wemeet.sw.decoding import decode
from wemeet.sw.rectify import apply_field
from wemeet.sw.rectify.tps import tps_maps

N_X, N_Y = 16, 3  # 확정 격자 (docs/stage2-handoff.md)
DST = np.array([[x, y] for y in np.linspace(0, 1, N_Y) for x in np.linspace(0, 1, N_X)])


def _detected(bgr: np.ndarray) -> DetectedBarcode:
    return DetectedBarcode(crop_bgr_uint8=bgr, angle_deg_ccw=0.0, confidence=0.9)


def _field(src: np.ndarray, dst: np.ndarray = DST, method: str = "tps") -> GeometryField:
    return GeometryField(
        control_points_dst_norm=dst,
        control_points_src_norm=src,
        method=method,
        confidence=0.9,
    )


def _ramp_bgr(h: int = 60, w: int = 200) -> np.ndarray:
    """가로로 밝기가 증가하는 이미지. 픽셀 값으로 x 좌표를 읽을 수 있다."""
    row = np.linspace(0, 255, w).astype(np.uint8)
    gray = np.tile(row, (h, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------- 출력 형식


@pytest.mark.parametrize("out_scale", [1.0, 1.5, 3.0])
def test_output_is_gray_uint8_with_scaled_shape(clean_barcode_bgr, out_scale):
    h, w = clean_barcode_bgr.shape[:2]

    result = apply_field(_detected(clean_barcode_bgr), _field(DST.copy()), out_scale=out_scale)

    assert isinstance(result, RectifiedBarcode)
    assert result.image_gray_uint8.dtype == np.uint8
    assert result.image_gray_uint8.shape == (round(h * out_scale), round(w * out_scale))
    assert result.field is not None


def test_identity_field_returns_the_same_image(clean_barcode_bgr):
    gray = cv2.cvtColor(clean_barcode_bgr, cv2.COLOR_BGR2GRAY)

    result = apply_field(_detected(clean_barcode_bgr), _field(DST.copy()))

    diff = np.abs(result.image_gray_uint8.astype(int) - gray.astype(int))
    assert diff.max() <= 1


def test_source_is_passed_through(clean_barcode_bgr):
    detected = _detected(clean_barcode_bgr)
    field = _field(DST.copy())

    result = apply_field(detected, field)

    assert result.source is detected
    assert result.field is field


# ---------------------------------------------------------------- 좌표 방향


def test_direction_output_pixel_reads_from_src():
    """src = dst + 0.1 (가로) 이면 출력 x 는 입력 x + 0.1·(w-1) 의 값을 가져와야 한다.

    방향을 뒤집으면 반대쪽(x - 0.1·(w-1))에서 가져온다. 약한 왜곡에서는 디코딩이
    우연히 성공해 이 버그를 못 잡으므로 픽셀 값으로 직접 확인한다.
    """
    bgr = _ramp_bgr()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    src = DST + np.array([0.1, 0.0])

    out = apply_field(_detected(bgr), _field(src), interpolation=cv2.INTER_LINEAR)

    shift = round(0.1 * (w - 1))
    x = 50
    assert abs(int(out.image_gray_uint8[30, x]) - int(gray[30, x + shift])) <= 2


def test_coarse_grid_matches_full_grid_on_curved_field():
    """1/4 격자로 평가해 늘린 매핑이 모든 픽셀 평가와 거의 같다."""
    h, w = 150, 400
    rng = np.random.default_rng(0)
    dst_px = DST * [w - 1, h - 1]
    # 원통처럼 가운데가 늘어나고 가장자리가 압축된 매핑 + 약간의 세로 흔들림
    u = DST[:, 0]
    src_norm = np.stack([0.5 + np.sin((u - 0.5) * 2.0) / (2 * np.sin(1.0)), DST[:, 1]], axis=1)
    src_norm[:, 1] += rng.normal(0, 0.01, len(src_norm))
    src_px = src_norm * [w - 1, h - 1]

    full_x, full_y = tps_maps(dst_px, src_px, (h, w), step=1.0)
    coarse_x, coarse_y = tps_maps(dst_px, src_px, (h, w), step=4.0)

    assert np.abs(full_x - coarse_x).max() < 0.1
    assert np.abs(full_y - coarse_y).max() < 0.1


# ---------------------------------------------------------------- 실제 바코드


def test_mild_warp_still_decodes_after_rectify(clean_barcode_bgr):
    """정답 제어점으로 왜곡한 뒤 같은 제어점으로 펴면 다시 읽힌다 (왕복)."""
    h, w = clean_barcode_bgr.shape[:2]
    u = DST[:, 0]
    # 가장자리가 압축된 원통형 관측: 펴진 u 가 관측의 어디에 있는가
    src_norm = np.stack([0.5 + np.sin((u - 0.5) * 1.6) / (2 * np.sin(0.8)), DST[:, 1]], axis=1)

    # 관측 이미지 만들기: 관측 픽셀 → 펴진 위치 (역방향) 로 원본을 당겨온다
    obs_x, obs_y = tps_maps(src_norm * [w - 1, h - 1], DST * [w - 1, h - 1], (h, w))
    observed = cv2.remap(
        clean_barcode_bgr, obs_x, obs_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    fixed = apply_field(_detected(observed), _field(src_norm))

    assert decode(fixed).text == "WEMEET0001"


# ---------------------------------------------------------------- 실패는 예외가 아니라 통과


def _assert_passthrough(result: RectifiedBarcode, bgr: np.ndarray) -> None:
    assert result.field is None
    assert np.array_equal(result.image_gray_uint8, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))


def test_singular_control_points_pass_through(clean_barcode_bgr):
    # 모든 점이 한 줄 위에 있으면 TPS 의 아핀 항이 퇴화한다
    dst = np.array([[x, 0.5] for x in np.linspace(0, 1, 8)])
    result = apply_field(_detected(clean_barcode_bgr), _field(dst.copy(), dst))
    _assert_passthrough(result, clean_barcode_bgr)


def test_duplicate_control_points_pass_through(clean_barcode_bgr):
    dst = DST.copy()
    dst[1] = dst[0]
    result = apply_field(_detected(clean_barcode_bgr), _field(dst.copy(), dst))
    _assert_passthrough(result, clean_barcode_bgr)


def test_nan_control_points_pass_through(clean_barcode_bgr):
    src = DST.copy()
    field = _field(src)
    field.control_points_src_norm[3, 0] = np.nan  # __post_init__ 이후에 오염된 경우
    result = apply_field(_detected(clean_barcode_bgr), field)
    _assert_passthrough(result, clean_barcode_bgr)


@pytest.mark.parametrize("out_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_out_scale_is_a_programming_error(clean_barcode_bgr, out_scale):
    with pytest.raises(ValueError):
        apply_field(_detected(clean_barcode_bgr), _field(DST.copy()), out_scale=out_scale)
