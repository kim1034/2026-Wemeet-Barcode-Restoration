"""검출기 출력 모사. 두 종류를 시그니처로 가른다 (설계 §5).

    geometric*  좌표(G)를 반드시 함께 받고 함께 돌려준다
    photometric 좌표를 받지도 않는다

여백은 -2% ~ +15% 다. -6% 는 Code128 의 start/stop 이 잘려 어떤 제어점으로도
못 읽는다 (실측). 검출기가 바코드를 잘라먹으면 그건 검출 실패다.
"""

import cv2
import numpy as np

MARGIN_MIN = -0.02
MARGIN_MAX = 0.15


def geometric_rotate(img: np.ndarray, g: np.ndarray, deg: float):
    """이미지를 회전하고 대응장에 같은 아핀을 곱한다.

    G 가 펴진->관측 방향이라 저장된 좌표에 행렬을 곱하는 것으로 끝난다.
    """
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), deg, 1.0)
    out = cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    homogeneous = np.concatenate([g, np.ones(g.shape[:2] + (1,))], axis=-1)
    return out, homogeneous @ matrix.T


def geometric_margin(
    img: np.ndarray, g: np.ndarray, left: float, right: float, top: float, bottom: float
):
    """여백을 더하거나(양수) 잘라먹는다(음수). 대응장을 같이 옮긴다.

    돌려주는 u_lo/u_hi 는 남은 크롭이 대응하는 펴진 범위다. 잘라먹으면
    좁아진다 — 크롭 뒤 제어점은 변환으로 얻을 수 없으므로 이 범위를
    control_points 에 넘겨 다시 정규화해야 한다.
    """
    h, w = img.shape
    fracs = [float(np.clip(f, MARGIN_MIN, MARGIN_MAX)) for f in (left, right, top, bottom)]
    dl, dr = int(round(fracs[0] * w)), int(round(fracs[1] * w))
    dt, db = int(round(fracs[2] * h)), int(round(fracs[3] * h))

    pad = [max(0, dl), max(0, dr), max(0, dt), max(0, db)]
    out = cv2.copyMakeBorder(img, pad[2], pad[3], pad[0], pad[1], cv2.BORDER_REPLICATE)
    moved = g + np.array([pad[0], pad[2]], dtype=np.float64)

    x0, y0 = max(0, -dl), max(0, -dt)
    x1 = out.shape[1] - max(0, -dr)
    y1 = out.shape[0] - max(0, -db)
    out = out[y0:y1, x0:x1].copy()
    moved = moved - np.array([x0, y0], dtype=np.float64)

    hh, ww = out.shape
    inside = ((moved[..., 0] >= 0) & (moved[..., 0] <= ww - 1)).all(axis=0)
    us = np.linspace(0.0, 1.0, g.shape[1])
    if not inside.any():
        return out, moved, 0.0, 1.0
    return out, moved, float(us[inside][0]), float(us[inside][-1])


def photometric(
    img: np.ndarray, rng, sigma: float = 0.0, noise: float = 0.0, jpeg: int | None = None
) -> np.ndarray:
    """초점 블러 -> 센서 노이즈 -> JPEG. 좌표는 건드리지 않는다."""
    out = img
    if sigma > 0.05:
        out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigma)
        out = np.clip(out, 0, 255).astype(np.uint8)
    if noise > 0.0:
        out = np.clip(out.astype(np.float64) + rng.normal(0.0, noise, out.shape), 0, 255).astype(
            np.uint8
        )
    if jpeg is not None:
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg)])
        if not ok:
            # 조용히 무시하면 jpeg=85 라 적어놓고 실제로는 무손실 이미지를 내보낸다 --
            # 레시피가 자기가 만드는 이미지를 더 이상 설명하지 못하게 된다.
            raise RuntimeError(f"JPEG 인코딩 실패: quality={jpeg}, shape={out.shape}")
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out
