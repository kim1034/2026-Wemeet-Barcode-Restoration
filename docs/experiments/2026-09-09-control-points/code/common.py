"""측정 원함수. 실험 스크립트 셋이 공유한다.

핵심은 remap_G 다. label_sample() 은 정답 제어점으로 펴서 구간을 판정하므로
목표 구간의 정의 자체가 n_x 에 의존한다 -- n_x 를 스윕하면서 표본 정의도 같이
움직이면 측정이 순환한다. 그래서 밀집 대응장 G 로 직접 편 것을 "복원 가능"의
상한으로 삼아 표본을 n_x 와 무관하게 고정한다. (스파이크 13 에서 G 에서 뽑은
제어점이 control_points() 와 차이 0.000000 으로 일치함이 확인돼 있어 이 정의는
기존 것과 연속적이다.)
"""

import math

import cv2
import numpy as np

from scripts.label_recipes import decode, tps_flow  # noqa: F401  (재수출)


def remap_G(obs, g, u_lo, u_hi, out_shape):
    """밀집 대응장으로 직접 편다 = 복원 가능성의 상한. n_x 를 안 쓴다.

    control_points() 와 같은 규약으로 u 를 [u_lo, u_hi] 에 걸친다 -- 크롭으로
    남은 펴진 범위를 다시 0~1 로 정규화하는 것 (설계 §5).
    """
    h_out, w_out = out_shape
    n_v, n_u = g.shape[:2]
    u = u_lo + np.linspace(0.0, 1.0, w_out) * (u_hi - u_lo)
    fu = np.tile((u * (n_u - 1)).astype(np.float32), (h_out, 1))
    fv = np.tile((np.linspace(0.0, 1.0, h_out) * (n_v - 1)).astype(np.float32)[:, None],
                 (1, w_out))
    mx = cv2.remap(g[..., 0].astype(np.float32), fu, fv, cv2.INTER_LINEAR)
    my = cv2.remap(g[..., 1].astype(np.float32), fu, fv, cv2.INTER_LINEAR)
    return cv2.remap(obs, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def true_profile(g, u_lo, u_hi, w_out, v=0.5):
    """정답 src_x(출력 열). remap_G 와 같은 u 격자 위에서 뽑는다."""
    n_v, n_u = g.shape[:2]
    u = u_lo + np.linspace(0.0, 1.0, w_out) * (u_hi - u_lo)
    fu = (u * (n_u - 1)).astype(np.float32)[None, :]
    fv = np.full_like(fu, v * (n_v - 1))
    return cv2.remap(g[..., 0].astype(np.float32), fu, fv, cv2.INTER_LINEAR)[0]


def lam_eff(g, u_lo, u_hi, n=1024, v=0.5):
    """제어점이 표현해야 하는 변위장의 파장. n_x 와 무관하다.

    나이키스트가 걸리는 것은 주름 표면이 아니라 **변위장**이다. J = sqrt(1+z_x^2)
    가 z_x 의 제곱에 걸리므로 변위장의 파장은 주름 파장의 약 절반이다 (실측:
    sine lam=1.0W 의 변위 잔차는 조파 k=2 가 지배). 그래서 레시피의 lam_f 나
    w_c 를 그대로 축으로 쓰지 않고 여기서 직접 잰다.

    돌려주는 것은 (파장, 진폭):
        파장  제어점이 덮는 범위(= 크롭에 남은 펴진 구간)의 몇 배인가.
              레일리 파장 2*pi*sqrt(<d^2>/<d'^2>) -- 연속값이라 표본이 적어도
              층화가 되고, 조파 하나에 고정되지 않는다
        진폭  아핀 성분을 뺀 변위 잔차의 최대 절대값 (관측 px)
    """
    x = true_profile(g, u_lo, u_hi, n, v).astype(np.float64)
    t = np.linspace(0.0, 1.0, n)
    basis = np.stack([np.ones(n), t], axis=1)
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    d = x - basis @ coef
    dp = np.gradient(d, t)
    lam = 2 * math.pi * math.sqrt((d ** 2).mean() / max((dp ** 2).mean(), 1e-30))
    return float(lam), float(np.abs(d).max())


def ridge_curvature(g, u_lo, u_hi, n=257):
    """능선이 굽이치는 정도 = src_x 가 v 에 대해 비선형인 정도 (관측 px).

    n_y 가 필요한 이유는 세로 변형이 아니라 "가로 변형이 높이에 따라 달라지는
    것" 이다. 능선이 곧고 비스듬하면 위상만 밀리므로 src_x 가 v 에 대해 거의
    1차이고 n_y=2 로 표현된다. 굽이치면 1차로 안 맞는다.

    주의: 능선이 곧아도 행별 정규화(s_hat = s / s[:, -1:])와 비선형 역보간
    때문에 잔차가 완전히 0 이 되지는 않는다. 순수한 "굽이침" 지표가 아니다.
    """
    n_v = g.shape[0]
    v = np.linspace(0.0, 1.0, n_v)
    sx = np.stack([true_profile(g, u_lo, u_hi, n, vv) for vv in v]).astype(np.float64)
    basis = np.stack([np.ones(n_v), v], axis=1)
    coef, *_ = np.linalg.lstsq(basis, sx, rcond=None)
    return float(np.abs(sx - basis @ coef).max())


def perturb_src(src_norm, sigma_px, obs_shape, rng):
    """정답 제어점의 src 에 가우시안 좌표 오차를 넣는다. sigma 는 **픽셀**이다.

    정규화로 넣으면 안 된다 -- 버킷마다 크롭 폭이 달라(평균 338/511/800 px)
    같은 정규화 sigma 가 버킷마다 다른 픽셀 오차가 되어 버킷 간 비교가 깨진다.

    dst 는 흔들지 않는다. dst 는 "펴진 뒤의 격자" 라 구성상 결정되는 값이고,
    모델이 실제로 틀리는 것은 "그 내용이 지금 어디 있나"(src) 다.
    """
    if sigma_px <= 0.0:
        return src_norm
    h, w = obs_shape
    scale = np.array([w - 1, h - 1], dtype=np.float64)
    return src_norm + rng.normal(0.0, sigma_px, src_norm.shape) / scale


def rectify_and_error(obs, dst_norm, src_norm, g, u_lo, u_hi, out_shape, v=0.5):
    """한 번의 tps_flow 로 (편 이미지, 최대 국소 배율 오차) 를 같이 낸다.

    이진 판독률은 표본이 적으면 차이를 못 잡는다 -- 실측으로 n_x 4/6/8 이
    14%/12%/15% (N=120) 로 구별되지 않았다. 그래서 연속값 보조 지표를 같이
    기록한다. Code128 은 심볼(11모듈) 안의 상대 폭 비율로 디코딩하므로,
    "정답 대비 국소 배율이 얼마나 틀어졌나" 가 판독 실패와 직접 이어진다.

    tps_flow 가 이 실험에서 가장 비싼 부분이라 디코딩과 오차 측정이 그 결과를
    나눠 쓴다. 따로 부르면 시간이 두 배가 된다.
    """
    h_out, w_out = out_shape
    h_in, w_in = obs.shape
    dst_px = dst_norm * np.array([w_out - 1, h_out - 1])
    src_px = src_norm * np.array([w_in - 1, h_in - 1])
    mx, my = tps_flow(src_px, dst_px, out_shape)
    fixed = cv2.remap(obs, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    row = int(round(v * (h_out - 1)))
    est = mx[row].astype(np.float64)
    true = true_profile(g, u_lo, u_hi, w_out, v).astype(np.float64)
    d_est, d_true = np.gradient(est), np.gradient(true)
    ok = np.abs(d_true) > 1e-9
    ratio = d_est[ok] / d_true[ok]
    return fixed, float(np.abs(ratio - 1.0).max()), float(np.median(np.abs(ratio - 1.0)))


def wilson(k: int, n: int, z: float = 1.96):
    """이항 비율의 Wilson 신뢰구간. 판독률이 0 이나 1 에 붙어도 무너지지 않는다."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(b: int, c: int):
    """짝지은 두 조건의 차이. b = A만 성공, c = B만 성공.

    같은 표본을 모든 셀에 재사용하므로 비교가 짝지어져 있다. 독립 이항검정보다
    검정력이 훨씬 높다 -- 지난 실험이 N=120 에서 14%/12%/15% 를 구별하지 못한
    이유의 절반이 이것이다.
    """
    n = b + c
    if n == 0:
        return 1.0
    # 정확검정(이항, p=0.5) 양측
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))
