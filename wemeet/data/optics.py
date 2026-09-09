"""Blinn-Phong 음영과 정반사. 법선은 기하에 쓴 바로 그 기울기에서 나온다.

    I = B * (k_a + k_d * max(0, n.l))  +  k_s * (n.h)^p,   h = (l + v)/|l + v|

정사영 가정이므로 v = (0, 0, 1) 이다.
정반사는 **더한다**. 곱하면 검은 막대가 절대 지워지지 않아 정보를 죽이는
메커니즘 자체가 사라진다 (설계 §4: 더하기 0.275 vs 곱하기 0.005).
"""

import numpy as np

_SATURATION = 0.99


def shade(base, zx, zy, light, ka: float = 0.35, kd: float = 0.50,
          ks: float = 0.0, p: float = 100.0):
    """음영·정반사를 얹고 (이미지, 포화 화소 비율) 을 돌려준다.

    포화 화소 비율은 설계 §7 의 tau 다 — 기하가 소실된 불가 샘플을 가르는
    유일한 신호이므로 여기서 같이 낸다. 렌더 직후라 추가 비용이 없다.
    """
    # 법선 계산: n ∝ (-z_x, -z_y, 1), 정규화됨
    n = np.stack([-zx, -zy, np.ones_like(zx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    # 광벡터 정규화
    l = np.asarray(light, dtype=np.float64)
    l = l / np.linalg.norm(l)

    # 반벡터 계산: h = (l + v) / |l + v|, v = (0, 0, 1)
    hv = l + np.array([0.0, 0.0, 1.0])
    hv /= np.linalg.norm(hv)

    # 확산 성분: max(0, n.l)
    diffuse = np.clip(n @ l, 0.0, None)

    # 정반사 성분: (n.h)^p
    specular = np.clip(n @ hv, 0.0, None) ** p

    # Blinn-Phong 음영: I = B * (k_a + k_d * diffuse) + k_s * specular
    # 정반사는 더한다 (곱하지 않는다)
    img = np.clip(np.asarray(base, dtype=np.float64) / 255.0
                  * (ka + kd * diffuse) + ks * specular, 0.0, 1.0)

    # 포화 화소 비율 계산: (img >= 0.99) 의 비율
    sat = float((img >= _SATURATION).mean())

    return (img * 255).astype(np.uint8), sat
