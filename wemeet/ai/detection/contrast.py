"""검출 입력 전용 대비 향상(CLAHE).

LAB 의 L 채널에만 적용해 색을 보존한다. 이 결과는 오직 검출 모델 입력으로만
쓴다 — 실제 크롭은 이 결과가 아니라 원본 이미지에서 뜬다. Stage 2 학습
데이터가 CLAHE 를 전제하지 않으므로, 크롭 자체를 CLAHE 처리된 픽셀로
넘기면 Stage 2 학습/실전 도메인이 어긋난다. CLAHE 는 이미 하얗게 날아간
(포화) 반사 픽셀은 복원하지 못하므로 반사 대응을 대체하지 않는다 —
저대비·저조도 상황에서 검출기가 바코드 경계를 더 잘 보도록 돕는
보완 수단일 뿐이다.

**주의**: `docs/parts/sw.md`·`docs/decisions/0006` 은 CLAHE 를 Stage 4
디코딩(SW파트, `wemeet/sw/decoding/`) 쪽에 배치한다. 이 모듈은 그와 별개로
Stage 1 검출 입력의 저대비 상황을 돕기 위해 추가한 것이고, Stage 4 CLAHE
도입 여부와는 무관하다.
"""

import cv2
import numpy as np

DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = (8, 8)


def enhance_for_detection(
    image_bgr: np.ndarray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
