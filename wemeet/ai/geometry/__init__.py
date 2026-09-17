"""Stage 2 (기하 추정) — 대응점 좌표로 왜곡 정도만 예측한다. 이미지는 반환하지 않는다.

@AI파트가 구현할 패키지입니다. docs/architecture.md, docs/decisions/0003 참고.

여러 파일로 나눠 구현해도 됩니다 — 바깥(`wemeet.sw.pipeline`)에는 이
`__init__.py` 가 내보내는 `estimate_geometry()` 하나만 보이면 됩니다.

이 패키지는 wemeet.sw / wemeet.data 를 import 하지 않습니다 (import-linter
규칙, docs/architecture.md 「의존 방향」). 공용이 필요하면 wemeet.schemas
만 씁니다.
"""

from wemeet.schemas import DetectedBarcode, GeometryField


def estimate_geometry(target: DetectedBarcode) -> GeometryField:
    """크롭 하나가 얼마나 휘었는지, 대응점 좌표로만 예측한다.

    좌표계 규칙 (wemeet/schemas.py 의 GeometryField 문서 반드시 참고):
    - 0.0~1.0 정규화 좌표, 기준은 이 target.crop_bgr_uint8 의 원본 크기
    - control_points_dst_norm: 펴진 뒤의 격자 위치
    - control_points_src_norm: 지금 왜곡된 이미지에서의 위치 (이게 실제 예측 대상)
    - 개수는 4개 이상이면 됨 (0003 문서 기준 8개 이상 권장, 기본안은 4x3=12개)
    """
    raise NotImplementedError("AI파트가 구현할 부분입니다.")
