"""AI파트와 SW파트가 주고받는 데이터 형식.

이 파일은 wemeet 안의 어떤 모듈도 import 하지 않는다.
변경하려면 AI·SW 양쪽과 먼저 상의한다 (CONTRIBUTING.md).

이름에 단위를 박아둔 것은 의도한 것이다. angle 이라고만 쓰면 라디안을
넣는 사람이 나오고, crop 이라고만 쓰면 RGB를 넣는 사람이 나온다.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectedBarcode:
    """Stage 1 (탐지) → Stage 2 (복원)"""

    crop_bgr_uint8: np.ndarray  # BGR 순서, 0~255, shape (H, W, 3)
    angle_deg_ccw: float  # 원본에서 회전 보정한 각도. 도(degree), 반시계
    confidence: float  # 0.0 ~ 1.0

    def __post_init__(self) -> None:
        assert self.crop_bgr_uint8.dtype == np.uint8
        assert self.crop_bgr_uint8.ndim == 3
        assert self.crop_bgr_uint8.shape[2] == 3
        assert 0.0 <= self.confidence <= 1.0


@dataclass
class RestoredBarcode:
    """Stage 2 (복원) → Stage 3 (디코딩)"""

    image_gray_uint8: np.ndarray  # 흑백 단일 채널, 0~255, shape (H, W)
    source: DetectedBarcode

    def __post_init__(self) -> None:
        assert self.image_gray_uint8.dtype == np.uint8
        assert self.image_gray_uint8.ndim == 2


@dataclass
class DecodeResult:
    """Stage 3 (디코딩) → 최종 출력"""

    text: str | None  # 판독 실패 시 None
    symbology: str | None  # 예: "CODE128". 실패 시 None
    retry_count: int  # 0 ~ 3
    stage_ms: dict[str, float] = field(default_factory=dict)
    # 예: {"detect": 42.1, "restore": 210.3, "decode": 18.7, "retry": 95.0}

    failure_reason: str | None = None  # invalid_input / not_detected / decode_failed
    candidate_count: int = 0  # 탐지된 바코드 총 개수
    degraded: bool = False  # 복원 실패로 원본 크롭을 쓴 경우

    def __post_init__(self) -> None:
        assert 0 <= self.retry_count <= 3
        if self.text is None:
            assert self.symbology is None
            assert self.failure_reason is not None  # 실패에는 항상 이유가 있다
        else:
            assert self.failure_reason is None

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


@dataclass
class PipelineResult:
    """파이프라인 최종 반환형. UI가 필요한 모든 것을 담는다.

    기획서 4.1.2 5단계("복원 이미지와 디코딩 텍스트를 UI에 동시 매핑")와
    7.2("복원 이미지와 판독 결과를 동시 표출하는 투명한 파이프라인")를
    만족하려면 중간 산출물이 최종 반환형까지 살아 있어야 한다.
    """

    original_bgr_uint8: np.ndarray  # 입력 원본 (대조 표시용)
    detected: DetectedBarcode | None  # 탐지 실패 시 None
    restored: RestoredBarcode | None  # 탐지 실패 시 None
    decode: DecodeResult  # 실패해도 항상 존재 (text=None)

    @property
    def ok(self) -> bool:
        return self.decode.text is not None
