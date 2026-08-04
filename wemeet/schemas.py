"""AI파트와 SW파트가 주고받는 데이터 형식.

이 파일은 wemeet 안의 어떤 모듈도 import 하지 않는다.
변경하려면 AI·SW 양쪽과 먼저 상의한다 (CONTRIBUTING.md).

이름에 단위를 박아둔 것은 의도한 것이다. angle 이라고만 쓰면 라디안을
넣는 사람이 나오고, crop 이라고만 쓰면 RGB를 넣는 사람이 나온다.

파이프라인은 4단계다 (docs/architecture.md).
    1 검출        AI    -> DetectedBarcode
    2 기하 추정    AI    -> GeometryField      (이미지가 아니다)
    3 기하 보정    SW    -> RectifiedBarcode
    4 디코딩       SW    -> DecodeResult
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectedBarcode:
    """Stage 1 (검출) → Stage 2 (기하 추정)"""

    crop_bgr_uint8: np.ndarray  # BGR 순서, 0~255, shape (H, W, 3)
    angle_deg_ccw: float  # 원본에서 회전 보정한 각도. 도(degree), 반시계
    confidence: float  # 0.0 ~ 1.0

    def __post_init__(self) -> None:
        assert self.crop_bgr_uint8.dtype == np.uint8
        assert self.crop_bgr_uint8.ndim == 3
        assert self.crop_bgr_uint8.shape[2] == 3
        assert 0.0 <= self.confidence <= 1.0


@dataclass
class GeometryField:
    """Stage 2 (기하 추정) → Stage 3 (기하 보정)

    얼마나 휘었는지만 담는다. 이미지 필드가 없으므로 AI 가 픽셀을 만들어
    반환할 방법이 구조적으로 없다. 이것이 이 설계의 핵심이다 —
    생성 모델이 만든 가짜 바코드는 체크섬을 우연히 통과할 수 있다
    (Code128 약 1/103). 틀린 번호가 조용히 통과하는 것은 못 읽는 것보다 나쁘다.

    좌표계 (반드시 지킬 것):
      · 0.0~1.0 정규화 좌표다. 픽셀이 아니다. (x, y) 순서다.
      · 기준은 **DetectedBarcode.crop_bgr_uint8 원본 크기**다.
        모델에 넣으려고 리사이즈한 이미지가 아니고, 원본 사진 전체도 아니다.
      · dst = 펴진 뒤의 격자 위치, src = 그 내용이 지금 있는 위치.

    왜 픽셀이 아니라 정규화인가:
      크롭은 매번 크기가 다르다(바코드가 크게 찍히면 800x600, 작게 찍히면
      200x150). 모델은 고정 크기로 리사이즈해서 받는데, 정규화 좌표는
      **리사이즈에 불변**이라 되돌릴 때 배율을 알 필요가 없다.
      픽셀로 계약했다면 "입력이 256인데 원본이 800이었다"는 정보를
      어딘가로 같이 넘겨야 하고, 그게 빠지면 반드시 사고가 난다.

    letterbox 함정:
      가로로 긴 크롭을 정사각형에 맞추려고 패딩하면, 모델이 내는 좌표는
      "패딩된 이미지" 기준이 된다. 그대로 넘기면 위아래로 밀린 채 펴진다.
      **패딩을 뺀 원본 크롭 기준으로 환산해서 넘겨야 한다.**

    방향을 뒤집으면 왜곡이 두 번 적용된다. 약한 왜곡에서는 우연히
    읽히기도 해서 버그를 놓치기 쉽다 — 사전 실험에서 실제로 겪었다.
    설계 문서(2026-08-04-geometry-pipeline-design.md) §8.5 를 볼 것.
    """

    control_points_dst_norm: np.ndarray  # (N, 2) 펴진 격자. 보통 규칙적인 격자
    control_points_src_norm: np.ndarray  # (N, 2) 휜 이미지에서의 위치
    method: str  # "tps" | "perspective"
    confidence: float  # 0.0 ~ 1.0

    def __post_init__(self) -> None:
        assert self.control_points_dst_norm.shape == self.control_points_src_norm.shape
        assert self.control_points_dst_norm.ndim == 2
        assert self.control_points_dst_norm.shape[1] == 2
        # 4개 미만은 TPS 를 풀 수 없다. 실측으로 8개를 권장한다 (설계 §8.2)
        assert len(self.control_points_dst_norm) >= 4
        assert self.method in ("tps", "perspective")
        assert 0.0 <= self.confidence <= 1.0

        # 픽셀 좌표를 넣는 실수를 여기서 잡는다.
        # dst 는 우리가 만드는 규칙적인 격자이므로 0~1 을 벗어날 이유가 없다.
        assert self.control_points_dst_norm.min() >= 0.0
        assert self.control_points_dst_norm.max() <= 1.0
        # src 는 크롭 밖을 가리킬 수 있다(바코드가 크롭 경계를 넘은 경우).
        # 다만 픽셀 값(수십~수백)과는 자릿수가 다르므로 여유를 두고 막는다.
        assert -0.5 <= self.control_points_src_norm.min()
        assert self.control_points_src_norm.max() <= 1.5


@dataclass
class RectifiedBarcode:
    """Stage 3 (기하 보정) → Stage 4 (디코딩)

    OpenCV 가 픽셀을 이동시킨 결과. 새 픽셀은 없다.
    """

    image_gray_uint8: np.ndarray  # 흑백 단일 채널, 0~255, shape (H, W)
    source: DetectedBarcode
    field: GeometryField | None  # None = 보정하지 않고 통과 (1차 디코딩 시도)

    def __post_init__(self) -> None:
        assert self.image_gray_uint8.dtype == np.uint8
        assert self.image_gray_uint8.ndim == 2


@dataclass
class DecodeResult:
    """Stage 4 (디코딩) → 최종 출력"""

    text: str | None  # 판독 실패 시 None
    symbology: str | None  # 예: "CODE128". 실패 시 None
    retry_count: int  # 0 ~ 3
    stage_ms: dict[str, float] = field(default_factory=dict)
    # 키: detect / decode_first / estimate / warp / decode / retry
    # 예: {"detect": 95.0, "decode_first": 22.0, "estimate": 140.0,
    #      "warp": 18.0, "decode": 26.0}

    failure_reason: str | None = None  # invalid_input / not_detected / decode_failed
    candidate_count: int = 0  # 탐지된 바코드 총 개수
    degraded: bool = False  # 보정하지 못하고 원본 크롭으로 디코딩한 경우

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
    detected: DetectedBarcode | None  # 검출 실패 시 None
    rectified: RectifiedBarcode | None  # 검출 실패 시 None
    decode: DecodeResult  # 실패해도 항상 존재 (text=None)

    @property
    def ok(self) -> bool:
        return self.decode.text is not None
