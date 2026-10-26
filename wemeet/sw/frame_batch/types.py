"""SW-local contracts. HTTP and camera sources both produce FrameBatch."""

from dataclasses import dataclass, field

import numpy as np

from wemeet.schemas import PipelineResult


@dataclass(frozen=True)
class Frame:
    frame_id: str
    offset_ms: float
    image_bgr: np.ndarray
    bbox_xywh: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class FrameBatch:
    session_id: str
    batch_id: str
    frames: list[Frame]
    window_ms: float = 1000.0
    use_roi: bool = False


@dataclass
class BatchOutcome:
    status: str
    completion_reason: str
    text: str | None = None
    attempts: list[dict] = field(default_factory=list)
    matching_frame_ids: list[str] = field(default_factory=list)
    representative: PipelineResult | None = None
    representative_frame_id: str | None = None
    primary_frame: Frame | None = None
    unique_frame_count: int = 0
    selection_ms: float = 0.0
    pipeline_ms: float = 0.0
    processing_ms: float = 0.0
