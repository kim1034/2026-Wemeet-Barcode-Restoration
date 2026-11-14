"""Camera-independent frame selection and cross-frame verification."""

from wemeet.sw.frame_batch.service import process_batch
from wemeet.sw.frame_batch.types import Frame, FrameBatch

__all__ = ["Frame", "FrameBatch", "process_batch"]
