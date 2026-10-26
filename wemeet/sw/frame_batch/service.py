"""Use the unchanged AI selector and single-image pipeline, at most three times."""

import hashlib
import time

import numpy as np

from wemeet.ai.best_frame import select_best_frame
from wemeet.sw.frame_batch.types import BatchOutcome, Frame, FrameBatch
from wemeet.sw.pipeline import run

MAX_ATTEMPTS = 3
BUDGET_MS = 9000.0
NEXT_PIPELINE_RESERVE_MS = 2500.0
SELECTION_PER_FRAME_MS = 39.0
MIN_FRAME_GAP_MS = 100.0


def _unique_frames(frames: list[Frame]) -> list[tuple[int, Frame]]:
    unique, seen_ids, seen_pixels = [], set(), set()
    for index, frame in enumerate(frames):
        image = frame.image_bgr
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or not image.size
        ):
            continue
        digest = hashlib.sha256(image.tobytes()).digest()
        if frame.frame_id in seen_ids or digest in seen_pixels:
            continue
        seen_ids.add(frame.frame_id)
        seen_pixels.add(digest)
        unique.append((index, frame))
    return unique


def process_batch(batch: FrameBatch, *, budget_ms: float = BUDGET_MS) -> BatchOutcome:
    """A confirmed result means two agreeing observations, not guaranteed ground truth.

    The budget only controls admission of another attempt. A running pipeline is
    never claimed to be cancellable. Scores belong to each remaining candidate set.
    """
    start = time.perf_counter()
    outcome = BatchOutcome("unread", "no_candidates")
    remaining = _unique_frames(batch.frames)
    outcome.unique_frame_count = len(remaining)
    observed: dict[str, list[str]] = {}
    used_times: list[float] = []

    while remaining and len(outcome.attempts) < MAX_ATTEMPTS:
        candidates = [
            item
            for item in remaining
            if all(abs(item[1].offset_ms - t) >= MIN_FRAME_GAP_MS for t in used_times)
        ]
        if not candidates:
            outcome.completion_reason = "insufficient_distinct_frames"
            break
        elapsed_ms = (time.perf_counter() - start) * 1000
        reserve = NEXT_PIPELINE_RESERVE_MS + len(candidates) * SELECTION_PER_FRAME_MS
        if elapsed_ms + reserve > budget_ms:
            outcome.completion_reason = "budget_exhausted"
            break
        selection_start = time.perf_counter()
        rois = [frame.bbox_xywh for _, frame in candidates]
        use_roi = batch.use_roi and all(box is not None for box in rois)
        try:
            best = select_best_frame(
                [frame.image_bgr for _, frame in candidates], bboxes=rois if use_roi else None
            )
        except Exception:
            outcome.completion_reason = "selection_error"
            break
        finally:
            outcome.selection_ms += (time.perf_counter() - selection_start) * 1000
        original_index, frame = candidates[best.frame_index]
        remaining = [item for item in remaining if item[0] != original_index]
        used_times.append(frame.offset_ms)
        if outcome.primary_frame is None:
            outcome.primary_frame = frame
        attempt = {
            "frame_id": frame.frame_id,
            "frame_index": original_index,
            "offset_ms": frame.offset_ms,
            "candidate_count": len(candidates),
            "quality_score": float(best.quality_score),
            "sharpness_score": float(best.sharpness_score),
            "glare_ratio": float(best.glare_ratio),
            "low_quality_fallback": bool(best.low_quality_fallback),
            "score_scope": "remaining_barcode_rois" if use_roi else "remaining_full_frames",
            "text": None,
            "failure_reason": None,
            "stage_ms": {},
            "retry_count": 0,
        }
        pipeline_start = time.perf_counter()
        result = None
        try:
            result = run(frame.image_bgr)
            attempt.update(
                text=result.decode.text,
                failure_reason=result.decode.failure_reason,
                stage_ms=result.decode.stage_ms,
                retry_count=result.decode.retry_count,
                degraded=result.decode.degraded,
            )
        except Exception:
            # The adapter remains available even if an underlying stage raises.
            attempt["failure_reason"] = "pipeline_error"
        attempt["wall_ms"] = (time.perf_counter() - pipeline_start) * 1000
        outcome.pipeline_ms += attempt["wall_ms"]
        outcome.attempts.append(attempt)
        if result is not None and outcome.representative is None:
            outcome.representative = result
            outcome.representative_frame_id = frame.frame_id
        if attempt["text"] is not None:
            text = attempt["text"]
            observed.setdefault(text, []).append(frame.frame_id)
            if len(observed) > 1:
                outcome.status = "conflict"
                outcome.text = None
                outcome.completion_reason = "conflicting_texts"
                break
            outcome.representative = result
            outcome.representative_frame_id = frame.frame_id
            outcome.text = text
            outcome.status = "provisional"
            outcome.matching_frame_ids = list(observed[text])
            if len(observed[text]) >= 2:
                outcome.status = "confirmed"
                outcome.completion_reason = "consensus"
                break
        outcome.completion_reason = (
            "max_attempts" if len(outcome.attempts) == MAX_ATTEMPTS else "candidates_exhausted"
        )

    outcome.processing_ms = (time.perf_counter() - start) * 1000
    return outcome
