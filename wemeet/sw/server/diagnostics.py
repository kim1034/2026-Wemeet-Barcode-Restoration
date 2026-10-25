"""Live-test summaries without images, barcode values, or client identifiers."""


def batch_diagnostics(body: dict) -> dict:
    groups: dict[str, int] = {}
    attempts = []
    for attempt in body["attempts"]:
        text = attempt["text"]
        group = groups.setdefault(text, len(groups) + 1) if text is not None else None
        reason = attempt["failure_reason"]
        attempts.append(
            {
                "frame_index": attempt["frame_index"],
                "offset_ms": round(attempt["offset_ms"], 1),
                "text_group": group,
                "failure_reason": reason
                if reason
                in {None, "not_detected", "decode_failed", "invalid_input", "pipeline_error"}
                else "unknown",
                "retry_count": attempt["retry_count"],
                "degraded": attempt.get("degraded", False),
                "stage_ms": {
                    stage: round(ms, 1)
                    for stage, ms in attempt["stage_ms"].items()
                    if stage in {"detect", "decode_first", "estimate", "warp", "decode", "retry"}
                },
            }
        )
    return {
        "status": body["status"],
        "completion_reason": body["completion_reason"],
        "frame_count": body["frame_count"],
        "unique_frame_count": body["unique_frame_count"],
        "frame_attempt_count": body["frame_attempt_count"],
        "matching_frame_count": len(body["matching_frame_ids"]),
        "timing_ms": {key: round(value, 1) for key, value in body["timing_ms"].items()},
        "attempts": attempts,
    }
