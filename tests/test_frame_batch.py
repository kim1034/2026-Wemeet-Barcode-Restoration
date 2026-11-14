"""Deterministic contract tests; no camera, weights, or model downloads."""

import numpy as np
import pytest

from wemeet.schemas import DecodeResult, PipelineResult
from wemeet.sw.frame_batch import Frame, FrameBatch, process_batch, service


def batch(count=8):
    return FrameBatch(
        "session",
        "batch",
        [Frame(f"f-{i}", i * 125, np.full((16, 16, 3), i + 20, np.uint8)) for i in range(count)],
    )


def result(image, text):
    return PipelineResult(
        image,
        None,
        None,
        DecodeResult(text, 0, failure_reason=None if text is not None else "not_detected"),
    )


@pytest.mark.parametrize(
    ("texts", "status", "count"),
    [
        (["00123", "00123"], "confirmed", 2),
        ([None, "00123", "00123"], "confirmed", 3),
        (["00123", None, None], "provisional", 3),
        (["00123", "00456"], "conflict", 2),
        ([None, None, None], "unread", 3),
    ],
)
def test_cross_frame_decisions(monkeypatch, texts, status, count):
    answers = iter(texts)
    monkeypatch.setattr(service, "run", lambda image: result(image, next(answers)))
    output = process_batch(batch())
    assert output.status == status
    assert len(output.attempts) == count
    if status in {"confirmed", "provisional"}:
        assert output.text == "00123"  # Never coerce barcode text to an integer.
    else:
        assert output.text is None
    assert [a["candidate_count"] for a in output.attempts] == list(range(8, 8 - count, -1))


def test_identical_pixels_cannot_supply_two_votes(monkeypatch):
    source = batch()
    source.frames[1:] = [
        Frame(f"copy-{i}", i * 125, source.frames[0].image_bgr.copy()) for i in range(1, 8)
    ]
    monkeypatch.setattr(service, "run", lambda image: result(image, "A"))
    output = process_batch(source)
    assert output.status == "provisional"
    assert output.unique_frame_count == 1
    assert len(output.attempts) == 1


def test_closely_spaced_frames_cannot_confirm(monkeypatch):
    source = batch(2)
    source.frames[1] = Frame("f-1", 10, source.frames[1].image_bgr)
    monkeypatch.setattr(service, "run", lambda image: result(image, "A"))
    output = process_batch(source)
    assert output.status == "provisional"
    assert output.completion_reason == "insufficient_distinct_frames"


def test_selected_index_maps_back_to_original(monkeypatch):
    source = batch(8)
    original_select = service.select_best_frame

    def choose_last(frames, bboxes):
        assert bboxes is None
        selected = original_select(frames, bboxes)
        selected.frame_index = len(frames) - 1
        return selected

    seen = []

    def run(image):
        seen.append(image)
        return result(image, "A" if len(seen) > 1 else None)

    monkeypatch.setattr(service, "select_best_frame", choose_last)
    monkeypatch.setattr(service, "run", run)
    output = process_batch(source)
    assert [a["frame_index"] for a in output.attempts] == [7, 6, 5]
    assert seen[0] is source.frames[7].image_bgr
    assert output.primary_frame.frame_id == "f-7"
    assert output.representative_frame_id == "f-5"


def test_budget_stops_admission_without_cancelling_pipeline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(service.time, "perf_counter", lambda: clock[0])

    def slow(image):
        clock[0] += 2.5
        return result(image, None)

    monkeypatch.setattr(service, "run", slow)
    output = process_batch(batch(), budget_ms=6000)
    assert len(output.attempts) == 2
    assert output.completion_reason == "budget_exhausted"


def test_empty_invalid_and_pipeline_exception(monkeypatch):
    assert process_batch(batch(0)).completion_reason == "no_candidates"
    broken = FrameBatch("s", "b", [Frame("f", 0, np.zeros((3, 3)))])
    assert process_batch(broken).unique_frame_count == 0

    def raising(image):
        raise RuntimeError("stage failed")

    monkeypatch.setattr(service, "run", raising)
    output = process_batch(batch())
    assert len(output.attempts) == 3
    assert all(a["failure_reason"] == "pipeline_error" for a in output.attempts)


def test_selection_exception_returns_failure(monkeypatch):
    def raising(*args, **kwargs):
        raise RuntimeError("selector failed")

    monkeypatch.setattr(service, "select_best_frame", raising)
    assert process_batch(batch()).completion_reason == "selection_error"
