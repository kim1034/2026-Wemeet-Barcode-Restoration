"""Passage identity, sparse observations, bounded memory and ROI contracts."""

import numpy as np
import pytest

from wemeet.schemas import DecodeResult, DetectedBarcode, PipelineResult
from wemeet.sw.frame_batch import Frame, FrameBatch, process_batch, service
from wemeet.sw.tracking import BarcodeBoxes, Box, MotionBoxes, Tracker
from wemeet.sw.tracking.session import TrackingSession


def frame(t, x=20):
    image = np.zeros((180, 320, 3), np.uint8)
    image[60:100, x : x + 40] = int(t) % 150 + 80
    return Frame(f"f-{t}", t, image)


def test_one_passage_crosses_windows_then_same_product_returns_as_new_track():
    tracker = Tracker(max_misses=2)
    done = []
    for base in (0, 3000):
        for i in range(15):
            t = base + i * 125
            x = 20 + i * 5
            done += tracker.update(frame(t, x), [Box((x, 60, 40, 40))])
        done += tracker.update(frame(base + 1875), [])
        done += tracker.update(frame(base + 2000), [])
    assert [p.track_id for p in done] == ["t-1", "t-2"]
    assert all(len(p.frames) == 8 for p in done)
    assert all(p.reason == "missing" for p in done)
    assert all(p.batch("s").use_roi for p in done)
    assert tracker.update(frame(6000), []) == []


def test_skipped_detection_is_not_a_negative_and_prediction_exits_view():
    tracker = Tracker()
    tracker.update(frame(0), [Box((200, 60, 40, 40))])
    tracker.update(frame(125), [Box((240, 60, 40, 40))])
    assert not tracker.update(frame(250), None)
    assert tracker.active.misses == 0
    ended = tracker.update(frame(375), None)
    assert len(ended) == 1 and ended[0].reason == "left_view"


def test_second_observation_corrects_buffered_roi_predictions():
    tracker = Tracker()
    tracker.update(frame(0), [Box((20, 60, 40, 40))])
    tracker.update(frame(125, 40), None)
    tracker.update(frame(250, 60), [Box((60, 60, 40, 40))])
    middle = tracker.active.frames[1]
    assert middle.bbox_xywh == (40, 60, 40, 40)
    assert middle.image_bgr[70, 45].any()


def test_new_distant_box_does_not_join_old_item():
    tracker = Tracker()
    tracker.update(frame(0), [Box((250, 60, 40, 40))])
    tracker.update(frame(125), [Box((270, 60, 40, 40))])
    ended = tracker.update(frame(250), [Box((0, 60, 40, 40))])
    assert ended[0].reason == "association_break"
    assert tracker.active.track_id == "t-2"


def test_gap_resolution_and_clock_guards():
    tracker = Tracker()
    tracker.update(frame(0), [Box((20, 60, 40, 40))])
    assert tracker.update(frame(3000), [Box((20, 60, 40, 40))])[0].reason == "input_gap"
    with pytest.raises(ValueError):
        tracker.update(frame(2999), [])
    changed = Frame("new-size", 3125, np.zeros((90, 160, 3), np.uint8))
    assert tracker.update(changed, [Box((10, 10, 20, 20))])[0].reason == "resolution_changed"


def test_stationary_item_does_not_emit_every_duration_limit():
    tracker = Tracker(max_track_ms=500)
    ended = []
    for t in range(0, 2000, 125):
        ended += tracker.update(frame(t), [Box((20, 60, 40, 40))])
    assert len(ended) == 1
    assert tracker.holding
    tracker.update(frame(2000), [])
    tracker.update(frame(2125), [])
    tracker.update(frame(2250), [Box((20, 60, 40, 40))])
    assert tracker.active.track_id == "t-2"


def test_barcode_source_is_sparse_and_requires_explicit_coordinates(monkeypatch):
    calls = []

    def detect(image):
        calls.append(image)
        return DetectedBarcode(image, 0, 1, (2, 3, 4, 5))

    monkeypatch.setattr("wemeet.sw.tracking.sources.detect", detect)
    source = BarcodeBoxes(interval_ms=2000)
    assert source.observe(frame(0))[0].xywh == (2, 3, 4, 5)
    assert all(source.observe(frame(t)) is None for t in range(125, 2000, 125))
    assert source.observe(frame(2000))
    assert len(calls) == 2
    monkeypatch.setattr(
        "wemeet.sw.tracking.sources.detect", lambda image: DetectedBarcode(image, 0, 1)
    )
    with pytest.raises(ValueError):
        source.observe(frame(4000))


def test_motion_provider_keeps_stationary_foreground_and_labels_fallback():
    source = MotionBoxes()
    blank = Frame("blank", 0, np.zeros((180, 320, 3), np.uint8))
    assert source.observe(blank) == []
    assert source.observe(frame(125))[0].source == "motion"
    assert source.observe(frame(5000))[0].source == "motion"
    assert source.observe(blank) == []


def test_roi_reaches_selector_and_pipeline_still_receives_full_original(monkeypatch):
    frames = [Frame(f"f{i}", i * 125, frame(i * 125).image_bgr, (20, 60, 40, 40)) for i in range(3)]
    original = service.select_best_frame
    calls = []

    def select(images, bboxes):
        calls.append(bboxes)
        assert bboxes and all(box == (20, 60, 40, 40) for box in bboxes)
        return original(images, bboxes)

    def run(image):
        assert image.shape == (180, 320, 3)
        return PipelineResult(image, None, None, DecodeResult("00123", 0))

    monkeypatch.setattr(service, "select_best_frame", select)
    monkeypatch.setattr(service, "run", run)
    out = process_batch(FrameBatch("s", "t", frames, use_roi=True))
    assert out.status == "confirmed" and len(calls) == 2
    assert out.attempts[0]["score_scope"] == "remaining_barcode_rois"


def test_pending_queue_bound_and_discard_release_images():
    source = MotionBoxes()
    session = TrackingSession("s", source, max_pending=1)
    for i in range(3):
        tracker = Tracker()
        tracker.update(frame(i * 1000), [Box((20, 60, 40, 40))])
        passage = tracker.close(i * 1000 + 100, "test")[0]
        passage.track_id = f"t-{i}"
        session.enqueue([passage])
    assert len(session.pending) == 1 and session.dropped == 2
    assert all(body["status"] == "unread" for _, body in session.results)
    session.discard()
    assert not session.pending and not session.results and session.closed


def test_result_retention_is_bounded_and_completion_cannot_revive_closed_session():
    session = TrackingSession("s", MotionBoxes())
    tracker = Tracker()
    tracker.update(frame(0), [Box((20, 60, 40, 40))])
    passage = tracker.close(125, "missing")[0]
    for i in range(40):
        passage.track_id = f"t-{i}"
        session.complete(passage, {"track_id": passage.track_id, "status": "unread"})
    state = session.snapshot(after=35)
    assert len(session.results) == 10 and len(state["results"]) == 5
    assert state["oldest_result_sequence"] == 31
    session.discard()
    session.enqueue([passage])
    session.complete(passage, {"track_id": passage.track_id, "status": "unread"})
    assert not session.pending and not session.results


class FixedBoxes:
    def observe(self, frame):
        return [Box((20, 60, 40, 40))]


def _visible_session(frames=8):
    session = TrackingSession("s", FixedBoxes())
    for i in range(frames):
        session.ingest([frame(i * 125)])
    return session


def _confirmed(track):
    return {"track_id": track.track_id, "status": "confirmed", "text": "00123"}


def test_interim_waits_for_enough_new_candidates_and_caps_its_attempts():
    session = TrackingSession("s", FixedBoxes())
    clock = iter(range(0, 10**6, 125))

    def feed(count):
        for _ in range(count):
            session.ingest([frame(next(clock))])

    feed(1)
    assert session.take_interim() is None  # 후보 1장: 서로 다른 두 프레임이 없다
    feed(4)
    track, snapshot = session.take_interim()
    assert snapshot is not track and snapshot.frames is not track.frames
    session.release_interim(track)
    assert session.take_interim() is None  # 새 후보가 4장 쌓일 때까지 다시 하지 않는다
    for _ in range(2):
        feed(4)
        taken = session.take_interim()
        assert taken is not None
        session.release_interim(taken[0])
    feed(4)
    assert session.take_interim() is None  # 못 읽는 바코드가 슬롯을 독점하지 못한다


def test_busy_slot_does_not_spend_an_interim_attempt():
    session = _visible_session()
    track, _ = session.take_interim()
    session.release_interim(track, retry=True)
    assert track.interim_runs == 0
    assert session.take_interim() is not None


def test_confirmed_interim_result_is_the_only_result_for_that_passage():
    session = _visible_session()
    track, _ = session.take_interim()
    session.complete_interim(track, _confirmed(track))
    assert track.resolved and session.snapshot()["active_resolved"]
    assert session.take_interim() is None
    ended = session.tracker.close(2000, "left_view")
    session.enqueue(ended)
    assert not session.pending and session.take_pending() is None
    assert [body["status"] for _, body in session.results] == ["confirmed"]
    assert session.finished == 1


def test_unconfirmed_interim_is_not_reported_and_the_passage_is_decoded_at_exit():
    session = _visible_session()
    track, _ = session.take_interim()
    session.complete_interim(track, {"track_id": track.track_id, "status": "provisional"})
    assert not track.resolved and not session.results
    session.enqueue(session.tracker.close(2000, "left_view"))
    assert session.take_pending() is track


def test_passage_that_ends_during_interim_decode_is_not_decoded_again():
    session = _visible_session()
    track, _ = session.take_interim()
    session.enqueue(session.tracker.close(2000, "left_view"))
    assert len(session.pending) == 1  # 판독 중에 퇴장해 이미 대기열에 들어갔다
    session.complete_interim(track, _confirmed(track))
    assert not session.pending
    assert len(session.results) == 1


def test_exit_pending_passage_takes_priority_over_a_new_interim():
    session = _visible_session()
    session.enqueue(session.tracker.close(1000, "left_view"))
    assert session.take_interim() is None  # 추적 중인 트랙이 없다
    assert session.take_pending() is not None
