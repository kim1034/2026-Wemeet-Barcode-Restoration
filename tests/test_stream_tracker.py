"""판독 먼저 추적: 프레임마다 판독한 결과의 박스로 물품을 잇고, 두 장 일치에서 바로 확정한다."""

import numpy as np

from wemeet.schemas import DecodeResult, DetectedBarcode, PipelineResult
from wemeet.sw.frame_batch import Frame
from wemeet.sw.tracking.stream import StreamTracker

IMAGE = np.zeros((1080, 1920, 3), np.uint8)
CROP = np.zeros((60, 120, 3), np.uint8)


def result(x, text=None, *, detected=True, w=200, h=80):
    found = DetectedBarcode(CROP, 0.0, 0.9, (x, 500, w, h)) if detected else None
    failure = None if text else ("decode_failed" if detected else "not_detected")
    return PipelineResult(IMAGE, found, None, DecodeResult(text, 0, failure_reason=failure))


def feed(tracker, observations, start=0, step=125):
    """(x, text, detected) 목록을 125ms 간격 프레임으로 넣고 나온 이벤트를 모두 돌려준다."""
    events = []
    for i, (x, text, detected) in enumerate(observations):
        t = start + i * step
        events += tracker.observe(
            Frame(f"f-{t}", float(t), IMAGE), result(x, text, detected=detected)
        )
    return events


def moving(n, text=None, x0=100, dx=40):
    return [(x0 + i * dx, text, True) for i in range(n)]


def empty(n):
    return [(0, None, False)] * n


def test_two_matching_reads_confirm_immediately_and_only_once():
    tracker = StreamTracker()
    first = feed(tracker, moving(1, "880"))
    assert first == []  # 한 장만으로는 확정하지 않는다
    events = feed(tracker, moving(10, "880"), start=125)
    confirmed = [e for e in events if e["status"] == "confirmed"]
    assert len(confirmed) == 1 and confirmed[0]["text"] == "880" and confirmed[0]["early"]
    assert confirmed[0]["track_end_reason"] is None
    # 물품이 나가도(검출 없음) 이미 확정한 구간은 다시 결과를 내지 않는다
    assert feed(tracker, empty(10), start=2000) == []
    assert tracker.snapshot()["event_counts"] == {"confirmed": 1}


def test_reads_too_close_in_time_do_not_confirm():
    tracker = StreamTracker(min_gap_ms=200)
    events = feed(tracker, moving(2, "880"))  # 125ms 간격 두 장
    assert events == []
    events = feed(tracker, moving(1, "880", x0=180), start=250)
    assert [e["status"] for e in events] == ["confirmed"]


def test_unreadable_passage_is_reported_once_after_it_leaves():
    tracker = StreamTracker()
    events = feed(tracker, moving(8) + empty(8))
    assert [e["status"] for e in events] == ["unread"]
    assert events[0]["track_end_reason"] == "missing"
    assert events[0]["frame_count"] == 8 and not events[0]["early"]


def test_a_brief_unreadable_track_is_not_reported():
    tracker = StreamTracker()
    events = feed(tracker, moving(2) + empty(8))  # 물품이 넘어가는 순간의 짧은 가짜 구간
    assert events == []


def test_single_read_is_provisional_and_different_reads_conflict():
    tracker = StreamTracker()
    events = feed(tracker, moving(4, None)[:3] + [(220, "111", True)] + empty(8))
    assert [(e["status"], e["text"]) for e in events] == [("provisional", "111")]

    tracker = StreamTracker()
    events = feed(tracker, [(100, "111", True), (140, "222", True), (180, None, True)] + empty(8))
    assert [(e["status"], e["text"]) for e in events] == [("conflict", None)]


def test_short_detection_dropouts_do_not_split_the_passage():
    tracker = StreamTracker()  # 끊김 허용 6장
    seen = (
        moving(4, "880")
        + [(260, None, False)] * 5
        + [(460 + i * 40, "880", True) for i in range(4)]
    )
    events = feed(tracker, seen + empty(8))
    assert [e["status"] for e in events] == ["confirmed"]


def test_a_second_number_in_the_same_track_is_confirmed_separately():
    # 바짝 붙은 두 물품이 한 구간으로 합쳐져도, 뒤 물품의 번호가 두 번 읽히면 따로 확정한다.
    tracker = StreamTracker()
    events = feed(tracker, moving(3, "111") + [(220 + i * 40, "222", True) for i in range(3)])
    assert [(e["status"], e["text"]) for e in events] == [
        ("confirmed", "111"),
        ("confirmed", "222"),
    ]
    assert events[0]["track_id"] != events[1]["track_id"]
    assert events[0]["event_id"] < events[1]["event_id"]


def test_results_that_are_not_confirmed_carry_a_representative_frame():
    def observe(tracker, t, x, text, confidence):
        found = DetectedBarcode(CROP, 0.0, confidence, (x, 500, 200, 80))
        failure = None if text else "decode_failed"
        decoded = PipelineResult(IMAGE, found, None, DecodeResult(text, 0, failure_reason=failure))
        return tracker.observe(Frame(f"f-{t}", float(t), IMAGE), decoded)

    def run(steps):
        tracker, events = StreamTracker(), []
        for i, (text, confidence) in enumerate(steps):
            events += observe(tracker, i * 125, 100 + i * 40, text, confidence)
        return events + tracker.close(10_000)

    # 못 읽은 구간은 검출 신뢰도가 가장 높았던 사진
    [unread] = run([(None, 0.5), (None, 0.9), (None, 0.7)])
    assert unread["status"] == "unread" and unread["representative_frame_id"] == "f-125"
    assert unread["_result"].detected.confidence == 0.9
    # 한 번이라도 읽힌 사진이 있으면 신뢰도가 낮아도 그 사진
    [provisional] = run([(None, 0.95), ("111", 0.4), (None, 0.9)])
    assert (
        provisional["status"] == "provisional" and provisional["representative_frame_id"] == "f-125"
    )


def test_close_reports_an_open_unreadable_track():
    tracker = StreamTracker()
    feed(tracker, moving(5))
    events = tracker.close(10_000)
    assert [(e["status"], e["track_end_reason"]) for e in events] == [("unread", "stopped")]
