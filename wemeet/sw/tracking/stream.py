"""판독 먼저(decode-first) 추적 — 매 프레임 판독 결과의 검출 박스로 같은 물품을 잇는다.

GPU 에서는 판독 한 장(약 0.1초)이 프레임 간격(약 0.13초)보다 짧아서 매 프레임 판독할 수 있다.
그래서 따로 주기 검출을 하거나 후보를 모았다가 고르지 않는다. 판독(`run()`)이 찾은 박스를
추적 관측으로 그대로 쓰고, 같은 물품에서 서로 다른 두 프레임의 번호가 일치하면 그 자리에서
확정한다. 이미지와 번호는 메모리에서만 쓰고 저장하지 않는다.

같은 물품 판단은 `Tracker` 와 같다(예상 위치와의 거리, 크기 비율). 다만 관측이 프레임마다
들어오므로 검출이 잠깐 끊겨도 구간이 쪼개지지 않도록 끊김 허용(`max_misses`)을 6장으로 둔다.
"""

from collections import Counter
from dataclasses import dataclass, field

from wemeet.schemas import PipelineResult
from wemeet.sw.frame_batch import Frame
from wemeet.sw.tracking.core import Box, Passage, Tracker

# 구간마다 보관하는 판독 기록 상한 (30초 구간 × 8fps 를 넘지 않게)
_MAX_HISTORY = 64


@dataclass
class _Read:
    frame_id: str
    offset_ms: float
    text: str | None
    failure_reason: str | None


@dataclass
class _Track:
    track_id: str
    start_ms: float
    last_ms: float
    detections: int = 0
    history: list = field(default_factory=list)
    confirmed: dict = field(default_factory=dict)  # 확정한 번호 → 이벤트 순번


class StreamTracker:
    """프레임 한 장의 판독 결과를 받아 화면에 보낼 이벤트 목록을 돌려준다.

    이벤트 상태는 기존 판독과 같다: confirmed / provisional / conflict / unread.
    - confirmed: 같은 구간에서 같은 번호가 서로 다른 프레임(`min_gap_ms` 이상 간격)에서 두 번 읽힘.
      통과 중에 바로 낸다(`early=True`).
    - 구간이 끝났는데 확정이 없으면: 한 번만 읽힘 → provisional, 서로 다른 번호 → conflict,
      못 읽음 → unread. 검출이 `min_detections` 장보다 적은 짧은 구간의 unread 는 내지 않는다
      (물품이 넘어가는 순간 생기는 가짜 구간).
    - 한 구간에서 이미 확정한 번호와 **다른 번호**가 다시 두 번 읽히면 별도 결과로 확정한다.
      바짝 붙은 두 물품이 한 구간으로 합쳐졌을 때 뒤 물품을 놓치지 않기 위해서다.
    """

    def __init__(
        self,
        *,
        confirm_reads: int = 2,
        min_gap_ms: float = 100.0,
        max_misses: int = 6,
        min_detections: int = 3,
    ):
        self.confirm_reads = confirm_reads
        self.min_gap_ms = min_gap_ms
        self.min_detections = min_detections
        self.tracker = Tracker(max_misses=max_misses, max_candidates=0)
        self.tracks: dict[str, _Track] = {}
        self.event_sequence = 0
        self.counts = Counter()

    def observe(self, frame: Frame, result: PipelineResult) -> list[dict]:
        detected = result.detected
        bbox = detected.bbox_xywh_original if detected is not None else None
        boxes = [Box(bbox)] if bbox else []
        events = self._finish(self.tracker.update(frame, boxes))
        active = self.tracker.active
        if active is None or not boxes:
            return events
        track = self.tracks.setdefault(
            active.track_id, _Track(active.track_id, active.start_ms, frame.offset_ms)
        )
        track.detections += 1
        track.last_ms = frame.offset_ms
        text = result.decode.text
        track.history.append(
            _Read(frame.frame_id, frame.offset_ms, text, result.decode.failure_reason)
        )
        del track.history[:-_MAX_HISTORY]
        if text is None or text in track.confirmed:
            return events
        same = [read for read in track.history if read.text == text]
        if (
            len(same) >= self.confirm_reads
            and same[-1].offset_ms - same[0].offset_ms >= self.min_gap_ms
        ):
            events.append(self._event(track, "confirmed", text, same, early=True))
            track.confirmed[text] = events[-1]["event_id"]
            active.resolved = True
        return events

    def close(self, now_ms: float) -> list[dict]:
        """입력이 끝났을 때(카메라 중지) 남은 구간을 정리한다."""
        return self._finish(self.tracker.close(now_ms, "stopped"))

    def snapshot(self) -> dict:
        active = self.tracker.active
        track = self.tracks.get(active.track_id) if active is not None else None
        return {
            "active_track": active.track_id if active is not None else None,
            "active_resolved": bool(track and track.confirmed),
            "event_counts": dict(self.counts),
        }

    def _finish(self, passages: list[Passage]) -> list[dict]:
        events = []
        for passage in passages:
            track = self.tracks.pop(passage.track_id, None)
            if track is None or track.confirmed:
                continue
            reads = [read for read in track.history if read.text is not None]
            texts = Counter(read.text for read in reads)
            if not reads:
                if track.detections < self.min_detections:
                    continue
                status, text = "unread", None
            elif len(texts) > 1:
                status, text = "conflict", None
            else:
                status, text = "provisional", reads[0].text
            matching = [read for read in reads if text is not None and read.text == text]
            events.append(self._event(track, status, text, matching, early=False, passage=passage))
        return events

    def _event(self, track: _Track, status, text, matching, *, early, passage=None) -> dict:
        self.event_sequence += 1
        self.counts[status] += 1
        suffix = f"-{len(track.confirmed) + 1}" if track.confirmed else ""
        tried = track.history[-8:]
        return {
            "event_id": self.event_sequence,
            "ok": status == "confirmed",
            "status": status,
            "text": text,
            "early": early,
            "track_id": track.track_id + suffix,
            "batch_id": track.track_id + suffix,
            "track_start_ms": track.start_ms,
            "track_end_ms": passage.end_ms if passage is not None else None,
            "track_end_reason": passage.reason if passage is not None else None,
            "box_source": "barcode",
            "window_ms": (passage.end_ms if passage is not None else track.last_ms)
            - track.start_ms,
            "frame_count": track.detections,
            "unique_frame_count": len(track.history),
            "frame_attempt_count": len(track.history),
            "matching_frame_ids": [read.frame_id for read in matching],
            "completion_reason": "consensus" if status == "confirmed" else "track_ended",
            "attempts": [
                {
                    "frame_id": read.frame_id,
                    "frame_index": index,
                    "offset_ms": read.offset_ms - track.start_ms,
                    "candidate_count": 1,
                    "quality_score": None,
                    "text": read.text,
                    "failure_reason": read.failure_reason,
                }
                for index, read in enumerate(tried)
            ],
        }
