"""One bounded capture session, reusable by HTTP or a local camera adapter."""

import json
import threading
import time
from collections import deque
from dataclasses import replace

from wemeet.sw.frame_batch import process_batch
from wemeet.sw.tracking.core import Tracker

# 통과 중 판독: 후보가 이만큼 새로 쌓일 때마다, 트랙당 최대 이 횟수까지 시도한다.
# 계속 못 읽는 바코드가 추론 슬롯을 독점해 재검출을 굶기지 않게 상한을 둔다.
INTERIM_MIN_NEW_FRAMES = 4
INTERIM_MAX_RUNS = 3


class TrackingSession:
    def __init__(self, session_id, source, *, max_pending=2, tracker=None):
        self.session_id = session_id
        self.source = source
        self.tracker = tracker or Tracker()
        self.pending = deque()
        self.results = deque(maxlen=10)
        self.max_pending = max_pending
        self.lock = threading.RLock()
        self.ingest_lock = threading.Lock()
        self.last_activity = time.monotonic()
        self.last_sequence = -1
        self.last_fingerprint = None
        self.closed = False
        self.decode_busy = False
        self.dropped = 0
        self.skipped_detections = 0
        self.received = 0
        self.finished = 0
        self.result_sequence = 0
        self.result_bytes = 0

    def _store(self, track_id, body):
        self.result_sequence += 1
        body["result_sequence"] = self.result_sequence
        size = len(json.dumps(body).encode())
        while self.results and (len(self.results) >= 10 or self.result_bytes + size > 16 * 1024**2):
            _, expired = self.results.popleft()
            self.result_bytes -= len(json.dumps(expired).encode())
        if size > 16 * 1024**2:
            body = {
                "track_id": track_id,
                "status": body["status"],
                "text": body.get("text"),
                "result_sequence": self.result_sequence,
                "completion_reason": "preview_too_large",
                "frame_attempt_count": body.get("frame_attempt_count", 0),
            }
            size = len(json.dumps(body).encode())
        self.results.append((track_id, body))
        self.result_bytes += size

    def ingest(self, frames):
        for frame in frames:
            with self.lock:
                if self.closed:
                    return
            boxes = self.source.observe(frame)
            with self.lock:
                if self.closed:
                    return
                self.received += 1
                self.enqueue(self.tracker.update(frame, boxes))

    def enqueue(self, passages):
        with self.lock:
            if self.closed:
                return
            for passage in passages:
                if passage.resolved:
                    continue  # 통과 중 이미 성공해 결과를 냈다. 같은 물품을 두 번 내지 않는다.
                if len(self.pending) >= self.max_pending:
                    self.dropped += 1
                    # Explicit unread result for this known passage, no fake vote.
                    self._store(
                        passage.track_id,
                        {
                            "track_id": passage.track_id,
                            "track_start_ms": passage.start_ms,
                            "track_end_ms": passage.end_ms,
                            "track_end_reason": passage.reason,
                            "window_ms": passage.end_ms - passage.start_ms,
                            "box_source": passage.box.source,
                            "frame_count": len(passage.frames),
                            "status": "unread",
                            "completion_reason": "queue_full",
                            "frame_attempt_count": 0,
                            "text": None,
                            "dropped": True,
                        },
                    )
                else:
                    self.pending.append(passage)

    def take_pending(self):
        with self.lock:
            if self.closed or self.decode_busy or not self.pending:
                return None
            self.decode_busy = True
            return self.pending.popleft()

    def take_interim(self):
        """추적 중인 트랙의 현재 후보 스냅샷을 판독용으로 꺼낸다. 퇴장한 통과가 우선이다."""
        with self.lock:
            track = self.tracker.active
            if (
                self.closed
                or self.decode_busy
                or track is None
                or track.resolved
                or track.interim_runs >= INTERIM_MAX_RUNS
                or track.appended - track.interim_at < INTERIM_MIN_NEW_FRAMES
                or len(track.frames) < 2
            ):
                return None
            self.decode_busy = True
            track.interim_runs += 1
            track.interim_at = track.appended
            return track, replace(track, frames=list(track.frames))

    def release_interim(self, track, *, retry=False):
        with self.lock:
            self.decode_busy = False
            if retry:
                # 슬롯이 바빠 시작하지 못했다. 시도 횟수를 쓰지 않고 다음 기회에 다시 한다.
                track.interim_runs -= 1
                track.interim_at = track.appended - INTERIM_MIN_NEW_FRAMES

    def complete_interim(self, track, response):
        """확정(confirmed)일 때만 결과를 낸다. 미확정은 퇴장 뒤 최종 판독에 맡긴다."""
        with self.lock:
            self.decode_busy = False
            if self.closed or response.get("status") != "confirmed":
                return
            track.resolved = True
            # 판독 중에 트랙이 끝나 이미 대기열에 들어갔다면 중복 판독하지 않는다.
            self.pending = deque(p for p in self.pending if p is not track)
            self.finished += 1
            self._store(track.track_id, response)

    def decode(self, passage):
        return process_batch(passage.batch(self.session_id))

    def complete(self, passage, response):
        with self.lock:
            self.decode_busy = False
            if not self.closed:
                self.finished += 1
                self._store(passage.track_id, response)

    def snapshot(self, *, after=0, include_results=True):
        with self.lock:
            track = self.tracker.active
            return {
                "active_track": track.track_id if track else None,
                "active_box": list(track.box.xywh) if track else None,
                "active_resolved": bool(track and track.resolved),
                "frame_count": self.received,
                "pending": len(self.pending),
                "decoding": self.decode_busy,
                "completed": self.finished,
                "dropped": self.dropped,
                "skipped_detections": self.skipped_detections,
                "results": [body for _, body in self.results if body["result_sequence"] > after]
                if include_results
                else [],
                "oldest_result_sequence": self.results[0][1]["result_sequence"]
                if self.results
                else self.result_sequence + 1,
            }

    def discard(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
            self.results.clear()
            self.result_bytes = 0
            self.tracker.active = None
