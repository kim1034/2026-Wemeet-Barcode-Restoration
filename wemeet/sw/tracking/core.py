"""One active passage, time-based association, bounded in-memory candidates."""

import math
from dataclasses import dataclass, field

from wemeet.sw.frame_batch import Frame, FrameBatch


@dataclass(frozen=True)
class Box:
    xywh: tuple[int, int, int, int]
    source: str = "barcode"

    @property
    def center(self):
        x, y, w, h = self.xywh
        return x + w / 2, y + h / 2


@dataclass
class Passage:
    track_id: str
    start_ms: float
    last_seen_ms: float
    box: Box
    velocity: tuple[float, float] = (0.0, 0.0)
    misses: int = 0
    observations: int = 1
    frames: list[Frame] = field(default_factory=list)
    end_ms: float | None = None
    reason: str | None = None

    def batch(self, session_id: str) -> FrameBatch:
        return FrameBatch(
            session_id,
            self.track_id,
            [
                Frame(f.frame_id, f.offset_ms - self.start_ms, f.image_bgr, f.bbox_xywh)
                for f in self.frames
            ],
            max(1.0, (self.end_ms or self.last_seen_ms) - self.start_ms),
            use_roi=self.box.source == "barcode",
        )


class Tracker:
    """Boxes are a list so a future multi-track associator can reuse the interface.

    None means detection was skipped; [] means an actual negative observation.
    A completed passage is removed BEFORE being emitted: it cannot emit twice.
    """

    def __init__(
        self, *, max_candidates=8, max_misses=2, lost_ms=4500, input_gap_ms=2500, max_track_ms=30000
    ):
        self.max_candidates = max_candidates
        self.max_misses = max_misses
        self.lost_ms = lost_ms
        self.input_gap_ms = input_gap_ms
        self.max_track_ms = max_track_ms
        self.active: Passage | None = None
        self.sequence = 0
        self.last_ms: float | None = None
        self.shape = None
        self.holding = False
        self.hold_misses = 0

    def close(self, now: float, reason: str) -> list[Passage]:
        passage, self.active = self.active, None
        if passage is None:
            return []
        passage.end_ms = now
        passage.reason = reason
        return [passage]

    def _predicted(self, now):
        track = self.active
        dt = now - track.last_seen_ms
        x, y, w, h = track.box.xywh
        return Box(
            (round(x + track.velocity[0] * dt), round(y + track.velocity[1] * dt), w, h),
            track.box.source,
        )

    def _matches(self, box, now):
        predicted = self._predicted(now)
        distance = math.dist(box.center, predicted.center)
        _, _, w, h = predicted.xywh
        _, _, nw, nh = box.xywh
        area_ratio = nw * nh / (w * h)
        # Permit hand speed changes, but reject a new box at the entry edge after exit.
        gate = max(60.0, math.hypot(w, h) * 1.5)
        if self.active.observations == 1:
            # No measured velocity yet; sparse first/second detections can be far apart.
            gate = max(gate, self.shape[1] * 0.75)
        return distance <= gate and 0.2 <= area_ratio <= 5.0

    def update(self, frame: Frame, boxes: list[Box] | None) -> list[Passage]:
        now = frame.offset_ms
        if not math.isfinite(now) or (self.last_ms is not None and now <= self.last_ms):
            raise ValueError("Frame capture timestamps must strictly increase")
        h, w = frame.image_bgr.shape[:2]
        completed = []
        if self.last_ms is not None and now - self.last_ms > self.input_gap_ms:
            completed += self.close(self.last_ms, "input_gap")
        if self.shape is not None and self.shape != (h, w):
            completed += self.close(self.last_ms, "resolution_changed")
        self.shape = h, w
        self.last_ms = now
        if self.holding:
            self.hold_misses = (
                self.hold_misses + 1 if boxes == [] else 0 if boxes else self.hold_misses
            )
            if self.hold_misses >= self.max_misses:
                self.holding = False
            return completed
        if boxes:
            # The current detector supplies one highest-confidence box.
            valid = [box for box in boxes if box.xywh[2] > 0 and box.xywh[3] > 0]
            matches = [box for box in valid if self.active and self._matches(box, now)]
            selected = matches[0] if matches else (valid[0] if valid else None)
            if selected and self.active and not matches:
                completed += self.close(now, "association_break")
            if selected and self.active is None:
                self.sequence += 1
                self.active = Passage(f"t-{self.sequence}", now, now, selected)
            elif selected:
                track = self.active
                dt = now - track.last_seen_ms
                old, new = track.box.center, selected.center
                if selected.source == "barcode":
                    # Correct buffered predictions once both endpoint observations
                    # are available, especially before the first velocity estimate.
                    corrected = []
                    for candidate in track.frames:
                        if candidate.offset_ms >= track.last_seen_ms:
                            fraction = (candidate.offset_ms - track.last_seen_ms) / dt
                            roi = tuple(
                                round(a + fraction * (b - a))
                                for a, b in zip(track.box.xywh, selected.xywh)
                            )
                            x, y, bw, bh = roi
                            x0, y0 = max(0, x), max(0, y)
                            x1, y1 = min(w, x + bw), min(h, y + bh)
                            roi = (x0, y0, x1 - x0, y1 - y0)
                            candidate = Frame(
                                candidate.frame_id, candidate.offset_ms, candidate.image_bgr, roi
                            )
                        corrected.append(candidate)
                    track.frames = corrected
                track.velocity = ((new[0] - old[0]) / dt, (new[1] - old[1]) / dt)
                track.last_seen_ms, track.box = now, selected
                track.misses = 0
                track.observations += 1
        elif boxes == [] and self.active:
            self.active.misses += 1
        if self.active:
            track = self.active
            predicted = self._predicted(now)
            x, y, bw, bh = predicted.xywh
            if track.misses >= self.max_misses:
                completed += self.close(now, "missing")
            elif now - track.last_seen_ms >= self.lost_ms:
                completed += self.close(now, "observation_timeout")
            elif x + bw <= 0 or y + bh <= 0 or x >= w or y >= h:
                completed += self.close(now, "left_view")
            elif now - track.start_ms >= self.max_track_ms:
                completed += self.close(now, "duration_limit")
                self.holding = True
            elif boxes != []:
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(w, x + bw), min(h, y + bh)
                roi = (x0, y0, x1 - x0, y1 - y0) if track.box.source == "barcode" else None
                track.frames.append(Frame(frame.frame_id, now, frame.image_bgr, roi))
                if len(track.frames) > self.max_candidates:
                    # Keep first/last and thin the densest interior time interval.
                    remove = min(
                        range(1, len(track.frames) - 1),
                        key=lambda i: track.frames[i + 1].offset_ms - track.frames[i - 1].offset_ms,
                    )
                    track.frames.pop(remove)
        return completed
