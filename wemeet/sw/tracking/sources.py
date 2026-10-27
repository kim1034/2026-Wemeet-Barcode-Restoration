"""Replaceable observation providers, independent of HTTP and camera APIs."""

from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

from wemeet.ai.detection import detect
from wemeet.sw.frame_batch import Frame
from wemeet.sw.tracking.core import Box


class BoxSource(Protocol):
    def observe(self, frame: Frame) -> list[Box] | None: ...


class BarcodeBoxes:
    def __init__(self, *, interval_ms=2000, execute: Callable = lambda fn: fn()):
        self.interval_ms = interval_ms
        self.execute = execute
        self.last_attempt = -float("inf")
        self.calls = 0

    def observe(self, frame):
        if frame.offset_ms - self.last_attempt < self.interval_ms:
            return None
        self.last_attempt = frame.offset_ms
        self.calls += 1
        found = self.execute(lambda: detect(frame.image_bgr))
        if found is None:
            return []
        if found.bbox_xywh_original is None:
            raise ValueError("Detector did not supply original coordinates")
        return [Box(found.bbox_xywh_original)]


class MotionBoxes:
    """Fallback only: a foreground object is NOT a detected barcode."""

    def __init__(self):
        self.background = None

    def observe(self, frame):
        image = frame.image_bgr
        gray = cv2.cvtColor(cv2.resize(image, (320, 180)), cv2.COLOR_BGR2GRAY)
        if self.background is None:
            self.background = gray.copy()
            return []
        mask = (cv2.absdiff(gray, self.background) > 25).astype(np.uint8)
        if mask.mean() > 0.7:  # Global exposure change, not an object observation.
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        components = [s for s in stats[1:count] if s[cv2.CC_STAT_AREA] >= 100]
        if not components:
            return []
        x, y, w, h, _ = max(components, key=lambda s: s[cv2.CC_STAT_AREA])
        sy, sx = image.shape[0] / 180, image.shape[1] / 320
        return [Box((round(x * sx), round(y * sy), round(w * sx), round(h * sy)), "motion")]
