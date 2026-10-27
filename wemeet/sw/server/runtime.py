"""One inference slot per process, also held after an HTTP client disconnects."""

import json
import threading
import time
from collections import OrderedDict

import numpy as np


class RuntimeUnavailable(Exception):
    def __init__(self, status_code: int, reason: str):
        self.status_code = status_code
        self.reason = reason


class InferenceRuntime:
    def __init__(self, *, ready: bool = False):
        self.ready = ready
        self.preparing = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._active: tuple | None = None
        self._cache: OrderedDict = OrderedDict()
        self.cache_ttl_s = 60
        self.cache_limit_bytes = 16 * 1024 * 1024

    def prepare(self):
        self.preparing = True
        try:
            from wemeet.ai.detection import detect
            from wemeet.ai.detection.yolo_obb import load_model as load_detection
            from wemeet.ai.geometry import estimate_geometry, load_model
            from wemeet.schemas import DetectedBarcode

            load_detection()
            load_model()
            detect(np.full((1080, 1920, 3), 127, np.uint8))
            estimate_geometry(DetectedBarcode(np.full((160, 384, 3), 127, np.uint8), 0, 1))
            self.ready = True
            self.error = None
        except Exception:
            self.ready = False
            self.error = "model_prepare_failed"
        finally:
            self.preparing = False

    def _prune(self):
        now = time.monotonic()
        for key, (_, expires, _, _) in list(self._cache.items()):
            if expires <= now:
                del self._cache[key]
        while self._cache and (
            len(self._cache) > 4
            or sum(item[3] for item in self._cache.values()) > self.cache_limit_bytes
        ):
            self._cache.popitem(last=False)

    def execute(self, function, *, key=None, fingerprint=None):
        with self._lock:
            if not self.ready:
                raise RuntimeUnavailable(503, self.error or "models_preparing")
            self._prune()
            if key is not None and key in self._cache:
                old_fingerprint, _, body, _ = self._cache[key]
                if old_fingerprint != fingerprint:
                    raise RuntimeUnavailable(409, "batch_id_reused")
                return body
            if self._active is not None:
                active_key, active_fingerprint = self._active
                if key is not None and active_key == key and active_fingerprint != fingerprint:
                    raise RuntimeUnavailable(409, "batch_id_reused")
                raise RuntimeUnavailable(429, "inference_busy")
            self._active = (key, fingerprint)
        try:
            body = function()
            if key is not None:
                size = len(json.dumps(body).encode("utf-8"))
                with self._lock:
                    self._cache[key] = (
                        fingerprint,
                        time.monotonic() + self.cache_ttl_s,
                        body,
                        size,
                    )
                    self._prune()
            return body
        finally:
            with self._lock:
                self._active = None

    @property
    def busy(self):
        with self._lock:
            return self._active is not None
