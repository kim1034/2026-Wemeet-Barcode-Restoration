"""Bounded session transport and independent decode scheduling for passages."""

import asyncio
import os
import time
from typing import Literal

import cv2
import numpy as np
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import Field

from wemeet.sw.frame_batch import Frame
from wemeet.sw.server.burst import FrameInfo, InputError, Manifest, read_batch
from wemeet.sw.server.runtime import RuntimeUnavailable
from wemeet.sw.tracking import BarcodeBoxes, MotionBoxes, Tracker
from wemeet.sw.tracking.session import TrackingSession

DETECT_INTERVAL_ENV = "WEMEET_TRACK_DETECT_INTERVAL_MS"
DEFAULT_DETECT_INTERVAL_MS = 500
# 프레임은 125ms(8fps) 간격이라 그보다 촘촘한 값은 의미가 없고, 너무 길면 트랙을 못 잇는다.
MIN_DETECT_INTERVAL_MS = 100
MAX_DETECT_INTERVAL_MS = 10000


def detect_interval_ms(value=None):
    """바코드 재검출 간격(ms). 값이 없으면 환경 변수, 그것도 없으면 기본값을 쓴다."""
    if value is None:
        value = os.environ.get(DETECT_INTERVAL_ENV)
    if value is None or value == "":
        return DEFAULT_DETECT_INTERVAL_MS
    try:
        interval = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{DETECT_INTERVAL_ENV}는 정수(ms)여야 합니다: {value!r}") from None
    if not MIN_DETECT_INTERVAL_MS <= interval <= MAX_DETECT_INTERVAL_MS:
        raise ValueError(
            f"{DETECT_INTERVAL_ENV}는 {MIN_DETECT_INTERVAL_MS}~{MAX_DETECT_INTERVAL_MS}ms "
            f"사이여야 합니다: {interval}"
        )
    return interval


class TrackFrameInfo(FrameInfo):
    offset_ms: float = Field(ge=0, lt=500)


class TrackManifest(Manifest):
    window_ms: Literal[500] = 500
    sequence: int = Field(strict=True, ge=0)
    captured_at_ms: float = Field(ge=0, le=1e12)
    box_source: Literal["barcode", "motion"] = "barcode"
    frames: list[TrackFrameInfo] = Field(min_length=1, max_length=4)


class TrackController:
    def __init__(self, runtime, to_response, data_uri, detect_interval=None):
        self.detect_interval_ms = detect_interval_ms(detect_interval)
        self.runtime = runtime
        self.to_response = to_response
        self.data_uri = data_uri
        self.sessions = {}
        self.sources = {}
        self.closed_sessions = {}

    def expire(self):
        now = time.monotonic()
        self.closed_sessions = {
            key: until for key, until in self.closed_sessions.items() if until > now
        }
        for key, session in list(self.sessions.items()):
            if now - session.last_activity > 30:
                self.close(key)

    def close(self, session_id):
        self.closed_sessions[session_id] = time.monotonic() + 60
        while len(self.closed_sessions) > 64:
            self.closed_sessions.pop(next(iter(self.closed_sessions)))
        session = self.sessions.pop(session_id, None)
        self.sources.pop(session_id, None)
        if session:
            session.discard()

    def session(self, key, source):
        self.expire()
        if key in self.closed_sessions:
            raise InputError(409, "종료된 추적 세션입니다. 자동 판독을 다시 시작하세요.")
        if key in self.sessions:
            if self.sources[key] != source:
                raise InputError(409, "세션 중에는 박스 공급원을 바꿀 수 없습니다.")
            return self.sessions[key]
        if len(self.sessions) >= 2:
            raise InputError(429, "추적 세션이 가득 찼습니다. 기존 세션을 중지하세요.")
        session = TrackingSession(
            key,
            MotionBoxes(),
            tracker=Tracker(
                max_misses=3 if source == "motion" else 2,
            ),
        )
        if source == "barcode":

            def execute(fn):
                return self.runtime.execute(fn)

            session.source = BarcodeBoxes(interval_ms=self.detect_interval_ms, execute=execute)
        self.sessions[key] = session
        self.sources[key] = source
        return session

    async def ingest(self, request):
        try:
            if not self.runtime.ready:
                raise RuntimeUnavailable(503, self.runtime.error or "models_preparing")
            manifest, encoded, fingerprint = await read_batch(request, TrackManifest)
            session = self.session(manifest.session_id, manifest.box_source)
            if not session.ingest_lock.acquire(blocking=False):
                raise InputError(429, "추적 프레임 처리 중입니다.")
            try:
                if manifest.sequence < session.last_sequence:
                    raise InputError(409, "지난 추적 묶음입니다.")
                if manifest.sequence == session.last_sequence:
                    if fingerprint != session.last_fingerprint:
                        raise InputError(409, "같은 추적 ID의 내용이 다릅니다.")
                else:

                    def work():
                        frames = []
                        for info, content in zip(manifest.frames, encoded):
                            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                            if image is None:
                                raise InputError(400, "JPEG를 읽지 못했습니다.")
                            frames.append(
                                Frame(
                                    info.frame_id, manifest.captured_at_ms + info.offset_ms, image
                                )
                            )
                        last = session.tracker.last_ms
                        if last is not None and frames[0].offset_ms <= last:
                            raise InputError(409, "촬영 시각이 이전 묶음보다 앞섭니다.")
                        for frame in frames:
                            try:
                                session.ingest([frame])
                            except RuntimeUnavailable:
                                # Inference is occupied by decode; still advance timestamps
                                # and prediction, without inventing a negative detection.
                                session.skipped_detections += 1
                                with session.lock:
                                    if session.closed:
                                        return
                                    session.received += 1
                                    ended = session.tracker.update(frame, None)
                                    session.enqueue(ended)

                    await run_in_threadpool(work)
                    session.last_sequence = manifest.sequence
                    session.last_fingerprint = fingerprint
                session.last_activity = time.monotonic()
                return JSONResponse(
                    {
                        "session_id": manifest.session_id,
                        "sequence": manifest.sequence,
                        **session.snapshot(include_results=False),
                    }
                )
            finally:
                session.ingest_lock.release()
        except (InputError, RuntimeUnavailable) as exc:
            return JSONResponse(
                {"error": getattr(exc, "message", None) or exc.reason}, status_code=exc.status_code
            )

    def response(self, session, passage, outcome, early=False):
        batch = passage.batch(session.session_id)
        return {
            "ok": outcome.status == "confirmed",
            # 통과 중에 확정한 결과다. 이때 track_end_* 는 아직 없다(null).
            "early": early,
            "status": outcome.status,
            "text": outcome.text,
            "completion_reason": outcome.completion_reason,
            "session_id": session.session_id,
            "batch_id": passage.track_id,
            "track_id": passage.track_id,
            "track_end_reason": passage.reason,
            "track_start_ms": passage.start_ms,
            "track_end_ms": passage.end_ms,
            "box_source": passage.box.source,
            "window_ms": batch.window_ms,
            "frame_count": len(batch.frames),
            "unique_frame_count": outcome.unique_frame_count,
            "frame_attempt_count": len(outcome.attempts),
            "attempts": outcome.attempts,
            "matching_frame_ids": outcome.matching_frame_ids,
            "primary_frame_id": outcome.primary_frame.frame_id if outcome.primary_frame else None,
            "representative_frame_id": outcome.representative_frame_id,
            "primary_image": self.data_uri(outcome.primary_frame.image_bgr, ".jpg")
            if outcome.primary_frame
            else None,
            "result": self.to_response(outcome.representative) if outcome.representative else None,
            "timing_ms": {
                "selection": outcome.selection_ms,
                "pipeline": outcome.pipeline_ms,
                "processing": outcome.processing_ms,
                "server": outcome.processing_ms,
            },
        }

    async def work_once(self):
        self.expire()
        for session in list(self.sessions.values()):
            passage = session.take_pending()
            if passage is None:
                if await self.interim_once(session):
                    return
                continue
            try:

                def work():
                    outcome = session.decode(passage)
                    return self.response(session, passage, outcome)

                response = await run_in_threadpool(self.runtime.execute, work)
            except RuntimeUnavailable:
                with session.lock:
                    if not session.closed:
                        session.pending.appendleft(passage)
                    session.decode_busy = False
                return
            except Exception:
                # Do not log payloads, exception messages containing values, or images.
                response = {
                    "track_id": passage.track_id,
                    "status": "unread",
                    "text": None,
                    "completion_reason": "processing_error",
                    "frame_attempt_count": 0,
                    "dropped": True,
                }
            session.complete(passage, response)
            return

    async def interim_once(self, session):
        """아직 화면에 있는 트랙을 판독해 확정되면 바로 결과를 낸다. 시도했으면 True."""
        taken = session.take_interim()
        if taken is None:
            return False
        track, snapshot = taken
        try:

            def work():
                outcome = session.decode(snapshot)
                return self.response(session, snapshot, outcome, early=True)

            response = await run_in_threadpool(self.runtime.execute, work)
        except RuntimeUnavailable:
            session.release_interim(track, retry=True)
            return False
        except Exception:
            # 이 시도만 버린다. 트랙은 퇴장 뒤 최종 판독에서 다시 판독한다.
            session.release_interim(track)
            return True
        session.complete_interim(track, response)
        return True

    async def worker(self):
        try:
            while True:
                await self.work_once()
                await asyncio.sleep(0.05)
        finally:
            for key in list(self.sessions):
                self.close(key)
