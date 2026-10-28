"""판독 먼저 스트림: 프레임을 한 장씩 받아 바로 판독하고 결과를 그 응답에 담는다.

GPU 기준 설계다. 묶음 전송·주기 검출·결과 조회(폴링)가 없다. 브라우저는 응답을 받은 뒤에
그때의 최신 프레임을 보내므로, 서버가 느려지면 중간 프레임은 자연히 건너뛴다(최신 프레임만 처리).
이미지와 번호는 디스크와 로그에 남기지 않는다.
"""

import threading
import time
from typing import Literal

import cv2
import numpy as np
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import Field

from wemeet.sw.frame_batch import Frame
from wemeet.sw.pipeline import run
from wemeet.sw.server.burst import FrameInfo, InputError, Manifest, read_batch
from wemeet.sw.server.runtime import RuntimeUnavailable
from wemeet.sw.tracking.stream import StreamTracker

MAX_SESSIONS = 2
SESSION_IDLE_S = 30
CLOSED_MEMORY_S = 60


class StreamFrameInfo(FrameInfo):
    offset_ms: float = Field(ge=0, lt=1)


class StreamManifest(Manifest):
    window_ms: Literal[0] = 0
    trigger: Literal["stream"] = "stream"
    sequence: int = Field(strict=True, ge=0)
    captured_at_ms: float = Field(ge=0, le=1e12)
    frames: list[StreamFrameInfo] = Field(min_length=1, max_length=1)


class StreamSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tracker = StreamTracker()
        self.lock = threading.Lock()
        self.last_sequence = -1
        self.last_captured_ms: float | None = None
        self.last_activity = time.monotonic()
        self.frames = 0


class StreamController:
    def __init__(self, runtime, to_response, data_uri):
        self.runtime = runtime
        self.to_response = to_response
        self.data_uri = data_uri
        self.sessions: dict[str, StreamSession] = {}
        self.closed: dict[str, float] = {}

    def expire(self):
        now = time.monotonic()
        self.closed = {key: until for key, until in self.closed.items() if until > now}
        for key, session in list(self.sessions.items()):
            if now - session.last_activity > SESSION_IDLE_S:
                self.sessions.pop(key, None)
                self.closed[key] = now + CLOSED_MEMORY_S

    def session(self, session_id: str) -> StreamSession:
        self.expire()
        if session_id in self.closed:
            raise InputError(409, "종료된 스트림입니다. 자동 판독을 다시 시작하세요.")
        if session_id not in self.sessions:
            if len(self.sessions) >= MAX_SESSIONS:
                raise InputError(429, "스트림이 가득 찼습니다. 기존 화면을 중지하세요.")
            self.sessions[session_id] = StreamSession(session_id)
        return self.sessions[session_id]

    def close(self, session_id: str) -> list[dict]:
        session = self.sessions.pop(session_id, None)
        self.closed[session_id] = time.monotonic() + CLOSED_MEMORY_S
        if session is None:
            return []
        with session.lock:
            end = (session.last_captured_ms or 0.0) + 1.0
            return session.tracker.close(end)

    async def ingest(self, request) -> JSONResponse:
        received = time.perf_counter()
        try:
            if not self.runtime.ready:
                raise RuntimeUnavailable(503, self.runtime.error or "models_preparing")
            manifest, encoded, _ = await read_batch(request, StreamManifest)
            session = self.session(manifest.session_id)
            if not session.lock.acquire(blocking=False):
                raise InputError(429, "이전 프레임을 처리 중입니다.")
            try:
                if manifest.sequence <= session.last_sequence:
                    raise InputError(409, "지난 프레임입니다.")
                if (
                    session.last_captured_ms is not None
                    and manifest.captured_at_ms <= session.last_captured_ms
                ):
                    raise InputError(409, "촬영 시각이 이전 프레임보다 앞섭니다.")
                info = manifest.frames[0]

                def work():
                    image = cv2.imdecode(np.frombuffer(encoded[0], np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise InputError(400, "JPEG를 읽지 못했습니다.")
                    started = time.perf_counter()
                    result = run(image)
                    pipeline_ms = (time.perf_counter() - started) * 1000
                    frame = Frame(info.frame_id, manifest.captured_at_ms, image)
                    events = session.tracker.observe(frame, result)
                    for event in events:
                        if event["status"] == "confirmed":
                            # 확정은 지금 프레임에서 일어나므로 이 프레임의 판독 결과를 붙인다.
                            event["result"] = self.to_response(result)
                            event["primary_frame_id"] = info.frame_id
                            event["representative_frame_id"] = info.frame_id
                            event["primary_image"] = self.data_uri(image, ".jpg")
                        event["timing_ms"] = {"selection": 0.0, "pipeline": pipeline_ms}
                    detected = result.detected
                    return (
                        events,
                        pipeline_ms,
                        (
                            list(detected.bbox_xywh_original)
                            if detected is not None and detected.bbox_xywh_original
                            else None
                        ),
                    )

                events, pipeline_ms, box = await run_in_threadpool(self.runtime.execute, work)
                session.last_sequence = manifest.sequence
                session.last_captured_ms = manifest.captured_at_ms
                session.last_activity = time.monotonic()
                session.frames += 1
                server_ms = (time.perf_counter() - received) * 1000
                for event in events:
                    event["timing_ms"]["server"] = server_ms
                    event["timing_ms"]["processing"] = server_ms
                return JSONResponse(
                    {
                        "session_id": manifest.session_id,
                        "sequence": manifest.sequence,
                        "frame_id": info.frame_id,
                        "events": events,
                        "detected_box": box,
                        "frames": session.frames,
                        "timing_ms": {"pipeline": pipeline_ms, "server": server_ms},
                        **session.tracker.snapshot(),
                    }
                )
            finally:
                session.lock.release()
        except (InputError, RuntimeUnavailable) as exc:
            return JSONResponse(
                {"error": getattr(exc, "message", None) or exc.reason}, status_code=exc.status_code
            )
