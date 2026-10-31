"""작업자 화면(web/index.html)과 판독 API. 실행: uv run uvicorn wemeet.sw.server:app

판독 실패도 HTTP 200 이다 (docs/parts/sw.md 「server.py」). 이미지와 바코드 번호는
디스크와 로그에 남기지 않는다 — 실제 송장에는 수취인 정보가 있다.
"""

# ruff: noqa: E402
# 스레드 제한은 numpy·torch 를 불러오기 전에 해야 해서 다른 import 보다 앞선다 (threads.py).
from wemeet.sw.server import threads

threads.limit_cpu_threads()

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from wemeet.ai.detection.yolo_obb import DEFAULT_HF_FILENAME as DETECTION_WEIGHTS
from wemeet.ai.geometry import DEFAULT_HF_FILENAME as GEOMETRY_WEIGHTS
from wemeet.schemas import PipelineResult
from wemeet.sw.pipeline import run
from wemeet.sw.server.burst import InputError, decode_burst, inspect_image, read_limited_body
from wemeet.sw.server.runtime import InferenceRuntime, RuntimeUnavailable
from wemeet.sw.server.stream import StreamController
from wemeet.sw.server.tracking import TrackController

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
PREVIEW_LONG_SIDE = 1280
WEB_INDEX = Path(__file__).resolve().parents[3] / "web" / "index.html"

log = logging.getLogger(__name__)
runtime = InferenceRuntime()


@asynccontextmanager
async def lifespan(app):
    runtime.preparing = True
    prepare = runtime.prepare

    def prepare_models():
        # torch 를 처음 불러오는 곳이라 이벤트 루프를 막지 않게 작업 스레드에서 한다.
        threads.apply_library_threads()
        prepare()

    preparation = asyncio.create_task(run_in_threadpool(prepare_models))
    tracker = tracking_controller()
    worker = asyncio.create_task(tracker.worker())
    yield
    worker.cancel()
    with suppress(asyncio.CancelledError):
        await worker
    await preparation


app = FastAPI(title="노리드존 판독기", lifespan=lifespan)
tracking = None
# 이름이 같은 하위 모듈(wemeet.sw.server.stream)을 가리지 않도록 _controller 를 붙인다.
stream_controller_instance = None


def tracking_controller():
    global tracking
    if tracking is None or tracking.runtime is not runtime:
        tracking = TrackController(runtime, to_response, _data_uri)
    return tracking


def stream_controller():
    global stream_controller_instance
    current = stream_controller_instance
    if current is None or current.runtime is not runtime:
        current = stream_controller_instance = StreamController(runtime, to_response, _data_uri)
    return current


def _data_uri(image: np.ndarray, fmt: str = ".png") -> str:
    h, w = image.shape[:2]
    scale = PREVIEW_LONG_SIDE / max(h, w)
    if scale < 1:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, 85] if fmt == ".jpg" else []
    _, buf = cv2.imencode(fmt, image, params)
    mime = "jpeg" if fmt == ".jpg" else "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


def _points(points: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(points, dtype=np.float64), 4).tolist()


def _grid_shape(dst: np.ndarray) -> list[int] | None:
    n_x = len(np.unique(np.round(dst[:, 0], 6)))
    n_y = len(np.unique(np.round(dst[:, 1], 6)))
    return [n_x, n_y] if n_x * n_y == len(dst) else None


def to_response(result: PipelineResult, image_fmt: str = ".png") -> dict:
    """PipelineResult 를 화면이 쓰는 JSON 으로 바꾼다. 형식은 web/index.html 이 읽는다.

    `image_fmt` 는 크롭·보정 이미지 형식이다. 사진 한 장 판독은 무손실 PNG 그대로 두고,
    매 프레임 판독(stream.py)은 응답이 작아야 다음 프레임이 밀리지 않아 JPEG 를 쓴다.
    """
    decode = result.decode
    detected = result.detected
    rectified = result.rectified
    field = rectified.field if rectified is not None else None
    oh, ow = result.original_bgr_uint8.shape[:2]

    body = {
        "ok": result.ok,
        "text": decode.text,
        "symbology": None,
        "failure_reason": decode.failure_reason,
        "retry_count": decode.retry_count,
        "degraded": decode.degraded,
        "stage_ms": {key: round(ms, 1) for key, ms in decode.stage_ms.items()},
        "total_ms": round(decode.total_ms, 1),
        "original_size": [ow, oh],
        "detection": None,
        "rectify": None,
        "images": {
            "original": _data_uri(result.original_bgr_uint8, ".jpg"),
            # 기존 단일 사진 응답 형식 유지. 원본 박스는 추적 어댑터에서 사용한다.
            "detected_box": None,
            "crop": None,
            "rectified": None,
        },
    }

    if detected is not None:
        ch, cw = detected.crop_bgr_uint8.shape[:2]
        body["detection"] = {
            "confidence": round(detected.confidence, 3),
            "angle_deg_ccw": round(detected.angle_deg_ccw, 1),
            "crop_size": [cw, ch],
        }
        body["images"]["crop"] = _data_uri(detected.crop_bgr_uint8, image_fmt)

        if field is not None:
            rh, rw = rectified.image_gray_uint8.shape
            body["rectify"] = {
                "method": field.method,
                "confidence": round(field.confidence, 3),
                "out_scale": round(rw / cw, 2),
                "size": [rw, rh],
                "grid": _grid_shape(np.asarray(field.control_points_dst_norm)),
                "control_points_src_norm": _points(field.control_points_src_norm),
                "control_points_dst_norm": _points(field.control_points_dst_norm),
            }
            body["images"]["rectified"] = _data_uri(rectified.image_gray_uint8, image_fmt)

    return body


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_INDEX, media_type="text/html; charset=utf-8")


@app.get("/assets/{filename}", include_in_schema=False)
def javascript(filename: str):
    if filename not in {
        "capture.mjs",
        "activity.mjs",
        "continuous.mjs",
        "result-state.mjs",
        "tracking.mjs",
        "stream.mjs",
        "camera.mjs",
    }:
        return _error(404, "파일을 찾지 못했습니다.")
    # Avoid Windows MIME registry overrides and stale cached modules during experiments.
    return FileResponse(
        WEB_INDEX.parent / filename,
        media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": runtime.ready,
        "ready": runtime.ready,
        "preparing": runtime.preparing,
        "busy": runtime.busy,
        "error": runtime.error,
        "models": {"detection": DETECTION_WEIGHTS, "geometry": GEOMETRY_WEIGHTS},
        "cpu_threads": threads.cpu_threads,  # None = 손대지 않음 (할당량 없음)
    }


@app.post("/api/decode-burst")
async def decode_frames(request: Request) -> JSONResponse:
    return await decode_burst(request, runtime, to_response, _data_uri)


@app.post("/api/stream-frame")
async def stream_frame(request: Request):
    return await stream_controller().ingest(request)


@app.delete("/api/stream/{session_id}")
def stream_close(session_id: str):
    # async 로 두면 안 된다: close() 는 판독 중인 프레임이 세션 잠금을 놓을 때까지 기다리는데,
    # 그 잠금은 이벤트 루프에서 풀린다. 루프에서 기다리면 서로를 기다려 서버 전체가 멈춘다.
    # def 로 두면 FastAPI 가 작업 스레드에서 실행해 루프는 계속 돈다.
    return {"ok": True, "events": stream_controller().close(session_id)}


@app.post("/api/track-frames")
async def track_frames(request: Request):
    return await tracking_controller().ingest(request)


@app.get("/api/tracking/{session_id}")
async def tracking_status(session_id: str, after: int = 0):
    controller = tracking_controller()
    controller.expire()
    session = controller.sessions.get(session_id)
    return session.snapshot(after=after) if session else {"expired": True, "results": []}


@app.delete("/api/tracking/{session_id}")
async def tracking_close(session_id: str):
    tracking_controller().close(session_id)
    return {"ok": True}


@app.post("/api/decode")
async def decode_image(request: Request) -> JSONResponse:
    try:
        data = await read_limited_body(request, MAX_UPLOAD_BYTES)
        inspect_image(data, max_pixels=16_000_000)

        def work():
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise InputError(400, "이미지를 읽지 못했습니다. JPG나 PNG 파일을 보내 주세요.")
            started = time.perf_counter()
            result = run(image)
            body = to_response(result)
            log.info("decode ok=%s wall_ms=%.1f", result.ok, (time.perf_counter() - started) * 1000)
            return body

        return JSONResponse(await run_in_threadpool(runtime.execute, work))
    except InputError as exc:
        return _error(exc.status_code, exc.message)
    except RuntimeUnavailable as exc:
        return JSONResponse(
            {"ok": False, "error": exc.reason},
            status_code=exc.status_code,
            headers={"Retry-After": "2"},
        )
