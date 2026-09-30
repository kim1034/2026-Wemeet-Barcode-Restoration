"""작업자 화면(web/index.html)과 판독 API. 실행: uv run uvicorn wemeet.sw.server:app

판독 실패도 HTTP 200 이다 (docs/parts/sw.md 「server.py」). 이미지와 바코드 번호는
디스크와 로그에 남기지 않는다 — 실제 송장에는 수취인 정보가 있다.
"""

import base64
import logging
import time
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

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
PREVIEW_LONG_SIDE = 1280
WEB_INDEX = Path(__file__).resolve().parents[3] / "web" / "index.html"

log = logging.getLogger(__name__)
app = FastAPI(title="노리드존 판독기")


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


def to_response(result: PipelineResult) -> dict:
    """PipelineResult 를 화면이 쓰는 JSON 으로 바꾼다. 형식은 web/index.html 이 읽는다."""
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
            # 원본 좌표계의 박스는 schemas.DetectedBarcode 에 없어서 아직 보낼 수 없다
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
        body["images"]["crop"] = _data_uri(detected.crop_bgr_uint8)

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
            body["images"]["rectified"] = _data_uri(rectified.image_gray_uint8)

    return body


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_INDEX, media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "models": {"detection": DETECTION_WEIGHTS, "geometry": GEOMETRY_WEIGHTS}}


@app.post("/api/decode")
async def decode_image(request: Request) -> JSONResponse:
    too_large = f"이미지가 너무 큽니다. {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 이하로 보내 주세요."
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
        return _error(413, too_large)
    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        return _error(413, too_large)

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        return _error(400, "이미지를 읽지 못했습니다. JPG나 PNG 파일을 보내 주세요.")

    started = time.perf_counter()
    result = await run_in_threadpool(run, image)
    body = await run_in_threadpool(to_response, result)
    log.info(
        "decode ok=%s failure=%s retry=%d degraded=%s stage_ms=%s wall_ms=%.1f",
        result.ok,
        result.decode.failure_reason,
        result.decode.retry_count,
        result.decode.degraded,
        body["stage_ms"],
        (time.perf_counter() - started) * 1000.0,
    )
    return JSONResponse(body)
