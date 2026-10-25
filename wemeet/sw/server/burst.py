"""Bounded JPEG multipart transport; selection itself has no HTTP dependencies."""

import hashlib
import io
import json
import time
from typing import Literal

import cv2
import numpy as np
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from wemeet.sw.frame_batch import Frame, FrameBatch, process_batch
from wemeet.sw.server.runtime import RuntimeUnavailable

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 1920 * 1080


class InputError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code, self.message = status_code, message


class FrameInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    frame_id: str = Field(min_length=1, max_length=80, pattern=r"^[\w-]+$")
    offset_ms: float = Field(ge=0, lt=1000)
    width: int = Field(strict=True, ge=1, le=1920)
    height: int = Field(strict=True, ge=1, le=1920)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    session_id: str = Field(min_length=1, max_length=80, pattern=r"^[\w-]+$")
    batch_id: str = Field(min_length=1, max_length=80, pattern=r"^[\w-]+$")
    window_ms: Literal[1000] = 1000
    source_id: Literal["browser"] = "browser"
    jpeg_quality: float = Field(default=0.9, ge=0, le=1)
    trigger: Literal["activity", "continuous"] = "activity"
    frames: list[FrameInfo] = Field(min_length=1, max_length=8)


async def read_limited_body(request: Request, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError as exc:
        raise InputError(400, "잘못된 Content-Length입니다.") from exc
    if length < 0:
        raise InputError(400, "잘못된 Content-Length입니다.")
    if length > limit:
        raise InputError(413, "업로드 전체 용량 제한을 초과했습니다.")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise InputError(413, "업로드 전체 용량 제한을 초과했습니다.")
        data.extend(chunk)
    return bytes(data)


def inspect_image(data: bytes, *, jpeg_only=False, max_pixels=MAX_PIXELS) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if jpeg_only and image.format != "JPEG":
                raise InputError(400, "묶음의 프레임은 JPEG여야 합니다.")
            if width * height > max_pixels:
                raise InputError(413, "이미지 픽셀 수 제한을 초과했습니다.")
            image.verify()
            return width, height
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InputError(400, "이미지를 읽지 못했습니다.") from exc


async def read_batch(request: Request) -> tuple[Manifest, list[bytes], str]:
    data = await read_limited_body(request)

    async def receive():
        return {"type": "http.request", "body": data, "more_body": False}

    bounded = Request(request.scope, receive)
    try:
        async with bounded.form(max_files=8, max_fields=1, max_part_size=64 * 1024) as form:
            if set(form) != {"manifest", "frames"}:
                raise InputError(400, "manifest와 frames가 필요합니다.")
            raw_manifest = form["manifest"]
            if not isinstance(raw_manifest, str):
                raise InputError(400, "manifest는 JSON 문자열이어야 합니다.")
            manifest = Manifest.model_validate_json(raw_manifest)
            files = form.getlist("frames")
            if len(files) != len(manifest.frames):
                raise InputError(400, "프레임 수와 메타데이터 수가 다릅니다.")
            ids = [frame.frame_id for frame in manifest.frames]
            times = [frame.offset_ms for frame in manifest.frames]
            if len(set(ids)) != len(ids) or any(b <= a for a, b in zip(times, times[1:])):
                raise InputError(400, "프레임 ID는 고유하고 촬영 시각은 증가해야 합니다.")
            encoded = []
            sizes = set()
            for info, file in zip(manifest.frames, files):
                if not isinstance(file, UploadFile) or file.content_type != "image/jpeg":
                    raise InputError(400, "frames에는 JPEG 파일만 넣어 주세요.")
                content = await file.read()
                size = inspect_image(content, jpeg_only=True)
                if size != (info.width, info.height):
                    raise InputError(400, "실제 해상도와 메타데이터가 다릅니다.")
                sizes.add(size)
                encoded.append(content)
            if len(sizes) != 1:
                raise InputError(400, "한 묶음의 프레임 해상도는 같아야 합니다.")
    except (ValidationError, HTTPException) as exc:
        raise InputError(400, "올바른 JPEG 묶음과 메타데이터를 보내 주세요.") from exc
    digest = hashlib.sha256(json.dumps(manifest.model_dump(), sort_keys=True).encode())
    for content in encoded:
        digest.update(hashlib.sha256(content).digest())
    return manifest, encoded, digest.hexdigest()


async def decode_burst(request: Request, runtime, to_response, data_uri) -> JSONResponse:
    received = time.perf_counter()
    try:
        manifest, encoded, fingerprint = await read_batch(request)

        def work():
            frames = []
            for info, content in zip(manifest.frames, encoded):
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (info.height, info.width):
                    raise InputError(400, "JPEG 프레임을 해제하지 못했습니다.")
                frames.append(Frame(info.frame_id, info.offset_ms, image))
            batch = FrameBatch(manifest.session_id, manifest.batch_id, frames)
            result = process_batch(batch)
            representative = to_response(result.representative) if result.representative else None
            return {
                "ok": result.status == "confirmed",
                "status": result.status,
                "text": result.text,
                "completion_reason": result.completion_reason,
                "session_id": batch.session_id,
                "batch_id": batch.batch_id,
                "window_ms": batch.window_ms,
                "frame_count": len(frames),
                "unique_frame_count": result.unique_frame_count,
                "frame_attempt_count": len(result.attempts),
                "attempts": result.attempts,
                "matching_frame_ids": result.matching_frame_ids,
                "primary_frame_id": result.primary_frame.frame_id if result.primary_frame else None,
                "representative_frame_id": result.representative_frame_id,
                "primary_image": (
                    data_uri(result.primary_frame.image_bgr, ".jpg")
                    if result.primary_frame
                    else None
                ),
                "result": representative,
                "timing_ms": {
                    "selection": result.selection_ms,
                    "pipeline": result.pipeline_ms,
                    "processing": result.processing_ms,
                    "server": (time.perf_counter() - received) * 1000,
                },
            }

        body = await run_in_threadpool(
            runtime.execute,
            work,
            key=(manifest.session_id, manifest.batch_id),
            fingerprint=fingerprint,
        )
        return JSONResponse(body)
    except InputError as exc:
        return JSONResponse({"ok": False, "error": exc.message}, status_code=exc.status_code)
    except RuntimeUnavailable as exc:
        return JSONResponse(
            {"ok": False, "error": exc.reason},
            status_code=exc.status_code,
            headers={"Retry-After": "2"},
        )
