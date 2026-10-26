import asyncio
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import wemeet.sw.server as server
from wemeet.schemas import DecodeResult, DetectedBarcode, PipelineResult
from wemeet.sw.frame_batch import service
from wemeet.sw.server.runtime import InferenceRuntime


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "runtime", InferenceRuntime(ready=True))
    monkeypatch.setattr(server, "tracking", None)
    monkeypatch.setattr(
        service, "run", lambda im: PipelineResult(im, None, None, DecodeResult("00123", 0))
    )
    return TestClient(server.app)


def payload(sequence, value=100):
    infos, files = [], []
    for i in range(4):
        image = np.full((180, 320, 3), value + i, np.uint8)
        _, encoded = cv2.imencode(".jpg", image)
        infos.append(
            {"frame_id": f"f-{sequence}-{i}", "offset_ms": i * 125, "width": 320, "height": 180}
        )
        files.append(("frames", ("frame.jpg", encoded.tobytes(), "image/jpeg")))
    return {
        "session_id": "private-session",
        "batch_id": f"c-{sequence}",
        "sequence": sequence,
        "captured_at_ms": sequence * 500,
        "window_ms": 500,
        "frames": infos,
    }, files


def post(client, manifest, files):
    return client.post("/api/track-frames", data={"manifest": json.dumps(manifest)}, files=files)


def test_track_across_chunks_emits_one_result_and_retry_does_not_add_votes(
    client, monkeypatch, caplog
):
    def detect(image):
        return DetectedBarcode(image, 0, 1, (10, 10, 150, 80)) if image.mean() < 150 else None

    monkeypatch.setattr("wemeet.sw.tracking.sources.detect", detect)
    for i in range(9):
        data = payload(i, 100 if i < 4 else 200)
        response = post(client, *data)
        assert response.status_code == 200
        assert post(client, *data).status_code == 200
    controller = server.tracking_controller()
    session = controller.sessions["private-session"]
    assert session.received == 36
    assert len(session.pending) == 1
    asyncio.run(controller.work_once())
    body = client.get("/api/tracking/private-session").json()
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["status"] == "confirmed"
    assert result["attempts"][0]["score_scope"] == "remaining_barcode_rois"
    assert client.get("/api/tracking/private-session?after=1").json()["results"] == []
    assert "00123" not in caplog.text and "data:image" not in caplog.text
    assert client.delete("/api/tracking/private-session").status_code == 200
    assert session.closed and not session.results and not session.pending
    assert post(client, *payload(10)).status_code == 409


def test_timestamp_source_and_identity_validation(client, monkeypatch):
    monkeypatch.setattr("wemeet.sw.tracking.sources.detect", lambda _: None)
    manifest, files = payload(0)
    assert post(client, manifest, files).status_code == 200
    manifest["box_source"] = "motion"
    assert post(client, manifest, files).status_code == 409
    manifest, files = payload(1)
    manifest["captured_at_ms"] = 0
    assert post(client, manifest, files).status_code == 409
    manifest["frames"][0]["offset_ms"] = 600
    assert post(client, manifest, files).status_code == 400
    manifest, files = payload(1)
    manifest["sequence"] = -1
    assert post(client, manifest, files).status_code == 400


def test_busy_decode_skips_observation_but_keeps_capture_and_expiry(client):
    server.runtime._active = (None, None)
    response = post(client, *payload(0))
    assert response.status_code == 200
    assert response.json()["skipped_detections"] == 1
    controller = server.tracking_controller()
    session = controller.sessions["private-session"]
    session.last_activity -= 31
    controller.expire()
    assert session.closed and not controller.sessions
    server.runtime._active = None


def test_tracking_off_does_not_replace_baseline_and_jpegs_never_spool(client, monkeypatch):
    import starlette.formparsers as parser
    from starlette.formparsers import SpooledTemporaryFile

    def spool(*args, **kwargs):
        assert kwargs["max_size"] > 15 * 1024**2
        return SpooledTemporaryFile(*args, **kwargs)

    monkeypatch.setattr(parser, "SpooledTemporaryFile", spool)
    monkeypatch.setattr("wemeet.sw.tracking.sources.detect", lambda _: None)
    assert post(client, *payload(0)).status_code == 200
    assert client.get("/assets/tracking.mjs").status_code == 200
    assert 'value="fixed"' in client.get("/").text
    assert client.post("/api/decode", content=b"bad").status_code == 400
