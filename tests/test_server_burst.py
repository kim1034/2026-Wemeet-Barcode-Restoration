import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import wemeet.sw.server as server
from wemeet.schemas import DecodeResult, PipelineResult
from wemeet.sw.frame_batch import service
from wemeet.sw.server import burst
from wemeet.sw.server.runtime import InferenceRuntime


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "runtime", InferenceRuntime(ready=True))
    monkeypatch.setattr(
        service, "run", lambda image: PipelineResult(image, None, None, DecodeResult("00123", 0))
    )
    return TestClient(server.app)


def payload(count=8):
    frames, files = [], []
    for i in range(count):
        image = np.full((24, 32, 3), i * 10 + 20, np.uint8)
        _, encoded = cv2.imencode(".jpg", image)
        frames.append({"frame_id": f"f-{i}", "offset_ms": i * 100, "width": 32, "height": 24})
        files.append(("frames", (f"f-{i}.jpg", encoded.tobytes(), "image/jpeg")))
    return {"session_id": "session", "batch_id": "batch", "frames": frames}, files


def post(client, manifest, files):
    return client.post("/api/decode-burst", data={"manifest": json.dumps(manifest)}, files=files)


def test_burst_response_and_idempotent_retry(client, monkeypatch):
    calls = []
    original = service.run
    monkeypatch.setattr(service, "run", lambda image: (calls.append(1), original(image))[1])
    manifest, files = payload()
    response = post(client, manifest, files)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "confirmed"
    assert body["text"] == "00123"
    assert body["frame_count"] == 8
    assert body["frame_attempt_count"] == 2
    assert body["primary_image"].startswith("data:image/jpeg;base64,")
    assert body["result"]["images"]["original"]
    assert body["attempts"][0]["candidate_count"] == 8
    assert "selection" in body["timing_ms"]
    assert post(client, manifest, files).json() == body
    assert len(calls) == 2
    manifest["trigger"] = "continuous"
    assert post(client, manifest, files).status_code == 409


@pytest.mark.parametrize("change", ["duplicate", "time", "size", "mismatch", "extra", "nan"])
def test_bad_metadata(client, change):
    manifest, files = payload()
    if change == "duplicate":
        manifest["frames"][1]["frame_id"] = "f-0"
    elif change == "time":
        manifest["frames"][1]["offset_ms"] = 0
    elif change == "size":
        manifest["frames"][0]["width"] = 31
    elif change == "mismatch":
        manifest["frames"].pop()
    elif change == "extra":
        manifest["unknown"] = True
    else:
        manifest["frames"][0]["offset_ms"] = float("nan")
    assert post(client, manifest, files).status_code == 400


def test_file_count_and_corrupt_jpeg(client):
    assert post(client, *payload(9)).status_code == 400
    manifest, files = payload(1)
    files[0] = ("frames", ("bad.jpg", b"garbage", "image/jpeg"))
    assert post(client, manifest, files).status_code == 400


def test_one_frame_is_provisional(client):
    response = post(client, *payload(1))
    assert response.status_code == 200
    assert response.json()["status"] == "provisional"
    assert response.json()["ok"] is False


def test_pixel_limit_checked_before_opencv(client, monkeypatch):
    image = np.zeros((1200, 1920, 3), np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    manifest, _ = payload(1)
    manifest["frames"][0].update(width=1920, height=1200)
    files = [("frames", ("large.jpg", encoded.tobytes(), "image/jpeg"))]
    monkeypatch.setattr(burst.cv2, "imdecode", lambda *args: pytest.fail("must not decode"))
    assert post(client, manifest, files).status_code == 413


def test_upload_limit_including_chunked_body(client, monkeypatch):
    original = burst.read_limited_body

    async def small(request):
        return await original(request, 10)

    monkeypatch.setattr(burst, "read_limited_body", small)
    response = client.post("/api/decode-burst", content=iter([b"x" * 6, b"y" * 6]))
    assert response.status_code == 413


def test_runtime_not_ready_and_static_assets(client):
    server.runtime.ready = False
    assert post(client, *payload()).status_code == 503
    response = client.get("/assets/continuous.mjs")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert "LatestBatchQueue" in response.text


def test_single_and_burst_routes_share_the_inference_slot(client, monkeypatch):
    started, release = Event(), Event()

    def blocking(image):
        started.set()
        assert release.wait(5)
        return PipelineResult(image, None, None, DecodeResult("A", 0))

    monkeypatch.setattr(server, "run", blocking)
    _, encoded = cv2.imencode(".png", np.zeros((24, 32, 3), np.uint8))
    with ThreadPoolExecutor() as pool:
        request = pool.submit(client.post, "/api/decode", content=encoded.tobytes())
        assert started.wait(5)
        try:
            response = post(client, *payload())
            assert response.status_code == 429
            assert response.headers["Retry-After"] == "2"
        finally:
            release.set()
        assert request.result().status_code == 200
    assert post(client, *payload()).status_code == 200
