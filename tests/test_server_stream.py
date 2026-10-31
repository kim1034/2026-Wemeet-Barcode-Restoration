import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import wemeet.sw.server as server
from wemeet.schemas import DecodeResult, DetectedBarcode, PipelineResult
from wemeet.sw.server.runtime import InferenceRuntime


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "runtime", InferenceRuntime(ready=True))
    monkeypatch.setattr(server, "stream_controller_instance", None)
    return TestClient(server.app)


def fake_run(readable=True):
    """밝기로 물품 유무를 정하는 가짜 판독. 밝으면 바코드 없음, 어두우면 x=밝기 위치에 바코드."""

    def run(image):
        level = int(image.mean())
        if level > 150:
            return PipelineResult(
                image, None, None, DecodeResult(None, 0, failure_reason="not_detected")
            )
        found = DetectedBarcode(
            np.zeros((40, 80, 3), np.uint8), 0.0, 0.9, (40 + level, 60, 120, 50)
        )
        text = "00123" if readable else None
        return PipelineResult(
            image,
            found,
            None,
            DecodeResult(text, 0, failure_reason=None if text else "decode_failed"),
        )

    return run


def frame(sequence, level=100, session="cam-1", captured=None):
    image = np.full((180, 320, 3), level, np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    manifest = {
        "session_id": session,
        "batch_id": f"s-{sequence}",
        "sequence": sequence,
        "captured_at_ms": captured if captured is not None else 1000 + sequence * 125,
        "window_ms": 0,
        "trigger": "stream",
        "frames": [{"frame_id": f"f-{sequence}", "offset_ms": 0, "width": 320, "height": 180}],
    }
    return {"manifest": json.dumps(manifest)}, [
        ("frames", ("frame.jpg", encoded.tobytes(), "image/jpeg"))
    ]


def post(client, data, files):
    return client.post("/api/stream-frame", data=data, files=files)


def test_second_matching_frame_returns_the_confirmed_result_in_its_own_response(
    client, monkeypatch, caplog
):
    monkeypatch.setattr("wemeet.sw.server.stream.run", fake_run())
    first = post(client, *frame(1, level=100))
    assert first.status_code == 200 and first.json()["events"] == []
    assert first.json()["active_track"] == "t-1" and first.json()["detected_box"]
    second = post(client, *frame(2, level=104)).json()
    [event] = second["events"]
    assert event["status"] == "confirmed" and event["text"] == "00123" and event["early"]
    assert event["result"]["text"] == "00123" and event["primary_image"].startswith(
        "data:image/jpeg"
    )
    assert set(event["timing_ms"]) >= {"pipeline", "server"}
    assert second["active_resolved"] and second["event_counts"] == {"confirmed": 1}
    # 같은 물품의 다음 프레임은 결과를 다시 내지 않는다
    assert post(client, *frame(3, level=108)).json()["events"] == []
    assert "00123" not in caplog.text and "data:image" not in caplog.text


def test_stopping_flushes_an_unreadable_passage(client, monkeypatch):
    monkeypatch.setattr("wemeet.sw.server.stream.run", fake_run(readable=False))
    for i in range(1, 5):
        assert post(client, *frame(i, level=100 + i * 4)).json()["events"] == []
    body = client.delete("/api/stream/cam-1").json()
    assert [(e["status"], e["track_end_reason"]) for e in body["events"]] == [("unread", "stopped")]
    # 닫힌 스트림의 늦은 프레임은 거부한다
    assert post(client, *frame(5)).status_code == 409


def test_stopping_while_a_frame_is_being_read_does_not_freeze_the_server(client, monkeypatch):
    # 2026-10-31 CHEETAH: 판독 중에 중지(DELETE)가 오면 서버 전체가 멈췄다. 중지가 이벤트 루프에서
    # 세션 잠금을 기다리는데, 그 잠금을 풀어야 할 판독 응답도 같은 이벤트 루프에서 돌기 때문이다.
    import threading

    reading, release = threading.Event(), threading.Event()
    readable = fake_run()

    def slow_run(image):
        reading.set()
        release.wait(10)
        return readable(image)

    monkeypatch.setattr("wemeet.sw.server.stream.run", slow_run)
    monkeypatch.setattr(server.runtime, "prepare", lambda: None)
    results = {}
    # 실제 서버처럼 모든 요청이 이벤트 루프 하나를 같이 쓰게 한다 (with 블록 = 루프 하나)
    with TestClient(server.app) as shared:
        posting = threading.Thread(
            target=lambda: results.update(post=post(shared, *frame(1)).status_code), daemon=True
        )
        posting.start()
        assert reading.wait(5)
        closing = threading.Thread(
            target=lambda: results.update(close=shared.delete("/api/stream/cam-1").status_code),
            daemon=True,
        )
        closing.start()
        closing.join(0.5)  # 중지는 판독이 끝날 때까지 기다린다
        release.set()
        posting.join(10)
        closing.join(10)
        assert results == {"post": 200, "close": 200}, "서버가 멈췄다"
        # 멈추지 않았다면 다른 요청에도 답한다
        assert shared.get("/api/health").status_code == 200


def test_order_and_input_are_validated(client, monkeypatch):
    monkeypatch.setattr("wemeet.sw.server.stream.run", fake_run())
    assert post(client, *frame(3)).status_code == 200
    assert post(client, *frame(3)).status_code == 409  # 같은 순번 재전송
    assert post(client, *frame(4, captured=500)).status_code == 409  # 촬영 시각 역행
    data, files = frame(5)
    manifest = json.loads(data["manifest"])
    manifest["frames"].append({"frame_id": "f-x", "offset_ms": 0, "width": 320, "height": 180})
    assert (
        post(client, {"manifest": json.dumps(manifest)}, files * 2).status_code == 400
    )  # 한 장만 허용


def test_busy_slot_and_unready_models_are_reported_without_processing(client, monkeypatch):
    monkeypatch.setattr("wemeet.sw.server.stream.run", fake_run())
    server.runtime._active = ("other", None)  # 다른 판독이 슬롯을 쓰는 중
    busy = post(client, *frame(1))
    assert busy.status_code == 429
    server.runtime._active = None
    server.runtime.ready = False
    assert post(client, *frame(2)).status_code == 503


def test_stream_module_is_served_and_is_the_default_capture_mode(client):
    assert client.get("/assets/stream.mjs").status_code == 200
    assert client.get("/assets/camera.mjs").status_code == 200
    page = client.get("/").text
    assert page.index('value="stream"') < page.index('value="barcode"')
