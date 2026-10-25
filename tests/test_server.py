"""wemeet/sw/server/ 검증. 모델 대신 pipeline 이 참조하는 stage 함수를 가짜로 바꿔 끼운다."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import wemeet.sw.server as server
from wemeet.schemas import DetectedBarcode, GeometryField

client = TestClient(server.app)


def _png(image_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", image_bgr)
    return buf.tobytes()


def _noise_bgr(seed: int) -> np.ndarray:
    gray = np.random.default_rng(seed).integers(0, 256, size=(120, 300), dtype=np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _fake_detect(image_bgr: np.ndarray) -> DetectedBarcode:
    return DetectedBarcode(crop_bgr_uint8=image_bgr, angle_deg_ccw=0.0, confidence=0.9)


def _fake_geometry(target: DetectedBarcode) -> GeometryField:
    dst = np.array([[x, y] for y in (0.0, 0.5, 1.0) for x in (0.0, 0.33, 0.66, 1.0)])
    return GeometryField(
        control_points_dst_norm=dst,
        control_points_src_norm=dst + 0.02,
        method="tps",
        confidence=1.0,
    )


def _post(data: bytes, content_type: str = "image/png"):
    return client.post("/api/decode", content=data, headers={"Content-Type": content_type})


def test_index_serves_the_operator_screen() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "/api/decode" in response.text


def test_health_reports_model_weights() -> None:
    body = client.get("/api/health").json()

    assert body["ok"] is True
    assert body["models"]["detection"].endswith(".pt")
    assert body["models"]["geometry"].endswith(".pt")


def test_decode_early_exit_returns_crop_without_rectification(
    monkeypatch: pytest.MonkeyPatch, clean_barcode_bgr: np.ndarray
) -> None:
    monkeypatch.setattr("wemeet.sw.pipeline.detect", _fake_detect)

    response = _post(_png(clean_barcode_bgr))
    body = response.json()

    assert response.status_code == 200
    assert body["ok"] is True
    assert body["text"] == "WEMEET0001"
    assert set(body["stage_ms"]) == {"detect", "decode_first"}
    assert body["detection"]["confidence"] == 0.9
    assert body["images"]["original"].startswith("data:image/jpeg;base64,")
    assert body["images"]["crop"].startswith("data:image/png;base64,")
    assert body["images"]["rectified"] is None
    assert body["rectify"] is None


def test_decode_failure_is_http_200_and_carries_the_rectified_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("wemeet.sw.pipeline.detect", _fake_detect)
    monkeypatch.setattr("wemeet.sw.pipeline.estimate_geometry", _fake_geometry)

    response = _post(_png(_noise_bgr(0)))
    body = response.json()

    assert response.status_code == 200
    assert body["ok"] is False
    assert body["failure_reason"] == "decode_failed"
    assert body["retry_count"] == 2
    assert body["rectify"]["grid"] == [4, 3]
    assert len(body["rectify"]["control_points_src_norm"]) == 12
    assert body["images"]["rectified"].startswith("data:image/png;base64,")


def test_decode_reports_not_detected(
    monkeypatch: pytest.MonkeyPatch, clean_barcode_bgr: np.ndarray
) -> None:
    monkeypatch.setattr("wemeet.sw.pipeline.detect", lambda image_bgr: None)

    body = _post(_png(clean_barcode_bgr)).json()

    assert body["failure_reason"] == "not_detected"
    assert body["detection"] is None
    assert body["images"]["crop"] is None


@pytest.mark.parametrize("data", [b"", b"not an image"], ids=["empty", "garbage"])
def test_decode_rejects_unreadable_input_with_400(data: bytes) -> None:
    response = _post(data, "application/octet-stream")

    assert response.status_code == 400
    assert response.json()["ok"] is False


def test_decode_rejects_oversized_input_with_413(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)

    response = _post(b"x" * 11, "application/octet-stream")

    assert response.status_code == 413
