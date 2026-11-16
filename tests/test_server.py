"""wemeet/sw/server/ 검증. 모델 대신 pipeline 이 참조하는 stage 함수를 가짜로 바꿔 끼운다."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import wemeet.sw.server as server
from wemeet.schemas import DetectedBarcode, GeometryField
from wemeet.sw.server.runtime import InferenceRuntime

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def ready_runtime(monkeypatch):
    monkeypatch.setattr(server, "runtime", InferenceRuntime(ready=True))


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


@pytest.mark.parametrize("name", ["intro.mjs", "three.module.min.js"])
def test_intro_scripts_are_served_from_our_own_server(name: str) -> None:
    """첫 화면 인트로는 CDN 이 아니라 우리 서버에서 받는다 — 망이 끊긴 현장에서도 떠야 한다."""
    response = client.get("/assets/" + name)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")


def test_page_loads_three_js_only_from_local_assets() -> None:
    page = client.get("/").text + client.get("/assets/intro.mjs").text

    assert '"/assets/three.module.min.js"' in page
    assert "cdn.jsdelivr" not in page and "unpkg.com" not in page


def test_live_view_is_one_video_stage_with_result_and_log_floating_on_it() -> None:
    """실시간 화면은 무대(영상) 하나 위에 섬 둘(판독 대기 · 판독 로그)이 떠 있다.
    판독 집계와 선택한 프레임은 판독 이력 화면으로 옮겼다."""
    page = client.get("/").text
    live = page[page.index('id="view-live"') : page.index('id="view-history"')]
    history = page[page.index('id="view-history"') : page.index('id="view-review"')]
    stage = live[live.index('class="stage"') :]
    hud = stage[stage.index('class="hud"') :]

    assert 'id="videoBox"' in stage
    # 위 섬이 판독 대기(최근 판독 결과), 아래 섬이 판독 로그
    assert hud.index('class="panel result-card"') < hud.index('class="panel log-panel"')
    assert 'id="readLog"' in hud
    for moved in ('id="tallyFirst"', 'id="burstPanel"'):
        assert moved not in live
        assert moved in history


def test_font_is_pretendard_served_from_our_own_server() -> None:
    """글꼴은 Pretendard 하나, 우리 서버가 내준다 — 망이 끊긴 현장에서도 글자가 그대로여야 한다."""
    page = client.get("/").text
    response = client.get("/fonts/Pretendard.woff2")

    assert "fonts.googleapis" not in page and "IBM Plex" not in page
    assert 'url("/fonts/Pretendard.woff2")' in page
    assert response.status_code == 200
    assert response.headers["content-type"] == "font/woff2"
    assert response.content[:4] == b"wOF2"


@pytest.mark.parametrize(
    "name",
    ["../index.html", "../../pyproject.toml", "없는글꼴.woff2", "Pretendard.ttf"],
    ids=["escapes-one-level", "escapes-to-repo-root", "missing", "wrong-extension"],
)
def test_font_route_serves_nothing_but_woff2_inside_web_fonts(name: str) -> None:
    """이름을 그대로 경로에 붙이면 web/ 밖의 파일이 나간다. 404 로 막는다."""
    assert client.get("/fonts/" + name).status_code == 404


def test_font_route_strips_the_directory_part_itself() -> None:
    """위 테스트는 httpx 가 `../` 를 먼저 정리해 버려 통과할 수도 있다.
    경로를 깎는 일이 핸들러 안에서 일어나는지 직접 불러 확인한다."""
    assert server.font("../../pyproject.toml").status_code == 404
    assert server.font("/etc/passwd").status_code == 404
    assert server.font("web/fonts/Pretendard.woff2").status_code == 200
