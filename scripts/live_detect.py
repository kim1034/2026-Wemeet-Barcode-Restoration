"""Stage 1 실시간 뷰어 — 카메라 화면에 검출 박스를 그린다. 실데이터 수집 도구를 겸한다.

    uv run python -m scripts.live_detect                       # 노트북 내장 카메라 (0번)
    uv run python -m scripts.live_detect --source 1            # USB 카메라
    uv run python -m scripts.live_detect --source rtsp://user:pw@192.168.0.10:554/stream1

SW 서버·웹 UI 와는 별개다 — `detect_obb()` 만 부른다. 화면 출력·카메라 연결은
이 파일 안에서 끝난다.

스레드 셋이 각자 돈다. 추론이 화면보다 느려도 화면은 카메라 속도 그대로 나오고,
박스는 마지막 추론 결과를 그린다. 캡처는 **가장 최근 프레임 하나만** 들고 있어서
추론이 밀려도 지연이 쌓이지 않는다 (IP 카메라 버퍼에 몇 초 전 화면이 쌓이는 문제).

    캡처   → 최신 프레임 1장
    추론   → 최신 프레임으로 detect_obb() → 최신 결과 1개
    화면   → 최신 프레임 + 최신 결과를 그림, 키 입력

키:  s 지금 프레임 저장 · r 연속 저장 켜기/끄기 · q / Esc 종료

저장은 `--save-dir/<세션 시각>/` 에 원본 jpg 와 같은 이름의 YOLO-OBB 라벨 txt
(`class x1 y1 ... x4 y4`, 0~1 정규화, 0=5:3 1=6:4) 를 쓴다. 라벨은 **모델이 낸
예비 라벨**이다 — CVAT/Roboflow 에서 사람이 검수해야 정답이 된다
(docs/decisions/0004). 미검출 프레임은 빈 txt 를 쓴다.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import wemeet.ai.detection as detection
from wemeet.ai.best_frame import compute_glare_ratio, compute_sharpness_score
from wemeet.ai.detection import ObbDetection, detect_obb
from wemeet.ai.detection.geometry import RATIO_5X3

WINDOW = "wemeet Stage 1 live"
GREEN, RED, WHITE, BLACK = (0, 220, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)


# ── 그리기·저장 (순수 함수, 테스트 대상) ────────────────────────────


def draw_obb(image: np.ndarray, found: ObbDetection, color: tuple = GREEN) -> np.ndarray:
    """회전 박스와 규격·확신도 꼬리표를 그린다. 입력을 직접 고친다."""
    pts = np.round(found.corners_px).astype(np.int32).reshape(-1, 1, 2)
    thick = max(2, image.shape[1] // 640)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thick)
    spec = "5:3" if found.ratio == RATIO_5X3 else "6:4"
    tag = f"{spec} {found.confidence:.2f}"
    x, y = int(found.corners_px[:, 0].min()), int(found.corners_px[:, 1].min())
    scale = thick * 0.35
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(th + 4, y - 4)
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(image, tag, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick)
    return image


def draw_hud(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """왼쪽 위에 상태 줄을 그린다. 한글은 cv2 폰트가 못 그려서 영문만 쓴다."""
    scale = max(0.5, image.shape[1] / 1920 * 0.8)
    step = int(32 * scale / 0.8)
    for i, line in enumerate(lines):
        org = (10, step * (i + 1))
        cv2.putText(image, line, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4)
        cv2.putText(image, line, org, cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1)
    return image


def obb_label_line(found: ObbDetection, width: int, height: int) -> str:
    """YOLO-OBB 라벨 한 줄. 학습 데이터(dataset_720)와 같은 클래스 번호를 쓴다."""
    cls = 0 if found.ratio == RATIO_5X3 else 1
    norm = found.corners_px / np.array([width, height], dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)  # 박스가 화면 밖으로 조금 나가면 라벨 도구가 거부한다
    return f"{cls} " + " ".join(f"{v:.6f}" for v in norm.reshape(-1))


# ── 스레드 ────────────────────────────────────────────────────────


def parse_source(text: str) -> int | str:
    return int(text) if text.isdigit() else text


def open_capture(source: int | str, width: int, height: int) -> cv2.VideoCapture:
    if isinstance(source, int):
        # Windows 기본 백엔드(MSMF)는 여는 데 수 초 걸리고 해상도 지정이 잘 안 먹는다
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


@dataclass
class Latest:
    """스레드끼리 주고받는 '가장 최근 값 하나'. seq 로 새 값인지 구분한다."""

    value: object = None
    seq: int = 0


class Capture(threading.Thread):
    """카메라에서 계속 읽어 최신 프레임만 남긴다. 끊기면 1초마다 다시 연다(IP 카메라)."""

    def __init__(self, source: int | str, width: int, height: int) -> None:
        super().__init__(daemon=True)
        self.source, self.width, self.height = source, width, height
        self.latest = Latest()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.fps = 0.0

    def run(self) -> None:
        cap = open_capture(self.source, self.width, self.height)
        last = time.perf_counter()
        while not self.stopped.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.release()
                time.sleep(1.0)
                cap = open_capture(self.source, self.width, self.height)
                continue
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
            last = now
            with self.lock:
                self.latest = Latest(frame, self.latest.seq + 1)
        cap.release()

    def get(self) -> Latest:
        with self.lock:
            return self.latest


@dataclass
class Inference:
    found: ObbDetection | None
    ms: float
    frame: np.ndarray
    frame_seq: int
    glare: float | None = None
    sharp: float | None = None


class Detector(threading.Thread):
    """최신 프레임이 바뀔 때마다 detect_obb() 를 돌린다."""

    def __init__(self, capture: Capture, quality: bool) -> None:
        super().__init__(daemon=True)
        self.capture = capture
        self.quality = quality
        self.latest = Latest()
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def run(self) -> None:
        done = 0
        while not self.stopped.is_set():
            snap = self.capture.get()
            if snap.value is None or snap.seq == done:
                time.sleep(0.001)
                continue
            frame = snap.value
            start = time.perf_counter()
            found = detect_obb(frame)
            ms = (time.perf_counter() - start) * 1000
            result = Inference(found, ms, frame, snap.seq)
            if self.quality and found is not None:
                # Stage 0 이 고를 때 보는 두 값을 박스 영역에서만 잰다
                x0, y0 = np.floor(found.corners_px.min(axis=0)).astype(int).clip(0)
                x1, y1 = np.ceil(found.corners_px.max(axis=0)).astype(int)
                roi = frame[y0:y1, x0:x1]
                if roi.size:
                    result.glare = compute_glare_ratio(roi)
                    result.sharp = compute_sharpness_score(roi)
            done = snap.seq
            with self.lock:
                self.latest = Latest(result, self.latest.seq + 1)

    def get(self) -> Latest:
        with self.lock:
            return self.latest


# ── 화면 루프 ─────────────────────────────────────────────────────


def save_sample(folder: Path, result: Inference) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%H%M%S_%f}"
    path = folder / f"{stem}.jpg"
    ok, buf = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(str(path))  # imwrite 는 Windows 에서 한글 경로를 못 쓴다
    h, w = result.frame.shape[:2]
    label = obb_label_line(result.found, w, h) + "\n" if result.found else ""
    (folder / f"{stem}.txt").write_text(label, encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source", default="0", help="카메라 번호(0, 1, ...) 또는 rtsp:// http:// 주소"
    )
    ap.add_argument("--width", type=int, default=1920, help="요청 해상도 (카메라가 지원해야 적용)")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--imgsz", type=int, default=None, help="검출 입력 크기. 기본은 DEFAULT_IMGSZ")
    ap.add_argument("--conf", type=float, default=None, help="확신도 임계값. 기본은 CONF_THRESHOLD")
    ap.add_argument(
        "--quality", action="store_true", help="박스 영역의 반사율·선명도(Stage 0 지표) 표시"
    )
    ap.add_argument("--save-dir", type=Path, default=Path("captures"))
    ap.add_argument(
        "--record-every", type=int, default=5, help="연속 저장(r) 때 N번째 추론마다 1장"
    )
    ap.add_argument("--display-width", type=int, default=1280, help="창 크기. 저장은 원본 해상도")
    args = ap.parse_args()

    if args.imgsz:
        detection.DEFAULT_IMGSZ = args.imgsz
    if args.conf is not None:
        detection.CONF_THRESHOLD = args.conf

    print("모델 로드 중...", flush=True)
    detect_obb(np.zeros((args.height, args.width, 3), np.uint8))  # 가중치 로드 + CUDA warmup

    capture = Capture(parse_source(args.source), args.width, args.height)
    detector = Detector(capture, args.quality)
    capture.start()
    detector.start()

    session = args.save_dir / f"{datetime.now():%Y%m%d_%H%M%S}"
    recording, saved, recorded_seq = False, 0, 0
    shown_fps, last = 0.0, time.perf_counter()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:
            snap = capture.get()
            if snap.value is None:
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    break
                continue
            frame = snap.value.copy()
            inf_snap = detector.get()
            result: Inference | None = inf_snap.value

            if result is not None and result.found is not None:
                draw_obb(frame, result.found)
            lines = [
                f"cam {capture.fps:4.1f} fps | view {shown_fps:4.1f} fps",
                f"detect {result.ms:5.1f} ms | imgsz {detection.DEFAULT_IMGSZ}"
                if result
                else "detect -",
                f"{frame.shape[1]}x{frame.shape[0]} | saved {saved}"
                + (" | REC" if recording else ""),
            ]
            if result is not None and result.glare is not None:
                lines.append(f"glare {result.glare * 100:4.1f}% | sharp {result.sharp:7.0f}")
            draw_hud(frame, lines)
            if recording:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12, RED, -1)

            scale = args.display_width / frame.shape[1]
            view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            cv2.imshow(WINDOW, view)

            now = time.perf_counter()
            shown_fps = 0.9 * shown_fps + 0.1 / max(now - last, 1e-6)
            last = now

            if (
                recording
                and result is not None
                and inf_snap.seq - recorded_seq >= args.record_every
            ):
                save_sample(session, result)
                saved, recorded_seq = saved + 1, inf_snap.seq

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and result is not None:
                path = save_sample(session, result)
                saved += 1
                print(f"저장: {path}", flush=True)
            if key == ord("r"):
                recording = not recording
                print(f"연속 저장 {'시작' if recording else '중지'} → {session}", flush=True)
    finally:
        detector.stopped.set()
        capture.stopped.set()
        cv2.destroyAllWindows()
        if saved:
            print(f"총 {saved}장 저장 → {session}")


if __name__ == "__main__":
    main()
