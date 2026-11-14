"""In-memory conveyor replay; writes aggregate metrics ONLY, never images/text.

Both modes receive identical 1080p JPEG90 frames at 8fps. A virtual single
inference slot uses explicit assumed costs, so host pauses cannot change which
items are tested. Actual decoder outputs and process CPU times are measured.
Queue latency/throughput are simulations, NOT camera or conveyor measurements.
"""

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from wemeet.ai.detection import detect
from wemeet.ai.geometry import estimate_geometry
from wemeet.schemas import DetectedBarcode
from wemeet.sw.frame_batch import Frame, FrameBatch, process_batch
from wemeet.sw.tracking import BarcodeBoxes, Tracker


class SlotBusy(Exception):
    pass


class ActivityGate:
    """Replay web/activity.mjs at the same 8fps; never use truth to gate uploads."""

    def __init__(self):
        self.background = None
        self.previous = None
        self.streak = 0
        self.last_active = -float("inf")

    def update(self, frame):
        small = cv2.resize(frame.image_bgr, (160, 90)).astype(np.float32)
        gray = np.floor(small.mean(axis=2) + 0.5).astype(np.int16)
        if self.background is None:
            self.background = gray.copy()
            self.previous = gray.copy()
            return False
        ratio = max(
            np.mean(np.abs(gray - self.background) > 18), np.mean(np.abs(gray - self.previous) > 18)
        )
        self.previous = gray
        self.streak = self.streak + 1 if ratio >= 0.025 else 0
        if self.streak >= 2:
            self.last_active = frame.offset_ms
        return frame.offset_ms - self.last_active < 300


def scenes(root):
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("band") == "first_ok"][:3]
    patches = []
    for row in rows:
        image = cv2.imdecode(np.fromfile(root / row["file"], np.uint8), cv2.IMREAD_COLOR)
        x, y, w, h = row["crop_box_xywh"]
        patch = image[max(0, y - 50) : y + h + 50, max(0, x - 50) : x + w + 50].copy()
        patches.append((patch, row["text"]))
    # Separate items, empty intervals, repeated product, close items, fast passage.
    schedule = [
        (1000, 5000, 0),
        (8000, 5000, 1),
        (15000, 5000, 0),
        (22000, 4000, 2),
        (26500, 4000, 1),
        (32000, 1000, 0),
    ]
    return patches, schedule


def stream(patches, schedule):
    for t in range(0, 39000, 125):
        image = np.full((1080, 1920, 3), 45, np.uint8)
        item_id = None
        for i, (start, duration, index) in enumerate(schedule):
            if start <= t < start + duration:
                patch = patches[index][0]
                h, w = patch.shape[:2]
                x = round(-w + (1920 + w) * (t - start) / duration)
                left, right = max(0, x), min(1920, x + w)
                if right > left:
                    image[440 : 440 + h, left:right] = patch[:, left - x : right - x]
                    item_id = i
        # Match browser JPEG encoding for both modes, without filesystem writes.
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        yield Frame(f"f-{t}", t, image), item_id


def evaluate(mode, patches, schedule):
    tracker = Tracker()
    pending = deque()
    records, members, windows = [], {}, []
    occupied = 0.0
    clock = 0.0
    inference_ms = 0.0
    tracking_ms = 0.0
    process_cpu_s = 0.0
    detector_wall_ms = []
    batch_wall_ms = []
    dropped = 0
    detection_calls = 0
    detection_skipped = 0
    frames_seen = 0
    activity = ActivityGate()
    window_active = False
    max_pending = 2 if mode == "tracking" else 1

    def detect_slot(fn):
        nonlocal occupied, inference_ms, detection_calls, detection_skipped, process_cpu_s
        if clock < occupied:
            detection_skipped += 1
            raise SlotBusy
        start = time.perf_counter()
        cpu_start = time.process_time()
        result = fn()
        process_cpu_s += time.process_time() - cpu_start
        detector_wall_ms.append((time.perf_counter() - start) * 1000)
        cost = 1000.0  # Assumption within the user's 0.65–1.1s CPU measurements.
        occupied = clock + cost
        inference_ms += cost
        detection_calls += 1
        return result

    source = BarcodeBoxes(execute=detect_slot)

    def enqueue(batch, end_ms):
        nonlocal dropped
        if len(pending) >= max_pending:
            dropped += 1
            if mode == "tracking":
                owner = members.get(batch.frames[0].frame_id) if batch.frames else None
                records.append(
                    {
                        "owner": owner,
                        "status": "unread",
                        "correct": False,
                        "wrong_confirmed": False,
                        "finish_ms": clock,
                        "close_ms": end_ms,
                        "processing_ms": 0,
                        "attempts": 0,
                    }
                )
                return
            pending.popleft()  # Same latest-window policy as the baseline browser.
        pending.append((batch, end_ms))

    def work():
        nonlocal occupied, inference_ms, process_cpu_s
        if not pending or clock < occupied:
            return
        batch, closed_at = pending.popleft()
        started = time.perf_counter()
        cpu_start = time.process_time()
        # Host scheduling pauses must not exhaust the live 9s admission budget.
        # Three assumed 2s attempts + selection fit that budget. Live behavior
        # remains unchanged; its admission/overload rules have separate tests.
        outcome = process_batch(batch, budget_ms=float("inf"))
        process_cpu_s += time.process_time() - cpu_start
        batch_wall_ms.append((time.perf_counter() - started) * 1000)
        cost = sum(2000 + 39 * a["candidate_count"] for a in outcome.attempts)
        inference_ms += cost
        occupied = clock + cost
        primary = outcome.primary_frame.frame_id if outcome.primary_frame else None
        owner = members.get(primary)
        # Attribution uses the selected frame's known item identity, not decoded text.
        truth = patches[schedule[owner][2]][1] if owner is not None else None
        records.append(
            {
                "owner": owner,
                "status": outcome.status,
                "correct": truth is not None and outcome.text == truth,
                "wrong_confirmed": outcome.status == "confirmed" and outcome.text != truth,
                "finish_ms": occupied,
                "close_ms": closed_at,
                "processing_ms": cost,
                "attempts": len(outcome.attempts),
            }
        )

    wall_start = time.perf_counter()
    for frame, owner in stream(patches, schedule):
        clock = frame.offset_ms
        members[frame.frame_id] = owner
        frames_seen += 1
        work()
        if mode == "tracking":
            try:
                boxes = source.observe(frame)
            except SlotBusy:
                boxes = None
            start = time.perf_counter()
            cpu_start = time.process_time()
            done = tracker.update(frame, boxes)
            process_cpu_s += time.process_time() - cpu_start
            tracking_ms += (time.perf_counter() - start) * 1000
            for passage in done:
                enqueue(passage.batch("replay"), passage.end_ms)
        else:
            windows.append(frame)
            window_active |= activity.update(frame)
            if len(windows) == 8:
                if window_active:
                    start_ms = windows[0].offset_ms
                    batch = FrameBatch(
                        "replay",
                        f"b-{int(start_ms)}",
                        [Frame(f.frame_id, f.offset_ms - start_ms, f.image_bgr) for f in windows],
                    )
                    enqueue(batch, start_ms + 1000)
                windows = []
                window_active = False
        if frames_seen % 80 == 0:
            print(f"{mode}: {frames_seen}/312 frames", flush=True)
    if mode == "tracking":
        for passage in tracker.close(clock, "end_of_input"):
            enqueue(passage.batch("replay"), clock)
    while pending:
        clock = max(clock, occupied)
        work()
    per_item = []
    for i, (start, duration, _) in enumerate(schedule):
        found = [r for r in records if r["owner"] == i]
        per_item.append(
            {
                "item": i,
                "results": len(found),
                "missing": not found,
                "correct_read": any(r["correct"] for r in found),
                "correct_confirmed": any(
                    r["correct"] and r["status"] == "confirmed" for r in found
                ),
                "wrong_confirmed": sum(r["wrong_confirmed"] for r in found),
                "provisional": sum(r["status"] == "provisional" for r in found),
                "duplicates": max(0, len(found) - 1),
                "first_result_relative_to_exit_ms": round(
                    min(r["finish_ms"] for r in found) - start - duration, 1
                )
                if found
                else None,
            }
        )
    delays = [p["first_result_relative_to_exit_ms"] for p in per_item if not p["missing"]]
    replay_wall_s = time.perf_counter() - wall_start
    return {
        "items": len(schedule),
        "frames": frames_seen,
        "duration_s": 39,
        "correct_read_items": sum(p["correct_read"] for p in per_item),
        "correct_confirmed_items": sum(p["correct_confirmed"] for p in per_item),
        "wrong_confirmed_results": sum(r["wrong_confirmed"] for r in records),
        "provisional_results": sum(p["provisional"] for p in per_item),
        "missing_items": sum(p["missing"] for p in per_item),
        "duplicate_results": sum(p["duplicates"] for p in per_item),
        "unassigned_results": sum(r["owner"] is None for r in records),
        "statuses": dict(Counter(r["status"] for r in records)),
        "simulated_delay_from_exit_p50_ms": round(float(np.median(delays)), 1) if delays else None,
        "observed_batch_wall_ms_p50": round(float(np.median(batch_wall_ms)), 1)
        if batch_wall_ms
        else None,
        "observed_detector_wall_ms_p50": round(float(np.median(detector_wall_ms)), 1)
        if detector_wall_ms
        else None,
        "observed_call_wall_ms_max": round(max(batch_wall_ms + detector_wall_ms, default=0), 1),
        "timing_outlier": any(t > 30000 for t in batch_wall_ms + detector_wall_ms),
        "simulated_slot_busy_s": round(inference_ms / 1000, 2),
        "measured_process_cpu_s": round(process_cpu_s, 2),
        "observed_tracking_wall_ms": round(tracking_ms, 1),
        "replay_wall_s_including_jpeg": round(replay_wall_s, 2),
        "observed_input_replay_fps": round(frames_seen / replay_wall_s, 2),
        "simulated_input_frames_per_slot_second": round(frames_seen * 1000 / inference_ms, 2),
        "queue_drops": dropped,
        "detection_calls": detection_calls,
        "detection_skipped_busy": detection_skipped,
        "per_item": per_item,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("downloads/benchmark-v1"))
    ap.add_argument("--output", type=Path, default=Path("runs/tracking-summary.json"))
    args = ap.parse_args()
    patches, schedule = scenes(args.root)
    # Exclude weight loading and warm-up from measured compute cost.
    detect(np.full((1080, 1920, 3), 127, np.uint8))
    estimate_geometry(DetectedBarcode(np.full((160, 384, 3), 127, np.uint8), 0, 1))
    report = {
        "method": (
            "assumed-cost virtual single-slot replay; actual decoding; no network/camera timing"
        ),
        "assumed_cost_ms": {
            "detection": 1000,
            "pipeline_attempt": 2000,
            "selection_per_candidate": 39,
        },
        "evaluation_budget": "wall budget disabled during replay; live 9s budget unchanged",
        "cpu_time_note": (
            "process CPU seconds summed across CPU threads; "
            "excludes JPEG/network; not wall throughput"
        ),
        "latency_note": "simulated first result minus item exit; negative means before exit",
        "source_scenes": 3,
        "independent_field_samples": 0,
        "seed": 2026,
        "fps": 8,
        "detection_interval_ms": 2000,
        "resolution": [1920, 1080],
        "jpeg_quality": 90,
        "baseline": evaluate("baseline", patches, schedule),
        "tracking": evaluate("tracking", patches, schedule),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
