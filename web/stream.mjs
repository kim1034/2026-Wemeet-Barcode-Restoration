// 판독 먼저(GPU) 스트림: 프레임을 한 장씩 보내고, 응답에 담긴 결과를 바로 표시한다.
// 응답이 오기 전에는 다음 프레임을 보내지 않으므로 서버가 느리면 최신 프레임만 처리된다.
// 묶음 전송과 결과 조회(폴링)가 없다.

export class StreamGate {
  constructor({ minIntervalMs = 125 } = {}) {
    this.minIntervalMs = minIntervalMs;
    this.reset();
  }
  reset() {
    this.inflight = false;
    this.lastSent = -Infinity;
    this.sent = 0;
    this.busy = 0;
    this.failed = 0;
  }
  ready(now) { return !this.inflight && now - this.lastSent >= this.minIntervalMs; }
  begin(now) { this.inflight = true; this.lastSent = now; this.sent++; }
  end(outcome) {
    this.inflight = false;
    if (outcome === "busy") this.busy++;
    else if (outcome === "error") this.failed++;
  }
}

export class StreamEvents {
  constructor(onResult) { this.onResult = onResult; this.reset(); }
  reset() { this.seen = new Set(); }
  accept(events) {
    for (const event of events || []) {
      if (this.seen.has(event.event_id)) continue;
      this.seen.add(event.event_id);
      if (this.seen.size > 256) this.seen.delete(this.seen.values().next().value);
      this.onResult(event, { capturedAt: event.track_start_ms ?? performance.now() });
    }
  }
}

export async function uploadStreamFrame({ session, sequence, capturedAt, frameId, blob, width, height }) {
  const form = new FormData();
  form.append("manifest", JSON.stringify({ session_id: session, batch_id: `s-${sequence}`, sequence,
    captured_at_ms: capturedAt, window_ms: 0, trigger: "stream",
    frames: [{ frame_id: frameId, offset_ms: 0, width, height }] }));
  form.append("frames", blob, "frame.jpg");
  const abort = new AbortController(), timer = setTimeout(() => abort.abort(), 15000);
  try {
    const response = await fetch("/api/stream-frame", { method: "POST", body: form, signal: abort.signal });
    const body = await response.json();
    if (!response.ok) throw Object.assign(new Error(body.error || "판독 요청 실패"), { status: response.status });
    return body;
  } finally { clearTimeout(timer); }
}

export function createStreamCapture({ video, onResult, onState, onError }) {
  let session, enabled = false, generation = 0, callbackId = null, sequence = 0, last = {};
  const gate = new StreamGate();
  const events = new StreamEvents(onResult);
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const state = () => ({ tracking: true, stream: true, enabled, sent: gate.sent,
    busy_skipped: gate.busy, failed: gate.failed, ...last });
  const encode = () => {
    const scale = Math.min(1, 1920 / video.videoWidth, 1080 / video.videoHeight);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const { width, height } = canvas;
    return new Promise((resolve) => {
      try {
        canvas.toBlob((blob) => resolve(blob?.type === "image/jpeg" ? { blob, width, height } : null),
          "image/jpeg", 0.9);
      } catch { resolve(null); }
    });
  };
  const send = async (now, current) => {
    gate.begin(now);
    let outcome = "ok";
    try {
      const image = await encode();
      if (!image) { outcome = "error"; return; }
      sequence += 1;
      const body = await uploadStreamFrame({ session, sequence, capturedAt: now,
        frameId: `f-${sequence}`, ...image });
      if (current !== generation) return;
      last = { active_track: body.active_track, active_resolved: body.active_resolved,
        frames: body.frames, counts: body.event_counts || {}, pipeline_ms: body.timing_ms?.pipeline };
      events.accept(body.events);
    } catch (error) {
      outcome = error.status === 429 ? "busy" : "error";
      if (current === generation && outcome === "error") onError(error);
    } finally {
      gate.end(outcome);
      if (current === generation) onState(state());
    }
  };
  const schedule = () => {
    callbackId = video.requestVideoFrameCallback
      ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
  };
  const tick = (now) => {
    if (!enabled) return;
    if (!video.paused && !document.hidden && video.readyState >= 2 && video.videoWidth && gate.ready(now)) {
      send(now, generation);
    }
    schedule();
  };
  const stop = () => {
    const closing = enabled ? session : null;
    enabled = false;
    generation++;
    if (callbackId !== null) {
      if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(callbackId);
      else cancelAnimationFrame(callbackId);
      callbackId = null;
    }
    // 카메라를 멈추면 서버에 남은 구간(미판독 등)을 정리해 받아 온다.
    if (closing) {
      fetch(`/api/stream/${closing}`, { method: "DELETE", keepalive: true })
        .then((response) => response.json()).then((body) => events.accept(body.events)).catch(() => {});
    }
    onState(state());
  };
  return {
    start() {
      stop();
      enabled = true;
      session = crypto.randomUUID();
      sequence = 0;
      last = {};
      gate.reset();
      events.reset();
      schedule();
      onState(state());
    },
    stop,
    recalibrate() { this.start(); },
    setFiltering() {},
  };
}
