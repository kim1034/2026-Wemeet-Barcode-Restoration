import { startFrameCapture } from "./capture.mjs";

// Transport is bounded; a lost chunk is counted, never silently replayed as new frames.
export class TrackQueue {
  constructor({ send, onReply = () => {}, onError = () => {}, now = () => performance.now(),
    schedule = (fn) => setTimeout(fn, 1000), cancel = (id) => clearTimeout(id) }) {
    Object.assign(this, { send, onReply, onError, now, schedule, cancel });
    this.busy = false;
    this.pending = [];
    this.skipped = 0;
  }
  start(session) { this.stop(); this.session = session; this.enabled = true; }
  stop() { this.enabled = false; this.pending = []; this.cancel(this.timer); this.retrying = false; }
  offer(chunk) {
    if (!this.enabled || chunk.session_id !== this.session) return;
    if (this.pending.length === 2) { this.pending.shift(); this.skipped++; }
    this.pending.push(chunk);
    this.pump();
  }
  async pump() {
    if (!this.enabled || this.busy || this.retrying || !this.pending.length) return;
    const chunk = this.pending.shift();
    if (this.now() - chunk.capturedAt > 2500) { this.skipped++; return this.pump(); }
    this.busy = true;
    try {
      const reply = await this.send(chunk);
      if (this.enabled && chunk.session_id === this.session) {
        if (reply.session_id !== this.session || reply.sequence !== chunk.sequence) {
          throw new Error("추적 응답의 세션/순서가 다릅니다.");
        }
        this.onReply(reply);
      }
    } catch (error) {
      if (this.enabled && chunk.session_id === this.session) {
        this.onError(error);
        if (error.retryable) {
          // Keep only a bounded amount; the in-flight ID is unchanged for retries.
          if (this.pending.length >= 2) { this.pending.pop(); this.skipped++; }
          this.pending.unshift(chunk);
          this.retrying = true;
          this.timer = this.schedule(() => { this.retrying = false; this.pump(); });
        } else this.skipped++;
      }
    } finally { this.busy = false; this.pump(); }
  }
}

export class TrackResults {
  constructor(onResult) { this.onResult = onResult; this.reset(); }
  reset() { this.seen = new Set(); this.cursor = 0; }
  accept(results) {
    for (const result of results || []) {
      this.cursor = Math.max(this.cursor, result.result_sequence || 0);
      if (this.seen.has(result.track_id)) continue;
      this.seen.add(result.track_id);
      if (this.seen.size > 256) this.seen.delete(this.seen.values().next().value);
      this.onResult(result, { capturedAt: result.track_start_ms ?? performance.now() });
    }
  }
}

export async function uploadTrackChunk(chunk) {
  const form = new FormData();
  form.append("manifest", JSON.stringify({ session_id: chunk.session_id,
    batch_id: `c-${chunk.sequence}`, sequence: chunk.sequence, window_ms: 500,
    captured_at_ms: chunk.capturedAt, box_source: chunk.box_source, trigger: "continuous",
    frames: chunk.frames.map(({ frame_id, offset_ms, width, height }) =>
      ({ frame_id, offset_ms, width, height })) }));
  for (const frame of chunk.frames) form.append("frames", frame.blob, "frame.jpg");
  const abort = new AbortController(), timer = setTimeout(() => abort.abort(), 15000);
  try {
    const response = await fetch("/api/track-frames", { method: "POST", body: form, signal: abort.signal });
    const body = await response.json();
    if (!response.ok) throw Object.assign(new Error(body.error || "추적 요청 실패"),
      { retryable: response.status === 429 || response.status === 503 });
    return body;
  } catch (error) {
    if (error instanceof TypeError || error.name === "AbortError") error.retryable = true;
    throw error;
  } finally { clearTimeout(timer); }
}

export function createTrackingCapture({ video, onResult, onState, onError, source = "barcode" }) {
  let session, stopFrames, timer, pollAbort, enabled = false, generation = 0, pollBusy = false;
  const results = new TrackResults(onResult);
  const reply = (body) => {
    results.accept(body.results);
    onState({ ...body, enabled, tracking: true, busy: body.decoding,
      skipped: queue.skipped, box_source: source });
  };
  const queue = new TrackQueue({ send: uploadTrackChunk, onReply: reply, onError });
  const stop = () => {
    enabled = false; generation++;
    stopFrames?.(); stopFrames = null;
    clearInterval(timer); queue.stop();
    pollAbort?.abort();
    if (session) fetch(`/api/tracking/${session}`, { method: "DELETE", keepalive: true }).catch(() => {});
    onState({ tracking: true, enabled: false, completed: 0, skipped: queue.skipped, pending: 0 });
  };
  return {
    start() {
      stop(); enabled = true; session = crypto.randomUUID(); results.reset();
      const current = generation;
      queue.start(session);
      stopFrames = startFrameCapture(video, { windowMs: 500, maxFrames: 4, getActivity: () => true,
        onStall: (stalled) => { if (stalled) onError(new Error("영상 수신 중단 · 추적 구간이 끊길 수 있습니다.")); },
        onBatch: (chunk) => {
          if (enabled && current === generation) queue.offer({ ...chunk, session_id: session, box_source: source });
        } });
      timer = setInterval(async () => {
        if (pollBusy || !enabled) return;
        pollBusy = true;
        pollAbort = new AbortController();
        const timeout = setTimeout(() => pollAbort?.abort(), 15000);
        try {
          const response = await fetch(`/api/tracking/${session}?after=${results.cursor}`, { signal: pollAbort.signal });
          if (!response.ok) throw new Error("추적 상태를 받지 못했습니다.");
          const body = await response.json();
          if (enabled && current === generation) reply(body);
        } catch (error) { if (enabled && current === generation) onError(error); }
        finally { clearTimeout(timeout); pollAbort = null; pollBusy = false; }
      }, 500);
    },
    stop,
    recalibrate() { this.start(); },
    setFiltering() {},
  };
}
