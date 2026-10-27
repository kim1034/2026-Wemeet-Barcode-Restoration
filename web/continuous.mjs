import { ActivityDetector } from "./activity.mjs";
import { startFrameCapture } from "./capture.mjs";

export class LatestBatchQueue {
  constructor({ send, onResult = () => {}, onChange = () => {}, onError = () => {},
    now = () => performance.now(), schedule = (fn, ms) => setTimeout(fn, ms),
    cancel = (timer) => clearTimeout(timer) }) {
    Object.assign(this, { send, onResult, onChange, onError, now, schedule, cancel });
    this.enabled = false;
    this.pending = null;
    this.busy = false;
    this.skipped = 0;
    this.completed = 0;
    this.retryAt = 0;
  }
  start(sessionId) {
    this.stop();
    this.sessionId = sessionId;
    this.latestSequence = -1;
    this.enabled = true;
    this.notify();
  }
  stop() {
    this.enabled = false;
    this.pending = null;
    this.cancel(this.timer);
    this.retryAt = 0;
    this.notify();
  }
  notify() {
    this.onChange({ enabled: this.enabled, busy: this.busy, pending: !!this.pending,
      skipped: this.skipped, completed: this.completed });
  }
  offer(batch) {
    if (!this.enabled || batch.session_id !== this.sessionId || !batch.active) return;
    if (batch.sequence <= this.latestSequence) { this.skipped++; this.notify(); return; }
    this.latestSequence = batch.sequence;
    if (this.pending) this.skipped++;
    this.pending = batch;
    this.notify();
    this.pump();
  }
  async pump() {
    if (!this.enabled || this.busy || !this.pending) return;
    if (this.now() < this.retryAt) return;
    const batch = this.pending;
    this.pending = null;
    if (this.now() - batch.capturedAt > 15000) {
      this.skipped++;
      this.notify();
      return;
    }
    this.busy = true;
    this.notify();
    try {
      const response = await this.send(batch);
      if (this.enabled && batch.session_id === this.sessionId) {
        if (response.session_id !== batch.session_id || response.batch_id !== batch.batch_id) {
          throw new Error("응답의 촬영 구간이 일치하지 않습니다.");
        }
        this.completed++;
        this.onResult(response, batch);
      }
    } catch (error) {
      if (this.enabled && batch.session_id === this.sessionId) {
        this.onError(error);
        if (error.retryable) {
          if (!this.pending) this.pending = batch;
          else this.skipped++;
          this.retryAt = this.now() + 2000;
          this.cancel(this.timer);
          this.timer = this.schedule(() => this.pump(), 2000);
        } else this.skipped++;
      }
    } finally {
      this.busy = false;
      this.notify();
      this.pump();
    }
  }
}

export async function uploadBatch(batch) {
  const form = new FormData();
  const manifest = { session_id: batch.session_id, batch_id: batch.batch_id,
    window_ms: 1000, source_id: "browser", jpeg_quality: 0.9, trigger: batch.trigger,
    frames: batch.frames.map(({ frame_id, offset_ms, width, height }) =>
      ({ frame_id, offset_ms, width, height })) };
  form.append("manifest", JSON.stringify(manifest));
  for (const frame of batch.frames) form.append("frames", frame.blob, `${frame.frame_id}.jpg`);
  if ([...form.getAll("frames")].reduce((sum, file) => sum + file.size, 0) > 14.9 * 1024 * 1024) {
    throw new Error("촬영 묶음이 15MiB를 초과했습니다. 카메라 해상도를 확인해 주세요.");
  }
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), 45000);
  try {
    const response = await fetch("/api/decode-burst", { method: "POST", body: form, signal: abort.signal });
    const body = await response.json();
    if (!response.ok) {
      const messages = { inference_busy: "다른 판독 처리 중 · 자동 재시도",
        models_preparing: "모델 준비 중 · 자동 재시도",
        model_prepare_failed: "모델 준비 실패 · 가중치 설정 후 서버를 다시 시작하세요." };
      const error = new Error(messages[body.error] || body.error || `HTTP ${response.status}`);
      error.retryable = response.status === 429 || response.status === 503;
      throw error;
    }
    return body;
  } catch (error) {
    if (error instanceof TypeError || error.name === "AbortError") {
      error.retryable = true;
      error.message = "서버 응답을 기다리지 못했습니다 · 자동 재시도 (실행 중 판독은 계속될 수 있음)";
    }
    throw error;
  } finally { clearTimeout(timer); }
}

export function createContinuousCapture({ video, onResult, onState, onError }) {
  const activity = new ActivityDetector();
  const tiny = document.createElement("canvas");
  tiny.width = 160; tiny.height = 90;
  const context = tiny.getContext("2d", { willReadFrequently: true });
  let stopCapture = null, filtering = true, lastCheck = -Infinity, active = false;
  let sessionId, activeGeneration = 0;
  const queue = new LatestBatchQueue({ send: uploadBatch, onResult, onChange: onState, onError });
  const stop = () => {
    activeGeneration++;
    stopCapture?.(); stopCapture = null;
    queue.stop();
  };
  return {
    queue,
    start() {
      stop();
      activity.reset();
      lastCheck = -Infinity; active = false;
      sessionId = crypto.randomUUID();
      const generation = activeGeneration;
      queue.start(sessionId);
      stopCapture = startFrameCapture(video, {
        onStall: (stalled) => {
          if (stalled) onError(new Error("영상 프레임 수신 중단 · 카메라와 탭 상태를 확인하세요."));
        },
        getActivity(now) {
          if (!filtering) return true;
          if (now - lastCheck < 90) return active;
          lastCheck = now;
          context.drawImage(video, 0, 0, 160, 90);
          const rgba = context.getImageData(0, 0, 160, 90).data;
          const gray = new Uint8Array(160 * 90);
          for (let i = 0; i < gray.length; i++) {
            gray[i] = Math.round((rgba[i * 4] + rgba[i * 4 + 1] + rgba[i * 4 + 2]) / 3);
          }
          active = activity.update(gray, now).active;
          return active;
        },
        onBatch(batch) {
          if (generation !== activeGeneration) return;
          queue.offer({ ...batch, session_id: sessionId, batch_id: `b-${batch.sequence}`,
            trigger: filtering ? "activity" : "continuous" });
        },
      });
    },
    stop,
    recalibrate() { activity.reset(); active = false; lastCheck = -Infinity; },
    setFiltering(value) { filtering = value; },
  };
}
