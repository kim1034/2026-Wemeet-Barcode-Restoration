// A one-second window contains at most eight DISTINCT presented video frames.
export class FrameCollector {
  constructor({ onBatch, windowMs = 1000, maxFrames = 8 }) {
    Object.assign(this, { onBatch, windowMs, maxFrames });
    this.generation = 0;
    this.sequence = 0;
    this.frameSequence = 0;
    this.reset();
  }
  reset() {
    this.generation++;
    this.start = null;
    this.lastSample = -Infinity;
    this.lastToken = null;
    this.frames = [];
    this.active = false;
  }
  push(now, token, capture, active) {
    if (this.start === null) this.start = now;
    if (now - this.start >= this.windowMs) {
      // A stalled/background tab must never join observations seconds apart.
      if (now - this.start < this.windowMs + 500) this.finish();
      this.start = now;
      this.frames = [];
      this.active = false;
      this.lastSample = -Infinity;
    }
    this.active ||= active;
    if (token === this.lastToken || now - this.lastSample < this.windowMs / this.maxFrames
        || this.frames.length >= this.maxFrames) return;
    this.lastToken = token;
    this.lastSample = now;
    const info = { frame_id: `f-${++this.frameSequence}`, offset_ms: now - this.start };
    // capture() snapshots the canvas now; later JPEG completion cannot reorder IDs.
    this.frames.push(Promise.resolve(capture()).then(
      (image) => image ? { ...info, ...image } : null,
      () => null,
    ));
  }
  finish() {
    const generation = this.generation;
    const batch = { sequence: ++this.sequence, capturedAt: this.start,
      active: this.active, window_ms: this.windowMs };
    Promise.all(this.frames).then((frames) => {
      if (generation !== this.generation) return;
      batch.frames = frames.filter(Boolean);
      if (batch.frames.length) this.onBatch(batch);
    });
  }
}

export function startFrameCapture(video, { onBatch, getActivity, onStall = () => {} }) {
  const collector = new FrameCollector({ onBatch });
  let stopped = false, callbackId, lastFrameAt = performance.now(), inflight = 0;
  let stalled = false;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const encode = () => {
    if (inflight >= 2 || !video.videoWidth || !video.videoHeight) return null;
    const scale = Math.min(1, 1920 / video.videoWidth, 1080 / video.videoHeight);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const width = canvas.width, height = canvas.height;
    inflight++;
    return new Promise((resolve) => {
      let settled = false;
      const done = (blob) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        inflight--;
        resolve(blob?.type === "image/jpeg" ? { blob, width, height } : null);
      };
      const timer = setTimeout(() => done(null), 2000);
      try { canvas.toBlob(done, "image/jpeg", 0.9); } catch { done(null); }
    });
  };
  const tick = (now, metadata) => {
    if (stopped) return;
    const token = metadata?.mediaTime ?? video.currentTime;
    if (token !== collector.lastToken) {
      lastFrameAt = now;
      if (stalled) { stalled = false; onStall(false); }
    }
    if (!video.paused && !document.hidden && video.readyState >= 2) {
      collector.push(now, token, encode, getActivity(now));
    }
    callbackId = video.requestVideoFrameCallback
      ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
  };
  callbackId = video.requestVideoFrameCallback
    ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
  const watchdog = setInterval(() => {
    if (performance.now() - lastFrameAt > 2000 && !stalled) {
      stalled = true;
      collector.reset();
      onStall(true);
    }
  }, 1000);
  return () => {
    stopped = true;
    collector.reset();
    clearInterval(watchdog);
    if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(callbackId);
    else cancelAnimationFrame(callbackId);
  };
}
