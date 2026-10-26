import test from "node:test";
import assert from "node:assert/strict";
import { FrameCollector } from "../../web/capture.mjs";
import { TrackQueue, TrackResults } from "../../web/tracking.mjs";

const flush = () => new Promise((resolve) => setImmediate(resolve));
const chunk = (sequence, session_id = "s") => ({ sequence, session_id, capturedAt: 0 });

test("tracking transport emits half-second chunks independently of passage duration", async () => {
  const received = [];
  const collector = new FrameCollector({ windowMs: 500, maxFrames: 4, onBatch: (x) => received.push(x) });
  for (let i = 0; i <= 12; i++) collector.push(i * 125, i, () => ({ blob: "jpg" }), true);
  await flush();
  assert.equal(received.length, 3);
  assert.deepEqual(received.map((x) => x.frames.length), [4, 4, 4]);
});

test("slow uploads are bounded and never overlap, including a restarted camera", async () => {
  const sent = [], resolve = [], received = [];
  const queue = new TrackQueue({ now: () => 0, send: (x) => {
    sent.push(x); return new Promise((done) => resolve.push(done));
  }, onReply: (x) => received.push(x) });
  queue.start("s");
  for (let i = 0; i < 6; i++) queue.offer(chunk(i));
  assert.equal(sent.length, 1);
  assert.equal(queue.pending.length, 2);
  assert.equal(queue.skipped, 3);
  queue.start("new"); queue.offer(chunk(0, "new"));
  resolve[0](chunk(0)); await flush();
  assert.equal(received.length, 0);
  assert.equal(sent.length, 2);
  resolve[1](chunk(0, "new")); await flush();
  assert.equal(received.length, 1);
  queue.stop();
});

test("transient retry keeps sequence and mismatched replies are rejected", async () => {
  const errors = [], timers = [], sent = [];
  const queue = new TrackQueue({ now: () => 0, schedule: (fn) => timers.push(fn), cancel: () => {},
    onError: (e) => errors.push(e), onReply: () => assert.fail("mismatch"), send: async (x) => {
      sent.push(x.sequence);
      if (sent.length === 1) throw Object.assign(new Error("busy"), { retryable: true });
      return chunk(99);
    } });
  queue.start("s"); queue.offer(chunk(1)); await flush();
  timers[0](); await flush();
  assert.deepEqual(sent, [1, 1]);
  assert.equal(errors.length, 2);
  queue.stop();
});

test("one result per track; the same barcode later is a new item", () => {
  const received = [];
  const results = new TrackResults((x) => received.push(x));
  const a = { track_id: "t-1", text: "00123", result_sequence: 1 };
  const b = { track_id: "t-2", text: "00123", result_sequence: 2 };
  results.accept([a]); results.accept([a, b]); results.accept([b]);
  assert.deepEqual(received.map((x) => x.track_id), ["t-1", "t-2"]);
  assert.equal(results.cursor, 2);
});

test("unread overload results are displayed once, not promoted to success", () => {
  const received = [];
  const results = new TrackResults((x) => received.push(x));
  const failure = { track_id: "t-3", status: "unread", dropped: true, text: null };
  results.accept([failure]); results.accept([failure]);
  assert.equal(received.length, 1);
  assert.equal(received[0].status, "unread");
});
