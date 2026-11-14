import test from "node:test";
import assert from "node:assert/strict";
import { LatestBatchQueue } from "../../web/continuous.mjs";

const flush = () => new Promise((resolve) => setImmediate(resolve));
const batch = (sequence, session_id = "s", active = true) =>
  ({ sequence, batch_id: `b-${sequence}`, session_id, active, capturedAt: 0 });
const response = (value) => ({ session_id: value.session_id, batch_id: value.batch_id });

test("browser timers are not invoked with the queue as their receiver", () => {
  const original = globalThis.clearTimeout;
  try {
    globalThis.clearTimeout = function () { assert.equal(this, undefined); };
    const queue = new LatestBatchQueue({ send: async () => ({}) });
    queue.start("s");
    queue.stop();
  } finally { globalThis.clearTimeout = original; }
});

test("slow inference preserves capture, bounded latest queue, and ignores empty windows", async () => {
  const sent = [], resolves = [], received = [];
  const queue = new LatestBatchQueue({ now: () => 0, onResult: (r) => received.push(r),
    send: (b) => { sent.push(b); return new Promise((resolve) => resolves.push(resolve)); } });
  queue.start("s");
  queue.offer(batch(1));
  queue.offer(batch(2));
  queue.offer(batch(3));
  queue.offer(batch(4, "s", false));
  assert.equal(sent.length, 1);
  assert.equal(queue.pending.sequence, 3);
  assert.equal(queue.skipped, 1);
  resolves[0](response(sent[0]));
  await flush();
  assert.deepEqual(sent.map((b) => b.sequence), [1, 3]);
  resolves[1](response(sent[1]));
  await flush();
  assert.equal(received.length, 2);
  assert.equal(queue.busy, false);
});

test("restart ignores previous responses and never overlaps HTTP requests", async () => {
  const sent = [], resolves = [], received = [];
  const queue = new LatestBatchQueue({ now: () => 0, onResult: (r) => received.push(r),
    send: (b) => { sent.push(b); return new Promise((resolve) => resolves.push(resolve)); } });
  queue.start("old"); queue.offer(batch(1, "old"));
  queue.stop(); queue.start("new"); queue.offer(batch(1, "new"));
  assert.equal(sent.length, 1);
  resolves[0](response(sent[0]));
  await flush();
  assert.equal(received.length, 0);
  assert.equal(sent.length, 2);
  resolves[1](response(sent[1]));
  await flush();
  assert.equal(received[0].session_id, "new");
});

test("busy response backs off and sends the newest activity window", async () => {
  let clock = 0;
  const sent = [], timers = [];
  const queue = new LatestBatchQueue({ now: () => clock,
    schedule: (fn) => timers.push(fn), cancel: () => {},
    send: async (b) => {
      sent.push(b);
      if (sent.length === 1) throw Object.assign(new Error("busy"), { retryable: true });
      return response(b);
    } });
  queue.start("s"); queue.offer(batch(1));
  await flush();
  queue.offer(batch(2));
  assert.equal(sent.length, 1);
  clock = 2000; timers[0]();
  await flush();
  assert.deepEqual(sent.map((b) => b.sequence), [1, 2]);
});

test("expired and late-encoded windows are counted as skipped", async () => {
  const queue = new LatestBatchQueue({ now: () => 16000, send: () => assert.fail("expired") });
  queue.start("s"); queue.offer(batch(2)); queue.offer(batch(1));
  await flush();
  assert.equal(queue.skipped, 2);
});

test("mismatched response is never displayed", async () => {
  const errors = [];
  const queue = new LatestBatchQueue({ now: () => 0, onResult: () => assert.fail("wrong session"),
    onError: (e) => errors.push(e), send: async () => response(batch(99)) });
  queue.start("s"); queue.offer(batch(1));
  await flush();
  assert.equal(errors.length, 1);
  assert.equal(queue.skipped, 1);
});
