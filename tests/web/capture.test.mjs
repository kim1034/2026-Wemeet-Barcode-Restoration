import test from "node:test";
import assert from "node:assert/strict";
import { FrameCollector } from "../../web/capture.mjs";

const flush = () => new Promise((resolve) => setImmediate(resolve));
const image = () => ({ blob: "jpeg", width: 32, height: 24 });

test("one second emits eight distinct frames with stable timestamps", async () => {
  const batches = [];
  const capture = new FrameCollector({ onBatch: (batch) => batches.push(batch) });
  for (let i = 0; i <= 8; i++) capture.push(i * 125, i, image, true);
  await flush();
  assert.equal(batches.length, 1);
  assert.equal(batches[0].frames.length, 8);
  assert.deepEqual(batches[0].frames.map((f) => f.offset_ms), [0, 125, 250, 375, 500, 625, 750, 875]);
});

test("frozen video never produces eight copies", async () => {
  const batches = [];
  const capture = new FrameCollector({ onBatch: (batch) => batches.push(batch) });
  for (let i = 0; i <= 8; i++) capture.push(i * 125, 0, image, true);
  await flush();
  assert.equal(batches[0].frames.length, 1);
});

test("JPEG completions can arrive out of order; failed captures stay missing", async () => {
  const batches = [], resolves = [];
  const capture = new FrameCollector({ onBatch: (batch) => batches.push(batch) });
  for (let i = 0; i < 8; i++) {
    capture.push(i * 125, i, () => new Promise((resolve) => resolves.push(resolve)), true);
  }
  capture.push(1000, 8, image, true);
  for (let i = 7; i >= 0; i--) resolves[i](i === 3 ? null : image());
  await flush();
  assert.deepEqual(batches[0].frames.map((f) => f.frame_id), ["f-1", "f-2", "f-3", "f-5", "f-6", "f-7", "f-8"]);
});

test("stop invalidates outstanding encoding and long stalls discard partial windows", async () => {
  const batches = [];
  const capture = new FrameCollector({ onBatch: (batch) => batches.push(batch) });
  capture.push(0, 0, image, true);
  capture.push(4000, 1, image, true);
  await flush();
  assert.equal(batches.length, 0);
  capture.push(5000, 2, image, true);
  capture.reset();
  await flush();
  assert.equal(batches.length, 0);
});
