import test from "node:test";
import assert from "node:assert/strict";
import { ActivityDetector } from "../../web/activity.mjs";

test("empty scene, hand entry, stationary object and departure", () => {
  const detector = new ActivityDetector();
  const empty = new Uint8Array(100).fill(80);
  const object = empty.slice(); object.fill(180, 0, 20);
  assert.equal(detector.update(empty, 0).calibrated, true);
  assert.equal(detector.update(empty, 100).active, false);
  assert.equal(detector.update(object, 200).active, false);
  assert.equal(detector.update(object, 300).active, true);
  assert.equal(detector.update(object, 3000).active, true);
  detector.update(empty, 3100);
  detector.update(empty, 3200);
  assert.equal(detector.update(empty, 3600).active, false);
});

test("small noise is ignored; lighting changes can be recalibrated", () => {
  const detector = new ActivityDetector();
  const empty = new Uint8Array(100).fill(80);
  const noisy = empty.slice(); noisy[0] = 150;
  detector.update(empty, 0);
  detector.update(noisy, 100);
  assert.equal(detector.update(noisy, 200).active, false);
  const bright = new Uint8Array(100).fill(140);
  detector.update(bright, 300);
  assert.equal(detector.update(bright, 400).active, true);
  detector.reset();
  assert.equal(detector.update(bright, 500).active, false);
  assert.equal(detector.update(bright, 600).active, false);
});
