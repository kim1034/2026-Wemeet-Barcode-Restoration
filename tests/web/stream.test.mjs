import test from "node:test";
import assert from "node:assert/strict";
import { StreamEvents, StreamGate } from "../../web/stream.mjs";

test("one frame is in flight at a time and frames are not sent faster than 8 per second", () => {
  const gate = new StreamGate({ minIntervalMs: 125 });
  assert.equal(gate.ready(0), true);
  gate.begin(0);
  assert.equal(gate.ready(500), false); // 응답 전에는 다음 프레임을 보내지 않는다 (최신 프레임만 처리)
  gate.end("ok");
  assert.equal(gate.ready(100), false);
  assert.equal(gate.ready(125), true);
  gate.begin(125); gate.end("busy");
  gate.begin(250); gate.end("error");
  assert.deepEqual([gate.sent, gate.busy, gate.failed], [3, 1, 1]);
  gate.reset();
  assert.deepEqual([gate.sent, gate.inflight], [0, false]);
});

test("each server event is shown once even if it arrives again", () => {
  const shown = [];
  const events = new StreamEvents((event, batch) => shown.push([event.event_id, batch.capturedAt]));
  events.accept([{ event_id: 1, track_start_ms: 100 }, { event_id: 2, track_start_ms: 900 }]);
  events.accept([{ event_id: 2, track_start_ms: 900 }]);
  events.accept(undefined);
  assert.deepEqual(shown, [[1, 100], [2, 900]]);
  events.reset();
  events.accept([{ event_id: 1, track_start_ms: 5 }]);
  assert.equal(shown.length, 3);
});
