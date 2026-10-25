import test from "node:test";
import assert from "node:assert/strict";
import { resultPresentation } from "../../web/result-state.mjs";

const event = (status, text = null) => ({
  outcome: status === "confirmed" ? "success" : "review",
  barcode: text,
  real: { ok: true, text: "8809824677151" }, // Representative succeeded, even for a conflict.
  burst: { status, text, frame_attempt_count: 3 },
});

test("reported regression: one successful frame is readable but not cross-confirmed", () => {
  const value = resultPresentation(event("provisional", "8809824677151"));
  assert.equal(value.label, "판독 성공 · 교차 확인 부족");
  assert.equal(value.value, "8809824677151");
  assert.equal(value.hasValue, true);
  assert.equal(value.confirmed, false);
  assert.match(value.note, /3장 시도 중 1장/);
  assert.doesNotMatch(value.note, /보정 후|읽지 못/);
});

test("two agreeing frames are confirmed without losing leading zeroes", () => {
  const value = resultPresentation(event("confirmed", "0012345678905"));
  assert.equal(value.label, "판독 성공 · 교차 확인 완료");
  assert.equal(value.value, "0012345678905");
  assert.equal(value.confirmed, true);
});

test("a successful representative cannot turn conflicting observations into success", () => {
  const value = resultPresentation(event("conflict"));
  assert.equal(value.label, "번호 충돌 · 확정 보류");
  assert.equal(value.value, null);
  assert.equal(value.hasValue, false);
  assert.equal(value.confirmed, false);
});

test("unread and unknown batch states never inherit representative success", () => {
  assert.equal(resultPresentation(event("unread")).label, "미판독");
  const unknown = resultPresentation(event("unexpected"));
  assert.equal(unknown.confirmed, false);
  assert.equal(unknown.hasValue, false);
});

test("the single-photo path does not claim cross-frame verification", () => {
  const value = resultPresentation({ outcome: "success", barcode: "0123", real: { ok: true } });
  assert.equal(value.label, "판독 성공");
  assert.equal(value.value, "0123");
  assert.doesNotMatch(value.label, /교차 확인/);
  const unread = resultPresentation({ outcome: "review", real: { ok: false } });
  assert.equal(unread.label, "미판독");
});
