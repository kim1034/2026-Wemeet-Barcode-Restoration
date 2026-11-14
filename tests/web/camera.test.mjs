import test from "node:test";
import assert from "node:assert/strict";
import { cameraKind, cameraOptions, loadCamera, preferredCamera, saveCamera } from "../../web/camera.mjs";

test("built-in and USB cameras are told apart by name", () => {
  assert.equal(cameraKind("Intel(R) AVStream Camera"), "builtin");
  // 노트북 내장 카메라도 속은 USB 라 번호가 붙는다 — 이름의 "Integrated" 를 먼저 본다
  assert.equal(cameraKind("Integrated Camera (04f2:b6dd)"), "builtin");
  assert.equal(cameraKind("Logitech BRIO (046d:085e)"), "external");
  assert.equal(cameraKind("USB2.0 HD UVC WebCam"), "external");
  assert.equal(cameraKind("OBS Virtual Camera"), "virtual");
  assert.equal(cameraKind("HD Camera"), "unknown");
  assert.equal(cameraKind(undefined), "unknown");
});

test("options keep only cameras and show a tidy name with the guessed kind", () => {
  const options = cameraOptions([
    { kind: "audioinput", deviceId: "mic", label: "Microphone" },
    { kind: "videoinput", deviceId: "a", label: "Intel(R) AVStream Camera" },
    { kind: "videoinput", deviceId: "b", label: "Logitech BRIO (046d:085e)" },
    { kind: "videoinput", deviceId: "c", label: "HD Camera" },
    { kind: "videoinput", deviceId: "", label: "" }, // 권한 전
  ]);
  assert.deepEqual(options.map((o) => o.text), [
    "Intel(R) AVStream Camera · 내장",
    "Logitech BRIO · 외장(USB)",
    "HD Camera",
    "카메라 4",
  ]);
});

test("the last chosen camera is used only while it is still plugged in", () => {
  const plugged = [{ deviceId: "a" }, { deviceId: "b" }];
  assert.equal(preferredCamera(plugged, "b"), "b");
  assert.equal(preferredCamera(plugged, "gone"), null); // 뽑혔으면 브라우저 기본 카메라
  assert.equal(preferredCamera(plugged, null), null);
  // 권한 전에는 목록 ID 를 못 읽으므로 저장값으로 일단 시도한다
  assert.equal(preferredCamera([{ deviceId: "" }], "b"), "b");
  assert.equal(preferredCamera([], "b"), "b");
});

test("the choice is remembered, and a blocked storage never breaks the camera", () => {
  const data = new Map();
  const storage = {
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => data.set(key, value),
    removeItem: (key) => data.delete(key),
  };
  assert.equal(loadCamera(storage), null);
  saveCamera(storage, "b");
  assert.equal(loadCamera(storage), "b");
  saveCamera(storage, "");
  assert.equal(loadCamera(storage), null);

  const blocked = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } };
  assert.equal(loadCamera(blocked), null);
  assert.doesNotThrow(() => saveCamera(blocked, "b"));
  assert.equal(loadCamera(null), null);
  assert.doesNotThrow(() => saveCamera(null, "b"));
});
