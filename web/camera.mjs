// 카메라 고르기 — 노트북 내장 카메라와 USB·USB-C 외장 카메라를 목록에서 고르고 마지막 선택을 기억한다.
//
// 브라우저는 카메라가 어떻게 연결됐는지 알려 주지 않는다. 그래서 내장/외장은 카메라 이름으로
// **추정**한다. 크롬·엣지는 USB 카메라 이름 뒤에 "(제조사:제품)" 번호를 붙이는데, 노트북 내장
// 카메라도 속은 USB 인 경우가 많아 이름의 "Integrated" 같은 단어를 먼저 본다.

const USB_ID = /\s*\([0-9a-f]{4}:[0-9a-f]{4}\)\s*$/i;
const BUILTIN = /integrated|built-?in|internal|facetime|avstream|user.?facing|front|내장/i;
const VIRTUAL = /virtual|가상/i;
const KIND_TEXT = { builtin: "내장", external: "외장(USB)", virtual: "가상" };
const STORAGE_KEY = "wemeet.camera";

/** "builtin" | "external" | "virtual" | "unknown" */
export function cameraKind(label) {
  const name = String(label || "");
  if (VIRTUAL.test(name)) return "virtual"; // OBS 가상 카메라 등 (IP 카메라를 웹캠처럼 쓸 때)
  if (BUILTIN.test(name)) return "builtin";
  if (USB_ID.test(name) || /usb/i.test(name)) return "external";
  return "unknown";
}

/** enumerateDevices() 결과를 화면 목록용으로 바꾼다. 권한 전에는 이름과 ID 가 비어 있다. */
export function cameraOptions(devices) {
  return (devices || [])
    .filter((device) => device.kind === "videoinput")
    .map((device, index) => {
      const kind = cameraKind(device.label);
      const name = String(device.label || "").replace(USB_ID, "").trim() || `카메라 ${index + 1}`;
      return {
        deviceId: device.deviceId,
        kind,
        text: KIND_TEXT[kind] ? `${name} · ${KIND_TEXT[kind]}` : name,
      };
    });
}

/**
 * 시작할 카메라 ID. 저장된 카메라가 목록에 없으면(뽑혔으면) null = 브라우저 기본 카메라.
 * 권한 전이라 목록 ID 를 아직 못 읽었으면 저장값으로 일단 시도한다 (실패하면 호출 쪽이 기본으로 다시 연다).
 */
export function preferredCamera(options, saved) {
  if (!saved) return null;
  if (!options.some((option) => option.deviceId)) return saved;
  return options.some((option) => option.deviceId === saved) ? saved : null;
}

// 저장소는 사생활 보호 모드 등에서 막힐 수 있다. 실패해도 카메라 사용에는 지장이 없다.
export function loadCamera(storage) {
  try {
    return storage.getItem(STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

export function saveCamera(storage, deviceId) {
  try {
    if (deviceId) storage.setItem(STORAGE_KEY, deviceId);
    else storage.removeItem(STORAGE_KEY);
  } catch {
    /* 기억하지 못해도 이번 선택은 그대로 쓴다 */
  }
}
