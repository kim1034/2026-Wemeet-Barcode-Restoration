// A successful representative frame is independent of the batch's verification status.
export function resultPresentation(event) {
  const batch = event.burst;
  if (batch) {
    const states = {
      confirmed: {
        label: "판독 성공 · 교차 확인 완료",
        note: "서로 다른 2개 프레임에서 같은 번호를 읽었습니다.",
        value: batch.text,
        confirmed: true,
      },
      provisional: {
        label: "판독 성공 · 교차 확인 부족",
        note: `${batch.frame_attempt_count}장 시도 중 1장에서 번호를 읽었습니다. 다른 프레임에서 같은 번호가 확인되지 않아 자동 확정은 보류합니다.`,
        value: batch.text,
        confirmed: false,
      },
      conflict: {
        label: "번호 충돌 · 확정 보류",
        note: "프레임마다 서로 다른 번호가 읽혔습니다. 어느 번호도 자동 확정하지 않았습니다.",
        value: null,
        confirmed: false,
      },
      unread: {
        label: "미판독",
        note: `${batch.frame_attempt_count}장 시도에서 읽힌 번호가 없습니다. 프레임별 실패 원인을 확인하세요.`,
        value: null,
        confirmed: false,
      },
    };
    const state = states[batch.status] || {
      label: "결과 확인 필요", note: "알 수 없는 응답 상태입니다.", value: null, confirmed: false,
    };
    return { ...state, hasValue: state.value != null, tone: state.confirmed ? "good" : "warn" };
  }
  const success = event.real?.ok ?? event.outcome === "success";
  return {
    label: success ? "판독 성공" : "미판독",
    note: success ? "사진 1장에서 번호를 읽었습니다." : "번호를 읽지 못했습니다. 처리 상세를 확인하세요.",
    value: success ? event.barcode : null,
    hasValue: success,
    confirmed: success,
    tone: success ? "good" : "warn",
  };
}
