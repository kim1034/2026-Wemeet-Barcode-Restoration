"""scripts/run_benchmark.py 의 ladder() — run() 결과 하나를 B·C·D 조건으로 가르는 규칙."""

import numpy as np

from scripts.run_benchmark import ladder
from wemeet.schemas import (
    DecodeResult,
    DetectedBarcode,
    GeometryField,
    PipelineResult,
    RectifiedBarcode,
)

_IMG = np.zeros((4, 8, 3), np.uint8)
_DET = DetectedBarcode(crop_bgr_uint8=_IMG, angle_deg_ccw=0.0, confidence=0.9)
_GRID = np.array([[x, y] for y in (0.0, 1.0) for x in (0.0, 1.0)])
_FIELD = GeometryField(_GRID, _GRID.copy(), method="tps", confidence=1.0)


def _result(text, retry_count=0, field=None, detected=True) -> PipelineResult:
    rectified = RectifiedBarcode(np.zeros((4, 8), np.uint8), _DET, field) if detected else None
    reason = None if text is not None else ("decode_failed" if detected else "not_detected")
    return PipelineResult(
        _IMG,
        _DET if detected else None,
        rectified,
        DecodeResult(text=text, retry_count=retry_count, failure_reason=reason),
    )


def test_first_decode_success_counts_for_every_condition():
    assert ladder(_result("X")) == {"B": "X", "C": "X", "D": "X"}


def test_first_rectify_attempt_counts_from_C():
    assert ladder(_result("X", retry_count=0, field=_FIELD)) == {"B": None, "C": "X", "D": "X"}


def test_retry_success_counts_only_for_D():
    assert ladder(_result("X", retry_count=2, field=_FIELD)) == {"B": None, "C": None, "D": "X"}


def test_not_detected_fails_everywhere():
    assert ladder(_result(None, detected=False)) == {"B": None, "C": None, "D": None}


def test_degraded_first_decode_failure_fails_everywhere():
    assert ladder(_result(None)) == {"B": None, "C": None, "D": None}
