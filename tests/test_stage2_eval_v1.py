"""Metric semantics and native runtime policy are independent of model quality."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
from scripts.evaluate_stage2_v1 import (
    crop_attempts,
    geometry_stats,
    observe_decode,
    outcome_stats,
    paired_bootstrap,
    rate,
    verdict,
)
from wemeet.schemas import DetectedBarcode, GeometryField, RectifiedBarcode
from wemeet.sw import decoding, pipeline


def test_endpoint_error_and_linear_quantile():
    values = np.zeros((48, 2))
    values[0] = (3, 4)
    records = [
        {"geometry_valid": True, "error_xy_px": values.tolist(), "monotonicity_violations": 0}
    ]
    metrics = geometry_stats(records)["metrics"]
    assert metrics["epe_mean_px"] == pytest.approx(5 / 48)
    assert metrics["epe_rmse_px"] == pytest.approx(np.sqrt(25 / 48))
    assert np.quantile([0, 0, 0, 10], 0.95) == pytest.approx(8.5)
    records.append({"geometry_valid": False})
    assert geometry_stats(records)["metrics"] is None
    assert geometry_stats(records)["invalid_field_count"] == 1


def test_exact_text_and_symbology():
    assert verdict({"text": "001", "format": "Code128"}, "001") == "C"
    assert verdict({"text": "1", "format": "Code128"}, "001") == "W"
    assert verdict({"text": "001", "format": "EAN13"}, "001") == "W"
    assert verdict({"text": None}, "001") == "U"
    assert rate(0, 0)["pct"] is None
    assert rate(0, 1500)["wilson95_pct"][1] > 0


def test_h_includes_misreads_and_policy_keeps_them():
    rows = [
        {"d0_status": d, "paths": {"crop_policy_resnet": {"status": m}}, "group_id_proxy": str(i)}
        for i, (d, m) in enumerate([("C", "C"), ("W", "W"), ("U", "C"), ("U", "U")])
    ]
    stats = outcome_stats(rows, "crop_policy_resnet")
    assert stats["RR_H"]["numerator"] == 1
    assert stats["RR_H"]["denominator"] == 3
    assert stats["RR_U"]["denominator"] == 2
    assert stats["transitions"]["W->W"] == 1
    assert stats["C"] + stats["W"] + stats["U"] == 4
    identical = paired_bootstrap(rows, "crop_policy_resnet", "crop_policy_resnet", draws=100)
    assert identical["delta_RR_H_pp"] == 0


def test_wrong_first_retry_is_not_replaced_by_later_correct(monkeypatch):
    import scripts.evaluate_stage2_v1 as evaluation
    from wemeet.schemas import DecodeResult

    grid = np.stack(np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 3)), axis=-1).reshape(
        -1, 2
    )
    target = DetectedBarcode(np.zeros((16, 32, 3), np.uint8), 0, 1)
    calls = []

    def fake_decode(image):
        calls.append(1)
        return DecodeResult("wrong", 0), {
            "text": "wrong",
            "format": "Code128",
            "backend": "fake",
            "reason": None,
        }

    monkeypatch.setattr(evaluation, "observe_decode", fake_decode)
    outcome = crop_attempts(target, grid, grid)
    assert outcome["final"]["text"] == "wrong"
    assert len(calls) == 1


def test_observer_and_crop_route_match_native_pipeline(monkeypatch):
    root = Path("../data/barcode-datasets-a833a3a59441/synthetic/v1/eval/low")
    if not root.is_dir():
        pytest.skip("Local synthetic fixture not installed")
    import cv2

    for row in [json.loads(x) for x in (root / "manifest.jsonl").read_text().splitlines()[:3]]:
        gray = cv2.imread(str(root / row["file"]), 0)
        target = DetectedBarcode(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0, 1)
        with np.load(root / row["npz"]) as data:
            dst, src = data["dst_norm"], data["src_norm"]
        plain = RectifiedBarcode(gray, target, None)
        native = decoding.decode(plain)
        observed, details = observe_decode(plain)
        assert native.text == observed.text == details["text"]
        forced = crop_attempts(target, src, dst)
        monkeypatch.setattr(pipeline, "detect", lambda image: target)
        monkeypatch.setattr(
            pipeline, "estimate_geometry", lambda target: GeometryField(dst, src, "tps", 1)
        )
        actual = pipeline.run(target.crop_bgr_uint8)
        expected = native.text if native.text is not None else forced["final"]["text"]
        assert actual.decode.text == expected


def test_group_bootstrap_preserves_pairs_and_counts():
    from scripts.summarize_stage2_v1 import grouped_intervals

    rows = [
        {
            "group_id_proxy": "one-source",
            "d0_status": "U",
            "paths": {"crop_policy_resnet": {"status": "C"}},
        },
        {
            "group_id_proxy": "one-source",
            "d0_status": "U",
            "paths": {"crop_policy_resnet": {"status": "W"}},
        },
    ]
    summary = grouped_intervals(rows, ["crop_policy_resnet"], draws=100)
    assert summary["groups"] == 1
    assert summary["paths"]["crop_policy_resnet"]["RR_H"]["ci95_pct"] == [50, 50]


def test_retry_counts_retain_early_wrong_return():
    from scripts.summarize_stage2_v1 import retries

    rows = [
        {"paths": {"crop_policy_resnet": {"status": "W", "early_exit": True, "attempts": []}}},
        {
            "paths": {
                "crop_policy_resnet": {
                    "status": "C",
                    "early_exit": False,
                    "attempts": [{"status": "U"}, {"status": "C"}],
                }
            }
        },
    ]
    stats = retries(rows, "crop_policy_resnet")
    assert stats[0]["cumulative"] == {"C": 0, "W": 1, "U": 1}
    assert stats[0]["added_W"] == 0  # D0's wrong return was not introduced by retry.
    assert stats[1]["cumulative"] == {"C": 1, "W": 1, "U": 0}
    assert stats[2]["called_n"] == 0
