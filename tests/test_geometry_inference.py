"""Tests for the lazy ResNet18-C3 geometry inference adapter."""

from __future__ import annotations

import numpy as np
import pytest

from wemeet.ai.geometry import clear_model_cache, estimate_geometry, resolve_weights
from wemeet.schemas import DetectedBarcode


def test_resolve_weights_prefers_explicit_local_file(tmp_path) -> None:
    checkpoint = tmp_path / "model_state_dict.pt"
    checkpoint.write_bytes(b"placeholder")

    assert resolve_weights(checkpoint) == checkpoint.resolve()


def test_estimate_geometry_loads_local_state_dict(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from wemeet.ai.geometry.model import ResNet18C3, destination_grid

    checkpoint = tmp_path / "model_state_dict.pt"
    model = ResNet18C3(pretrained=False)
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setenv("WEMEET_GEOMETRY_WEIGHTS", str(checkpoint))
    monkeypatch.setenv("WEMEET_GEOMETRY_DEVICE", "cpu")
    clear_model_cache()

    target = DetectedBarcode(np.zeros((120, 300, 3), dtype=np.uint8), 0.0, 0.9)
    field = estimate_geometry(target)

    expected = destination_grid().numpy()
    assert field.method == "tps"
    assert field.confidence == 1.0
    assert field.control_points_dst_norm.shape == (48, 2)
    assert field.control_points_src_norm.shape == (48, 2)
    # A freshly initialized head is the identity field.
    np.testing.assert_allclose(field.control_points_dst_norm, expected)
    np.testing.assert_allclose(field.control_points_src_norm, expected, atol=1e-6)
    clear_model_cache()
