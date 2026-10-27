"""Inference and decision rules without access to ground-truth labels."""

import hashlib
from collections import Counter
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from scripts.controlled_geometry.common import digest
from scripts.controlled_geometry.data import preprocess
from scripts.evaluate_stage2_v1 import field_from, observe_decode
from wemeet.ai.geometry.controlled import ControlledGeometry
from wemeet.ai.geometry.model import destination_grid
from wemeet.schemas import DetectedBarcode, RectifiedBarcode
from wemeet.sw import decoding
from wemeet.sw.rectify import apply_field

SCALES = (1.0, 1.5, 3.0)
POLICIES = ("early_exit", "always_rectify", "agreement")
NATIVE_DECODE = decoding.decode


@contextmanager
def decoder_mode(enhance):
    if decoding._zxingcpp is None:
        raise RuntimeError("Pinned zxing-cpp backend is missing")
    with (
        patch.object(decoding, "_pyzbar_decode", None),
        patch.object(decoding, "decode", partial(NATIVE_DECODE, enhance=enhance)),
    ):
        yield


def image_hash(gray):
    return hashlib.sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()


def decode_view(image, view):
    _, record = observe_decode(image)
    return {
        **record,
        "view": view,
        "image_sha256": image_hash(image.image_gray_uint8),
        "passthrough": image.field is None,
    }


def empty(reason="decode_failed"):
    return {"text": None, "format": None, "backend": None, "reason": reason}


def choose(policy, d0, warped):
    """Only observed strings, formats, and image identity enter acceptance."""
    if policy not in POLICIES:
        raise ValueError(policy)
    if policy == "early_exit" and d0["text"] is not None:
        return dict(d0)
    if policy in ("early_exit", "always_rectify"):
        return dict(next((x for x in warped if x["text"] is not None), empty()))
    observations = [x for x in [d0, *warped] if x["text"] is not None]
    candidates = {(x["text"], x["format"]) for x in observations}
    if len(candidates) > 1:
        return empty("conflicting_reads")
    # Duplicate pixels or a failed warp passed through as-is cannot cast extra votes.
    unique_images = {x["image_sha256"] for x in observations}
    if len(unique_images) < 2:
        return empty("insufficient_agreement")
    return dict(observations[0])


def plain(target):
    return RectifiedBarcode(cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY), target, None)


class Predictor:
    def __init__(self, entry, backbone, device="cuda:0"):
        path = Path(entry["path"])
        if digest(path) != entry["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["config"]["model"] != backbone:
            raise ValueError("Checkpoint backbone mismatch")
        self.model = ControlledGeometry(backbone)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(device).eval()
        self.device = device
        self.dst = destination_grid().numpy().astype(np.float64)

    def field(self, target):
        gray = cv2.cvtColor(target.crop_bgr_uint8, cv2.COLOR_BGR2GRAY)
        with torch.inference_mode():
            pred = self.model(preprocess(gray).unsqueeze(0).to(self.device))[0]
            pred = pred.cpu().numpy().astype(np.float64)
        if pred.shape != (48, 2) or not np.isfinite(pred).all():
            raise ValueError("Invalid geometry prediction")
        if pred.min() < -0.5 or pred.max() > 1.5:
            raise ValueError("Geometry outside fixed coordinate contract")
        return field_from(pred, self.dst)


def audit_crop(target, predictor):
    """Observe every fixed view; scoring is performed separately by the caller."""
    first = decode_view(plain(target), "d0")
    field = predictor.field(target)
    warped = [decode_view(apply_field(target, field, out_scale=s), f"warp-{s}") for s in SCALES]
    return {
        "d0": first,
        "warped": warped,
        "prediction": field.control_points_src_norm.tolist(),
        "decisions": {p: choose(p, first, warped) for p in POLICIES},
    }


def execute_crop(target, predictor, policy):
    """Execute only the steps required by a policy, including early returns."""
    d0 = empty()
    if policy != "always_rectify":
        d0 = decode_view(plain(target), "d0")
        if policy == "early_exit" and d0["text"] is not None:
            return d0
    field = predictor.field(target)
    warped = []
    for scale in SCALES:
        observation = decode_view(apply_field(target, field, out_scale=scale), f"warp-{scale}")
        warped.append(observation)
        if policy != "agreement" and observation["text"] is not None:
            break
    return choose(policy, d0, warped)


def target_from_gray(gray):
    return DetectedBarcode(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.0, 1.0)


def status(result, truth):
    if result["text"] is None:
        return "U"
    return "C" if result["text"] == truth and result["format"] == "Code128" else "W"


def counts(statuses):
    c = Counter(statuses)
    n = sum(c.values())
    return {
        "n": n,
        **{k: c[k] for k in ("C", "W", "U")},
        "EDR": 100 * c["C"] / n if n else None,
        "FDR": 100 * c["W"] / n if n else None,
        "NRR": 100 * c["U"] / n if n else None,
    }
