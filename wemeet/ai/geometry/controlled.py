"""Two backbones with the same spatial head for controlled comparison v2.

This is a new MobileNet candidate, not a reconstruction of the unavailable
historical MobileNet experiment. The production ResNet adapter stays unchanged.
"""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, resnet18

from wemeet.ai.geometry.model import N_POINTS, N_X, N_Y, destination_grid

BACKBONES = ("resnet18-c3", "mobilenetv3-small")


class ControlledGeometry(nn.Module):
    """OS16 features -> C-to-256 adapter -> shared head -> 48 source points."""

    def __init__(self, backbone: str, weights: str | Path | None = None):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.backbone_name = backbone
        base = (
            resnet18(weights=None) if backbone == BACKBONES[0] else mobilenet_v3_small(weights=None)
        )
        if weights is not None:
            base.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True), strict=True
            )
        self.backbone = (
            nn.Sequential(*list(base.children())[:7])
            if backbone == BACKBONES[0]
            else base.features[:9]
        )
        self.adapter = nn.Conv2d(256 if backbone == BACKBONES[0] else 48, 256, 1)
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((N_Y, N_X)),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(64, 2, 1)
        self.register_buffer(
            "identity_grid", destination_grid().reshape(N_Y, N_X, 2).unsqueeze(0), persistent=False
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image):
        raw = self.output(self.head(self.adapter(self.backbone(image))))
        raw = raw.permute(0, 2, 3, 1).contiguous()
        identity = self.identity_grid.to(device=raw.device, dtype=raw.dtype)
        return (identity + 0.5 * torch.tanh(raw)).reshape(-1, N_POINTS, 2)
