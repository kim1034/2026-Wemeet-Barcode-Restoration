"""Stage 2 geometry model used by the ResNet18-C3 experiment.

The model predicts only normalized source control points. It never creates an
image; OpenCV/TPS in Stage 3 consumes the returned coordinates.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

N_X, N_Y = 16, 3
N_POINTS = N_X * N_Y


def destination_grid(device: torch.device | None = None) -> torch.Tensor:
    """Return the fixed row-major ``(48, 2)`` normalized destination grid."""
    ys = torch.linspace(0.0, 1.0, N_Y, device=device)
    xs = torch.linspace(0.0, 1.0, N_X, device=device)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    return grid.reshape(-1, 2)[:, [1, 0]]


class ResNet18C3(nn.Module):
    """ImageNet ResNet18 through layer3 plus a spatial geometry head.

    Input is ``(B, 3, 160, 384)``. The layer3 feature is ``(B, 256, 10, 24)``.
    The output is ``(B, 48, 2)`` in the contract's row-major ``(x, y)`` order.
    The head starts at the identity grid and learns a bounded residual. This
    keeps predictions in the allowed ``[-0.5, 1.5]`` source-coordinate range.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:7])
        self.adapter = nn.Conv2d(256, 256, kernel_size=1)
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((N_Y, N_X)),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(64, 2, kernel_size=1)
        self.register_buffer(
            "identity_grid",
            destination_grid().reshape(N_Y, N_X, 2).unsqueeze(0),
            persistent=False,
        )
        self._initialize_identity()

    def _initialize_identity(self) -> None:
        """Start the residual head at zero, yielding the identity field."""
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.adapter(features)
        features = self.head(features)
        raw = self.output(features).permute(0, 2, 3, 1).contiguous()
        identity = self.identity_grid.to(device=raw.device, dtype=raw.dtype)
        return (identity + 0.5 * torch.tanh(raw)).reshape(-1, N_POINTS, 2)


__all__ = ["N_POINTS", "N_X", "N_Y", "ResNet18C3", "destination_grid"]
