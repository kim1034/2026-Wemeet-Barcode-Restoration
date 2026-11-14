"""EfficientNet-B0-C3: ResNet18-C3 와 같은 head 에 백본만 EfficientNet-B0 로 바꾼 모델.

`wemeet/ai/geometry/model.py` 의 `ResNet18C3` 를 복사했다. 바뀐 것은 백본·adapter
두 줄뿐이고 head, output, `_initialize_identity`, `forward` 는 원본과 한 글자도 다르지
않다. `destination_grid`, `N_X`, `N_Y`, `N_POINTS` 는 원본에서 import 한다.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from wemeet.ai.geometry.model import N_POINTS, N_X, N_Y, destination_grid


class EfficientNetB0C3(nn.Module):
    """ImageNet EfficientNet-B0 ``features[:6]`` plus the ResNet18-C3 geometry head.

    Input is ``(B, 3, 160, 384)``. The ``features[:6]`` output is ``(B, 112, 10, 24)``.
    The output is ``(B, 48, 2)`` in the contract's row-major ``(x, y)`` order.
    The head starts at the identity grid and learns a bounded residual. This
    keeps predictions in the allowed ``[-0.5, 1.5]`` source-coordinate range.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights).features[:6]  # 바뀐 줄
        self.adapter = nn.Conv2d(112, 256, kernel_size=1)  # 바뀐 줄
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


def count_params(model: nn.Module) -> int:
    """가이드 6장의 정의: 백본 + adapter + head 전체."""
    return sum(p.numel() for p in model.parameters())


__all__ = ["EfficientNetB0C3", "count_params"]
