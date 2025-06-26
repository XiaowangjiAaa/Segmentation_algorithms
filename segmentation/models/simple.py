from typing import Any
import torch
from torch import nn
import torch.nn.functional as F


class SimpleSegmentationModel(nn.Module):
    """Minimal segmentation model used as a fallback implementation."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.enc1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.enc1(x))
        x = self.pool(x)
        x = torch.relu(self.enc2(x))
        x = self.pool(x)
        x = torch.relu(self.dec1(x))
        x = self.up(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x


def get_simple_model(num_classes: int) -> Any:
    """Return the fallback segmentation model."""
    return SimpleSegmentationModel(num_classes)
