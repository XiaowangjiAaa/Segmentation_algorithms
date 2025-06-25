from typing import Any

try:
    import timm
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - placeholder
    timm = None
    nn = None


class SwinSegmentation(nn.Module):
    def __init__(self, num_classes: int, version: str, pretrained: bool):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for Swin Transformer model")
        self.backbone = timm.create_model(version, pretrained=pretrained, features_only=True)
        self.head = nn.Conv2d(self.backbone.feature_info.channels()[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[-1]
        out = self.head(feats)
        out = torch.nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


def get_model(num_classes: int, version: str = "swin_tiny_patch4_window7_224", pretrained: bool = True) -> Any:
    return SwinSegmentation(num_classes, version, pretrained)
