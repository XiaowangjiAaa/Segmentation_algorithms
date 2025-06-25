from typing import Any

import segmentation_models_pytorch as smp


def get_model(num_classes: int, version: str = "resnet34", pretrained: bool = True) -> Any:
    """Return UNet model with configurable encoder."""
    weights = "imagenet" if pretrained else None
    model = smp.Unet(encoder_name=version, classes=num_classes, activation=None, encoder_weights=weights)
    return model
