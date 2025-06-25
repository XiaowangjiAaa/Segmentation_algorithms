from typing import Any

import segmentation_models_pytorch as smp


def get_model(num_classes: int, version: str = "resnet50", pretrained: bool = True) -> Any:
    """Return DeepLabV3+ model with configurable encoder."""
    weights = "imagenet" if pretrained else None
    model = smp.DeepLabV3Plus(encoder_name=version, classes=num_classes, activation=None, encoder_weights=weights)
    return model
