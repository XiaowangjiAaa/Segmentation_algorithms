from typing import Any

import segmentation_models_pytorch as smp


def get_model(num_classes: int) -> Any:
    """Return DeepLabV3+ model."""
    model = smp.DeepLabV3Plus(encoder_name="resnet50", classes=num_classes, activation=None)
    return model
