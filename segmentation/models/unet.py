from typing import Any

import segmentation_models_pytorch as smp


def get_model(num_classes: int) -> Any:
    """Return UNet model."""
    model = smp.Unet(encoder_name="resnet34", classes=num_classes, activation=None)
    return model
