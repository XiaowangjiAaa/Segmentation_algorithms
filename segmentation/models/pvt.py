from typing import Any

try:
    import pvt
except ImportError:  # pragma: no cover - placeholder
    pvt = None


def get_model(num_classes: int, version: str = "pvt-tiny", pretrained: bool = True) -> Any:
    """Return PVT segmentation model."""
    if pvt is None:
        raise ImportError("pvt package is required for PVT model")
    model = pvt.PVTSegmentation(num_classes=num_classes, version=version, pretrained=pretrained)
    return model
