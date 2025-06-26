from typing import Any

try:
    import pvt
except ImportError:  # pragma: no cover - placeholder
    pvt = None

from .simple import get_simple_model


def get_model(num_classes: int, version: str = "pvt-tiny", pretrained: bool = True) -> Any:
    """Return PVT segmentation model or a simple fallback implementation."""
    if pvt is None:
        return get_simple_model(num_classes)
    return pvt.PVTSegmentation(
        num_classes=num_classes, version=version, pretrained=pretrained
    )
