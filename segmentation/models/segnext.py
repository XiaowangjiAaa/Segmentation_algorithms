from typing import Any

try:
    import segnext
except ImportError:  # pragma: no cover - placeholder for segNext implementation
    segnext = None

from .simple import get_simple_model


def get_model(num_classes: int, version: str = "segnext-base", pretrained: bool = True) -> Any:
    """Return segNext model or a simple fallback implementation."""
    if segnext is None:
        # Use a very small segmentation model if the package is unavailable
        return get_simple_model(num_classes)
    return segnext.segmentation.segNext(
        num_classes=num_classes, version=version, pretrained=pretrained
    )
