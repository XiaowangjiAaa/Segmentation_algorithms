from typing import Any

try:
    import segnext
except ImportError:  # pragma: no cover - placeholder for segNext implementation
    segnext = None


def get_model(num_classes: int) -> Any:
    """Return segNext model.

    This is a placeholder that requires the segnext package.
    """
    if segnext is None:
        raise ImportError("segnext package is required for segNext model")
    model = segnext.segmentation.segNext(num_classes=num_classes)
    return model
