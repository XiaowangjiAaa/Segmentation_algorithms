from typing import Any

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:  # pragma: no cover - placeholder
    SegformerForSemanticSegmentation = None

from .simple import get_simple_model


def get_model(num_classes: int, version: str = "nvidia/segformer-b0-finetuned-ade-512-512", pretrained: bool = True) -> Any:
    """Return SegFormer model or a simple fallback implementation."""
    if SegformerForSemanticSegmentation is None:
        return get_simple_model(num_classes)
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            version,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    return SegformerForSemanticSegmentation.from_pretrained(
        version,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
