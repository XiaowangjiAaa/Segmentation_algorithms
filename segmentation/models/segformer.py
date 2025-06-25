from typing import Any

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:  # pragma: no cover - placeholder
    SegformerForSemanticSegmentation = None


def get_model(num_classes: int, version: str = "nvidia/segformer-b0-finetuned-ade-512-512", pretrained: bool = True) -> Any:
    """Return SegFormer model."""
    if SegformerForSemanticSegmentation is None:
        raise ImportError("transformers>=4.21 is required for SegFormer")
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            version,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            version,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    return model
