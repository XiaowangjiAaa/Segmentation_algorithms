from typing import Any

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:  # pragma: no cover - placeholder
    SegformerForSemanticSegmentation = None


def get_model(num_classes: int) -> Any:
    """Return SegFormer model."""
    if SegformerForSemanticSegmentation is None:
        raise ImportError("transformers>=4.21 is required for SegFormer")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model
