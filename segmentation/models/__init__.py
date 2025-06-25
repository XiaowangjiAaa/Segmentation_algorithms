from typing import Any

from . import deeplab, pvt, segformer, segnext, swin_transformer, unet


MODEL_REGISTRY = {
    "unet": unet.get_model,
    "deeplab": deeplab.get_model,
    "segnext": segnext.get_model,
    "swin_transformer": swin_transformer.get_model,
    "segformer": segformer.get_model,
    "pvt": pvt.get_model,
}


def create_model(name: str, num_classes: int) -> Any:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    return MODEL_REGISTRY[name](num_classes)
