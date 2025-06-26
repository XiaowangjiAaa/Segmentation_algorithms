import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from .models import create_model
from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for segmentation")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--version", type=str, default="", help="Backbone or model version")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if isinstance(args.output, str):
        args.output = Path(args.output)
    if isinstance(args.checkpoint, str):
        args.checkpoint = Path(args.checkpoint)
    if isinstance(args.image, str):
        args.image = Path(args.image)
    model = create_model(args.model, args.num_classes, version=args.version, pretrained=args.pretrained)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    img = transform(Image.open(args.image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        pred = model(img)
    mask = pred.argmax(dim=1)[0].byte().cpu()
    mask = T.functional.to_pil_image(mask)
    mask.save(args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
