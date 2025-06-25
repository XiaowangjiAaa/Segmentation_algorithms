import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from .models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for segmentation")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = create_model(args.model, args.num_classes)
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
