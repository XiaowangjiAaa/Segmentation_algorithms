import argparse
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch import nn, optim
from torchvision import transforms as T
from tqdm import tqdm

from .dataset import get_dataloader
from .models import create_model
from .utils import compute_metrics
from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation models")
    parser.add_argument("--data-dir", type=str, required=True, help="VOC2012 root directory")
    parser.add_argument("--model", type=str, default="unet", help="Model name")
    parser.add_argument("--version", type=str, default="", help="Backbone or model version")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--num-classes", type=int, default=21, help="Number of classes")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("training_results"))
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
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
    checkpoint_dir = args.output / "checkpoints"
    log_dir = args.output / "train_logs"
    accelerator = Accelerator()
    if args.wandb and accelerator.is_local_main_process:
        import wandb

        wandb.init(project="segmentation", config=vars(args))

    model = create_model(args.model, args.num_classes, version=args.version, pretrained=args.pretrained)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_dataloader(args.data_dir, "train", args.batch_size)
    val_loader = get_dataloader(args.data_dir, "val", args.batch_size)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    import csv
    import logging

    logging.basicConfig(
        filename=log_dir / "training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    metrics_file = log_dir / "metrics.csv"
    write_header = not metrics_file.exists()

    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        for imgs, masks in tqdm(
            train_loader, disable=not accelerator.is_local_main_process
        ):
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks.squeeze(1))
            accelerator.backward(loss)
            optimizer.step()
        if accelerator.is_local_main_process:
            metrics = evaluate(
                model, val_loader, args.num_classes, epoch, args.wandb
            )

            if write_header:
                with open(metrics_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["epoch", *metrics.keys()])
                    writer.writeheader()
                    writer.writerow({"epoch": epoch, **metrics})
                write_header = False
            else:
                with open(metrics_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["epoch", *metrics.keys()])
                    writer.writerow({"epoch": epoch, **metrics})

            logging.info("Epoch %d: %s", epoch, metrics)

            if metrics["miou"] > best_miou:
                best_miou = metrics["miou"]
                best_path = checkpoint_dir / "best.pt"
                torch.save(
                    accelerator.unwrap_model(model).state_dict(), best_path
                )

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                save_path = checkpoint_dir / f"model_{epoch}.pt"
                torch.save(
                    accelerator.unwrap_model(model).state_dict(), save_path
                )


def evaluate(
    model: Any, loader: Any, num_classes: int, epoch: int, use_wandb: bool
) -> dict:
    model.eval()
    metrics = {"miou": 0.0, "accuracy": 0.0, "f1": 0.0}
    count = 0
    with torch.no_grad():
        for imgs, masks in loader:
            preds = model(imgs)
            m = compute_metrics(preds, masks, num_classes)
            for k, v in m.items():
                metrics[k] += v
            count += 1
    for k in metrics:
        metrics[k] /= count
    print(f"Epoch {epoch}: {metrics}")
    if use_wandb:
        import wandb

        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=epoch)
    return metrics


if __name__ == "__main__":
    main()
