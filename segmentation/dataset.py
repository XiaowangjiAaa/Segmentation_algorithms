import os
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import VOCSegmentation


def get_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (512, 512),
) -> DataLoader:
    """Return dataloader for VOC2012 dataset."""
    assert split in {"train", "val", "trainval"}
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])
    target_transform = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor(),
    ])
    dataset = VOCSegmentation(
        root=data_dir,
        year="2012",
        image_set=split,
        download=False,
        transform=transform,
        target_transform=target_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
