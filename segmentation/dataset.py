from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class VOCDataset(Dataset):
    """Minimal Pascal VOC2012 dataset loader."""

    def __init__(
        self,
        root: str,
        split: str,
        transform: T.Compose,
        target_transform: T.Compose,
    ) -> None:
        self.root = Path(root)
        assert split in {"train", "val", "trainval"}
        split_file = (
            self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        )
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        self.images = [self.root / "JPEGImages" / f"{id}.jpg" for id in ids]
        self.masks = [self.root / "SegmentationClass" / f"{id}.png" for id in ids]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.images)

    def __getitem__(self, idx: int):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        return img, mask


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
    dataset = VOCDataset(
        root=data_dir,
        split=split,
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
