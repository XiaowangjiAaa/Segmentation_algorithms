# Segmentation Algorithms

This repository provides a minimal framework to train and evaluate various image segmentation models on datasets in the VOC2012 format. Models include:

- UNet
- DeepLabV3+
- segNext (requires the external `segnext` package)
- Swin Transformer
- SegFormer
- PVT (requires the external `pvt` package)

The training pipeline uses [Hugging Face Accelerate](https://github.com/huggingface/accelerate) for easy single or multi-GPU training and [Weights & Biases](https://wandb.ai/) for optional experiment tracking.

## Installation

Install the required packages:

```bash
pip install torch torchvision accelerate tqdm wandb segmentation-models-pytorch timm transformers
```

Additional packages are needed for segNext or PVT models.

## Training

```bash
python -m segmentation.train --data-dir /path/to/VOC2012 \
    --model unet --epochs 50 --batch-size 8 --wandb
```

This command trains UNet on the VOC2012 dataset. Replace `unet` with any available model name. When `--wandb` is specified, metrics will be logged to Weights & Biases.

## Inference

```bash
python -m segmentation.inference --model unet \
    --checkpoint checkpoints/model_49.pt \
    --image input.jpg --output pred.png
```

## Metrics

During validation the following metrics are computed:

- **mIoU** – mean intersection over union
- **Accuracy** – pixel accuracy
- **F1-score** – averaged F1-score over all classes

## Dataset

Training and validation loaders expect the standard VOC2012 directory structure. Download the dataset from [the official site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

