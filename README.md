# Segmentation Algorithms

This repository provides a minimal framework to train and evaluate various image segmentation models on datasets in the VOC2012 format. Models include:

- UNet
- DeepLabV3+
- segNext (uses a simple fallback if the `segnext` package is unavailable)
- Swin Transformer
- SegFormer (falls back to a lightweight implementation when the `transformers` package is missing)
- PVT (uses a basic model if the `pvt` package is unavailable)

The training pipeline uses [Hugging Face Accelerate](https://github.com/huggingface/accelerate) for easy single or multi-GPU training and [Weights & Biases](https://wandb.ai/) for optional experiment tracking.

## Installation

Install the required packages:

```bash
pip install torch torchvision accelerate tqdm wandb segmentation-models-pytorch timm transformers pyyaml
```

Additional packages are needed for segNext or PVT models.

## Configuration

Sample configuration files for each model are stored in `segmentation/configs/`.
Use these as starting points and modify as needed. Each configuration includes
an `output` path specifying where checkpoints for that model will be saved.

## Training

```bash
python -m segmentation.train --data-dir /path/to/VOC2012 \
    --config segmentation/configs/unet.yaml --wandb
```

Each model has a sample YAML configuration under `segmentation/configs/`. These files define the model variant, whether to use pretrained weights and other hyperparameters. Command line arguments override values from the config file.

## Inference

```bash
python -m segmentation.inference --config segmentation/configs/unet.yaml \
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

