# CIFAR-10 Image Classification

> A reproducible CIFAR-10 image classification project using the public `torchvision ResNet-18` model.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-93.84%25-2ea44f)

## Overview

This project fine-tunes torchvision's public ResNet-18 with ImageNet pretrained weights
for the ten-class CIFAR-10 classification task. To support 32×32 images, the first
convolution is replaced with a 3×3 convolution with stride 1, and the initial max pooling
layer is removed. Training uses random cropping, horizontal flipping, label smoothing,
cosine annealing, and automatic mixed precision.

## Results

| Metric | Value |
| --- | ---: |
| Model | ResNet-18 |
| Dataset | CIFAR-10 |
| Training epochs | 20 |
| Best epoch | 16 |
| Best test accuracy | **93.84%** |
| GPU | NVIDIA RTX 3090 24GB |
| Training time | 244.04 seconds |

## Project Structure

```text
Classification_test/
├── configs/                 # YAML experiment configurations
├── data/                    # Dataset files (ignored by Git)
├── datasets/                # Data loading and augmentation
├── docs/                    # Experiment documentation
├── evaluation/              # Evaluation metrics
├── experiments/             # Metrics and model outputs
├── losses/                  # Loss functions
├── models/                  # Model definitions
├── scripts/                 # Training entry point
├── tools/                   # Configuration and reproducibility utilities
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python scripts/train.py --config configs/resnet18.yaml
```

The best checkpoint and per-epoch metrics are saved to:

```text
experiments/resnet18_cifar10/
├── best.pt
└── metrics.json
```

### 3. Run a Smoke Test

```bash
python scripts/train.py --config configs/resnet18.yaml --epochs 1
```

## Configuration

The main hyperparameters are defined in `configs/resnet18.yaml`, including batch size,
learning rate, training epochs, weight decay, label smoothing, and data and output paths.
Edit this configuration to run experiment variants without changing the training code.

## Model and Dataset Sources

- Model: [torchvision ResNet-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

See [`docs/RESULTS.md`](docs/RESULTS.md) for the experiment report. Dataset files and model
weights are excluded from Git to keep the repository lightweight and organized.

