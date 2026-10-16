# CIFAR-10 Image Classification

> A reproducible CIFAR-10 image classification project using the public `torchvision ResNet-18` model.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![Protocol](https://img.shields.io/badge/Evaluation-Train_Val_Test-2ea44f)

## Overview

This project fine-tunes torchvision's public ResNet-18 with ImageNet pretrained weights
for the ten-class CIFAR-10 classification task. To support 32×32 images, the first
convolution is replaced with a 3×3 convolution with stride 1, and the initial max pooling
layer is removed. The new first convolution and classification layer are randomly
initialized; the remaining layers retain ImageNet pretrained parameters.
Training uses random cropping, horizontal flipping, label smoothing,
cosine annealing, and automatic mixed precision.

## Results

The following table describes the **historical experiment**. It selected the best epoch
using the test set, so 93.84% is a test-selected maximum, not an independent final score.
The corrected pipeline splits the 50,000 training images into 45,000 training and 5,000
validation images (seed 42), selects checkpoints using validation accuracy, and evaluates
the selected checkpoint once on the 10,000 test images. A full 20-epoch result requires
retraining; the corrected one-epoch functional check reached 82.23% test accuracy.

| Metric | Value |
| --- | ---: |
| Model | ResNet-18 |
| Dataset | CIFAR-10 |
| Training epochs | 20 |
| Best epoch | 16 |
| Historical maximum test accuracy (used for selection) | **93.84%** |
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
├── scripts/                 # Training, evaluation, and prediction commands
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

Every run creates a separate timestamped directory. Use `--run-name baseline` for an
explicit name; an existing name is rejected to protect previous results.
The best checkpoint, metrics, and actual environment versions are saved to:

```text
experiments/resnet18_cifar10/<run-name>/
├── best.pt
├── metrics.json
└── environment.json
```

### 3. Run a Smoke Test

```bash
python scripts/train.py --config configs/resnet18.yaml --epochs 1
```

### 4. Evaluate a Checkpoint

```bash
python scripts/evaluate.py --checkpoint experiments/resnet18_cifar10/<run-name>/best.pt
```

### 5. Predict an Image

```bash
python scripts/predict.py --checkpoint experiments/resnet18_cifar10/<run-name>/best.pt --image example.png
```

Replace `<run-name>` with the actual directory name. Prediction resizes RGB images to
32×32 and prints the top three class probabilities. Performance on arbitrary images
outside CIFAR-10 is not established.

## Configuration

The main hyperparameters are defined in `configs/resnet18.yaml`, including batch size,
learning rate, training epochs, weight decay, label smoothing, and data and output paths.
Edit this configuration to run experiment variants without changing the training code.

`deterministic: true` enables deterministic PyTorch algorithms and disables cuDNN
benchmarking. Exact results can still differ across hardware and software versions.
Each run records Python, PyTorch, torchvision, NumPy, PyYAML, CUDA, cuDNN, and GPU details
in `environment.json`. The historical run used Python 3.13.11, PyTorch 2.9.0+cu126,
and torchvision 0.24.0+cu126; its remaining package versions were not recorded.
The minimum versions in `requirements.txt` are not an exact environment lock.

## Model and Dataset Sources

- Model: [torchvision ResNet-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

See [`docs/RESULTS.md`](docs/RESULTS.md) for the experiment report. Dataset files and model
weights are excluded from Git to keep the repository lightweight and organized.

