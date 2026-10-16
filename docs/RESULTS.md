# Experiment Results

- Dataset: CIFAR-10 (50,000 train / 10,000 test images)
- Model: torchvision ResNet-18 with public ImageNet weights
- Hardware: NVIDIA GeForce RTX 3090 (24 GB)
- Epochs: 20
- Best epoch: 16
- Best test accuracy: 93.84%
- Total training time: 244.04 seconds

Artifacts are stored in `experiments/resnet18_cifar10/`:

- `best.pt`: best model checkpoint
- `metrics.json`: configuration and per-epoch metrics

