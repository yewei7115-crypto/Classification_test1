# Experiment Results

## Historical experiment: test-set-selected checkpoint

This original run evaluated the test set after every epoch and selected the checkpoint
with the highest test accuracy. Its 93.84% maximum is retained for transparency and
must not be interpreted as an independent final evaluation. The original metrics file
is preserved unchanged at `experiments/resnet18_cifar10/metrics.json`; its `val_*`
fields actually refer to the test set.

- Dataset: CIFAR-10 (50,000 train / 10,000 test images)
- Model: torchvision ResNet-18 with public ImageNet weights
- Hardware: NVIDIA GeForce RTX 3090 (24 GB)
- Epochs: 20
- Best epoch: 16
- Maximum test accuracy used for checkpoint selection: 93.84%
- Total training time: 244.04 seconds

Artifacts are stored in `experiments/resnet18_cifar10/`:

- `best.pt`: best model checkpoint
- `metrics.json`: configuration and per-epoch metrics

The checkpoint is stored on the training server and is excluded from Git.

## Corrected protocol

The updated code uses a seeded 45,000/5,000 split of the official training set for
training and validation. Validation images use evaluation transforms without random
augmentation. Only validation accuracy determines the best checkpoint. The test set
is evaluated after checkpoint selection, once per training run.

New runs use separate directories containing `best.pt`, `metrics.json`, and
`environment.json`. Their metrics distinguish `best_val_acc` from `test_acc`.
## Functional validation

A one-epoch smoke run of the corrected protocol completed on the RTX 3090:

- Validation accuracy: 82.56%
- Final test accuracy: 82.23%
- Independent evaluation of its checkpoint reproduced 82.23% test accuracy.
- The seeded split was reproducible, contained 45,000/5,000 images, and had no overlap.
- Reusing an existing run name raised `FileExistsError` as expected.
- Single-image prediction returned the top three class probabilities.

This smoke run checks functionality only. A full 20-epoch corrected-protocol experiment
has not been run. Its metrics and environment are stored in
`experiments/resnet18_cifar10/protocol-check-20261016/`.

