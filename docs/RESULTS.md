# Experiment Results

## Formal experiment: validation-selected checkpoint

| Setting or metric | Value |
| --- | ---: |
| Run | `baseline-20ep-20261016` |
| Training / validation / test images | 45,000 / 5,000 / 10,000 |
| Seed | 42 |
| Epochs | 20 |
| Best epoch selected by validation | 18 |
| Best validation accuracy | 93.98% |
| Final test accuracy | **93.30%** |
| Test loss (label smoothing 0.1) | 0.672500 |
| Training and final evaluation time | 144.06 seconds |
| GPU | NVIDIA RTX 3090 24GB |

An independent evaluation of the selected checkpoint reproduced 93.30% test accuracy.
Metrics and environment versions are available under
`experiments/resnet18_cifar10/baseline-20ep-20261016/`.
The server retains both `best.pt` for inference and `last.pt` for recovery; model
weights are excluded from Git. This is one seeded run, not a multi-seed average.

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

## Regression and resume validation

All five offline unittest cases passed on the training server. A separate two-epoch
integration check compared uninterrupted training with an epoch-one interruption
followed by `--resume`. Per-epoch histories and every final model state tensor were
identical. This verifies continuation in the tested environment at epoch boundaries.
Runs used `resume-reference-20261016` and `resume-interrupted-20261016`; their model
files remain on the server and are excluded from Git.

The earlier one-epoch smoke run checks functionality only. Its metrics and environment are stored in
`experiments/resnet18_cifar10/protocol-check-20261016/`.

