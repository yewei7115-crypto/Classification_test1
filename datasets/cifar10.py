from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def evaluation_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def build_test_loader(data_dir, batch_size, num_workers):
    dataset = datasets.CIFAR10(root=Path(data_dir), train=False, download=True,
                               transform=evaluation_transform())
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=torch.cuda.is_available())


def build_loaders(data_dir: str, batch_size: int, num_workers: int, seed=42, val_size=5000):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    root = Path(data_dir)
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(root=root, train=True, download=False, transform=evaluation_transform())
    if not 0 < val_size < len(train_set):
        raise ValueError("val_size must be between 1 and the training set size minus 1")
    indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(seed)).tolist()
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    train_loader = DataLoader(Subset(train_set, indices[val_size:]), shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(Subset(val_set, indices[:val_size]), shuffle=False, **common)
    return train_loader, val_loader

