"""Evaluate a saved checkpoint on the CIFAR-10 test set."""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.cifar10 import build_test_loader
from losses import build_loss
from models import build_model
from scripts.train import evaluate
from tools.utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    seed_everything(cfg["seed"], cfg.get("deterministic", False))
    model = build_model(cfg["model"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = build_test_loader(args.data_dir or str(ROOT / cfg["data_dir"]),
                               cfg["training"]["batch_size"], cfg["training"]["num_workers"])
    loss, acc = evaluate(model, loader, build_loss(cfg["training"]["label_smoothing"]), device)
    print(f"Test loss: {loss:.4f}; test accuracy: {acc:.4%}")


if __name__ == "__main__":
    main()
