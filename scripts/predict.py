"""Classify an image with a saved CIFAR-10 checkpoint."""
import argparse
import sys
from pathlib import Path

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.cifar10 import CLASSES, evaluation_transform
from models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"]["model"]["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with Image.open(args.image) as image:
        tensor = evaluation_transform()(image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR))
    with torch.inference_mode():
        probabilities = model(tensor.unsqueeze(0)).softmax(dim=1)[0]
    classes = checkpoint.get("classes", CLASSES)
    for index in probabilities.argsort(descending=True)[:3].tolist():
        print(f"{classes[index]}: {probabilities[index].item():.2%}")


if __name__ == "__main__":
    main()
