import argparse
import sys
import time
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import build_loaders
from losses import build_loss
from models import build_model
from tools.utils import load_config, save_json, seed_everything


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/resnet18.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = build_loaders(
        str(ROOT / cfg["data_dir"]), cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    )
    model = build_model(cfg["model"]["num_classes"], cfg["model"]["pretrained"]).to(device)
    criterion = build_loss(cfg["training"]["label_smoothing"])
    optimizer = SGD(model.parameters(), lr=cfg["training"]["learning_rate"], momentum=0.9,
                    weight_decay=cfg["training"]["weight_decay"], nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")
    history, best_acc = [], 0.0
    started = time.time()
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        running_loss = correct = total = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']}")
        for images, targets in progress:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
            progress.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": running_loss / total, "train_acc": correct / total,
               "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        save_json({"config": cfg, "history": history, "best_acc": max(best_acc, val_acc)}, out / "metrics.json")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "val_acc": val_acc}, out / "best.pt")
        scheduler.step()
        print(f"epoch={epoch} val_loss={val_loss:.4f} val_acc={val_acc:.4%} best={best_acc:.4%}", flush=True)
    save_json({"config": cfg, "history": history, "best_acc": best_acc,
               "elapsed_seconds": time.time() - started}, out / "metrics.json")
    print(f"Training complete. Best accuracy: {best_acc:.4%}")


if __name__ == "__main__":
    main()

