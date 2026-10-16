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
from datasets.cifar10 import CLASSES, build_test_loader
from losses import build_loss
from models import build_model
from tools.utils import load_config, save_json, seed_everything, create_run_dir, environment_info
from tools.utils import validate_config, capture_rng, restore_rng, save_checkpoint


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
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Resume an unfinished run from last.pt")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Save and stop at this epoch without evaluating the test set")
    args = parser.parse_args()
    resumed = None
    if args.resume:
        if args.epochs is not None or args.run_name is not None:
            parser.error("--resume cannot be combined with --epochs or --run-name")
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "optimizer" not in resumed or "rng" not in resumed:
            parser.error("Resume requires last.pt, not a best-model checkpoint")
    cfg = resumed["config"] if resumed else load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    validate_config(cfg)
    if args.stop_after is not None and not 1 <= args.stop_after < cfg["training"]["epochs"]:
        parser.error("--stop-after must be at least 1 and less than the total epochs")
    seed_everything(cfg["seed"], cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.resume).resolve().parent if resumed else create_run_dir(ROOT / cfg["output_dir"], args.run_name)
    if resumed and (out / "metrics.json").exists():
        import json
        if "test_acc" in json.loads((out / "metrics.json").read_text()):
            parser.error("This run already completed its final test evaluation")
    print(f"Run directory: {out}", flush=True)
    save_json(environment_info(), out / ("resume_environment.json" if resumed else "environment.json"))
    train_loader, val_loader = build_loaders(
        str(ROOT / cfg["data_dir"]), cfg["training"]["batch_size"], cfg["training"]["num_workers"],
        seed=cfg["seed"], val_size=cfg.get("val_size", 5000)
    )
    model = build_model(cfg["model"]["num_classes"], cfg["model"]["pretrained"] and not resumed).to(device)
    criterion = build_loss(cfg["training"]["label_smoothing"])
    optimizer = SGD(model.parameters(), lr=cfg["training"]["learning_rate"], momentum=0.9,
                    weight_decay=cfg["training"]["weight_decay"], nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")
    history, best_acc = [], -1.0
    start_epoch, elapsed, best_epoch, best_state = 1, 0.0, None, None
    if resumed:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        history, best_acc = resumed["history"], resumed["best_val_acc"]
        start_epoch, elapsed = resumed["epoch"] + 1, resumed["elapsed_seconds"]
        best_epoch, best_state = resumed["best_epoch"], resumed["best_model"]
        restore_rng(resumed["rng"], train_loader.generator)
        save_checkpoint({"model": best_state, "config": cfg, "epoch": best_epoch,
                         "val_acc": best_acc, "classes": list(CLASSES)}, out / "best.pt")
    if args.stop_after is not None and args.stop_after < start_epoch:
        parser.error("--stop-after must not precede the next epoch")
    started = time.time()
    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
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
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": running_loss / total, "train_acc": correct / total,
               "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint({"model": best_state, "config": cfg, "epoch": epoch,
                        "val_acc": val_acc, "classes": list(CLASSES)}, out / "best.pt")
        scheduler.step()
        save_checkpoint({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "scaler": scaler.state_dict(), "rng": capture_rng(train_loader.generator),
                         "history": history, "best_val_acc": best_acc, "best_epoch": best_epoch,
                         "best_model": best_state, "elapsed_seconds": elapsed + time.time() - started},
                        out / "last.pt")
        save_json({"config": cfg, "history": history, "best_val_acc": best_acc}, out / "metrics.json")
        print(f"epoch={epoch} val_loss={val_loss:.4f} val_acc={val_acc:.4%} best={best_acc:.4%}", flush=True)
        if args.stop_after == epoch:
            print(f"Stopped after epoch {epoch}. Resume using --resume {out / 'last.pt'}", flush=True)
            return
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_loader = build_test_loader(str(ROOT / cfg["data_dir"]), cfg["training"]["batch_size"],
                                    cfg["training"]["num_workers"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    save_json({"config": cfg, "history": history, "best_val_acc": best_acc,
               "best_epoch": checkpoint["epoch"], "test_loss": test_loss, "test_acc": test_acc,
               "elapsed_seconds": elapsed + time.time() - started}, out / "metrics.json")
    print(f"Training complete. Best validation accuracy: {best_acc:.4%}; test accuracy: {test_acc:.4%}")


if __name__ == "__main__":
    main()

