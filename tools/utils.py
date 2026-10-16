import json
import random
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg):
    if cfg["model"]["name"] != "resnet18":
        raise ValueError("Supported model: resnet18")
    if cfg["model"]["num_classes"] != 10:
        raise ValueError("CIFAR-10 requires num_classes=10")
    t = cfg["training"]
    for key in ("epochs", "batch_size"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(t["num_workers"]) is not int or t["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(cfg.get("val_size", 5000)) is not int or not 0 < cfg.get("val_size", 5000) < 50000:
        raise ValueError("val_size must be an integer between 1 and 49999")
    if not 0 <= t["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if not np.isfinite(t["learning_rate"]) or t["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    for value in (cfg.get("deterministic", False), cfg["model"]["pretrained"], t["amp"]):
        if type(value) is not bool:
            raise ValueError("deterministic, pretrained and amp must be booleans")


def capture_rng(loader_generator):
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader_generator.get_state()}


def restore_rng(state, loader_generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    loader_generator.set_state(state["loader"].cpu())


def save_checkpoint(state, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def create_run_dir(base: Path, run_name=None) -> Path:
    name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not name or name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("run_name must be a single directory name")
    path = base / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_info():
    import torchvision
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "pyyaml": yaml.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)

