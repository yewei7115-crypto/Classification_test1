"""Offline regression tests: no network access or dataset download required."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from datasets.cifar10 import build_loaders
from models import build_model
from tools.utils import (load_config, validate_config, create_run_dir, capture_rng,
                         restore_rng, save_checkpoint, seed_everything)

ROOT = Path(__file__).resolve().parents[1]


class FakeCIFAR:
    def __init__(self, **kwargs):
        self.transform = kwargs["transform"]

    def __len__(self):
        return 50000


class RegressionTests(unittest.TestCase):
    def test_config_validation(self):
        original = load_config(ROOT / "configs/resnet18.yaml")
        validate_config(original)
        cases = [("model", "name", "resnet50"), ("model", "num_classes", 9),
                 ("training", "epochs", 0), ("training", "batch_size", -1),
                 ("training", "num_workers", -1), ("training", "learning_rate", float("nan")),
                 ("training", "label_smoothing", 1)]
        for section, key, value in cases:
            with self.subTest(key=key):
                cfg = copy.deepcopy(original)
                cfg[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(cfg)
        with self.assertRaises(ValueError):
            build_model(num_classes=11, pretrained=False)

    @patch("datasets.cifar10.datasets.CIFAR10", FakeCIFAR)
    def test_split(self):
        a, b = build_loaders("unused", 16, 0, seed=42)
        c, d = build_loaders("unused", 16, 0, seed=42)
        self.assertEqual(len(a.dataset), 45000)
        self.assertEqual(len(b.dataset), 5000)
        self.assertFalse(set(a.dataset.indices) & set(b.dataset.indices))
        self.assertEqual(len(set(a.dataset.indices) | set(b.dataset.indices)), 50000)
        self.assertEqual(a.dataset.indices, c.dataset.indices)
        self.assertEqual(b.dataset.indices, d.dataset.indices)
        self.assertNotIn("Random", repr(b.dataset.dataset.transform))

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            create_run_dir(base, "run")
            with self.assertRaises(FileExistsError):
                create_run_dir(base, "run")
            for name in ("..", "../escape", "a/b", "a\\b"):
                with self.assertRaises(ValueError):
                    create_run_dir(base, name)

    def test_rng_checkpoint_roundtrip(self):
        seed_everything(42)
        generator = torch.Generator().manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint({"rng": capture_rng(generator)}, path)
            expected = (random.random(), np.random.rand(), torch.rand(3),
                        torch.randperm(10, generator=generator))
            saved = torch.load(path, weights_only=True)
            restore_rng(saved["rng"], generator)
            actual = (random.random(), np.random.rand(), torch.rand(3),
                      torch.randperm(10, generator=generator))
            self.assertEqual(expected[:2], actual[:2])
            self.assertTrue(torch.equal(expected[2], actual[2]))
            self.assertTrue(torch.equal(expected[3], actual[3]))

    def test_model_checkpoint_roundtrip(self):
        model = build_model(pretrained=False).eval()
        images = torch.randn(1, 3, 32, 32)
        with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
            expected = model(images)
            self.assertEqual(tuple(expected.shape), (1, 10))
            path = Path(directory) / "best.pt"
            save_checkpoint({"model": model.state_dict()}, path)
            loaded = build_model(pretrained=False).eval()
            loaded.load_state_dict(torch.load(path, weights_only=True)["model"])
            self.assertTrue(torch.equal(expected, loaded(images)))


if __name__ == "__main__":
    unittest.main()
