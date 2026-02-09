import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from data.dataloader import get_dataloaders, split_70_15_15


CANONICAL = ["anger", "fear", "disgust", "sadness", "happiness", "surprise"]


def _make_rgb_image(path: Path, size=(80, 80), color=(120, 20, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def _build_fake_fer_root(tmp_path: Path, n_per_class: int = 5) -> Path:
    """
    Builds a fake FER-like folder structure:
      root/
        training/<class>/*.png
        validation/<class>/*.png
        test/<class>/*.png
    """
    fer_root = tmp_path / "fake_fer"
    splits = ["training", "validation", "test"]

    # Make deterministic but varied colors per class
    base_colors = [
        (200, 30, 30),   # anger
        (30, 30, 200),   # fear
        (30, 200, 30),   # disgust
        (120, 120, 120), # sadness
        (220, 220, 40),  # happiness
        (200, 40, 200),  # surprise
    ]

    for split in splits:
        for ci, cls in enumerate(CANONICAL):
            for i in range(n_per_class):
                p = fer_root / split / cls / f"{cls}_{split}_{i}.png"
                _make_rgb_image(p, size=(96, 96), color=base_colors[ci])

    return fer_root


def test_split_70_15_15_sizes():
    # quick unit test for the split helper itself
    class Dummy(torch.utils.data.Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return i

    ds = Dummy(100)
    tr, va, te = split_70_15_15(ds, seed=123)

    assert len(tr) == 70
    assert len(va) == 15
    assert len(te) == 15
    assert len(tr) + len(va) + len(te) == 100


def test_get_dataloaders_end_to_end(tmp_path: Path):
    fer_root = _build_fake_fer_root(tmp_path, n_per_class=4)  # small but enough

    cfg = {
        "seed": 42,
        "dataloader": {
            "batch_size": 8,
            "eval_batch_size": 8,
            "num_workers": 0,          # important for tests (no multiprocessing headaches)
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
        },
        "data": {
            "fer": {"enabled": True, "root": str(fer_root), "limit": None},
            "rafdb": {"enabled": False},
            "affectnet": {"enabled": False},
        },
    }

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # basic sanity: loaders exist
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # dataset sizes: pooled = 3 splits * 6 classes * n_per_class
    pooled_n = 3 * len(CANONICAL) * 4  # = 72
    # expected split sizes follow your split logic
    n_train = int(pooled_n * 0.70)  # 50
    n_val = int(pooled_n * 0.15)    # 10
    n_test = pooled_n - n_train - n_val  # 12

    assert len(train_loader.dataset) == n_train
    assert len(val_loader.dataset) == n_val
    assert len(test_loader.dataset) == n_test

    # check a training batch: tensors, shapes, dtypes
    xb, yb = next(iter(train_loader))
    assert isinstance(xb, torch.Tensor)
    assert isinstance(yb, torch.Tensor)

    assert xb.dtype == torch.float32
    assert yb.dtype == torch.long

    # expected image shape: [B, 3, 64, 64] (your transforms resize to 64)
    assert xb.ndim == 4
    assert xb.shape[0] == cfg["dataloader"]["batch_size"]
    assert xb.shape[1:] == (3, 64, 64)

    assert yb.ndim == 1
    assert yb.shape[0] == cfg["dataloader"]["batch_size"]

    # labels in range [0..5]
    assert int(yb.min()) >= 0
    assert int(yb.max()) <= 5

    # no NaNs/Infs after normalize
    assert torch.isfinite(xb).all()


def test_train_loader_deterministic_shuffle(tmp_path: Path):
    fer_root = _build_fake_fer_root(tmp_path, n_per_class=4)

    cfg = {
        "seed": 123,
        "dataloader": {
            "batch_size": 8,
            "eval_batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
        },
        "data": {
            "fer": {"enabled": True, "root": str(fer_root), "limit": None},
            "rafdb": {"enabled": False},
            "affectnet": {"enabled": False},
        },
    }

    tl1, _, _ = get_dataloaders(cfg)
    tl2, _, _ = get_dataloaders(cfg)

    y1 = next(iter(tl1))[1]
    y2 = next(iter(tl2))[1]

    # same seed -> same first batch label order (because generator + worker seeding)
    assert torch.equal(y1, y2)