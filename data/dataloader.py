# dataloader is used to create batches from the dataset and to load them during training/validation/testing
# for the dataloader implementation I use the DataLoader class from torch.utils.data
# a worker in Pytorch is a predifined subprocess that is used to load data in parallel to the main process, which can significantly speed up the data loading process, especially when dealing with large datasets or complex data transformations

# since multiple datasets can be combined dataloader is now changed to use mixtures of different datasets
# ConcatDataset takes a list of datasets and concatenates them to one big dataset
# So Dataset can be mixed
# After conctenating the now big Dataset is split into Training/validation/test using (70/15/15) ratio
# Then every set chooses the right augmentation from transforms.py


from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from data.dataset import FER_Dataset
from data.transforms import get_train_transforms, get_val_transforms, get_test_transforms
from data.adapters import folder_adapter, rafdb_csv_adapter, affectnet_csv_adapter
from data.calculate_mean_std import compute_mean_std


IMG_SAMPLE = Tuple[str, int]  # (img_path, canonical_label_id)


# Reproducibility: 
def seed_worker(worker_id: int) -> None:
    # Every worker gets a different, deterministic seed derived from the global seed -> fixed random order
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Tranformation Wrapper Class:
class TransformDataset(Dataset):
    """
    Applies transform *after* splitting
    Expects the underlying dataset to return (PIL_image, label_tensor_or_int)
    Our FER_Dataset returns (PIL_image, torch.long) when transform=None
    """

    def __init__(self, base_ds: Dataset, transform=None):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        x, y = self.base_ds[idx] 
        if self.transform is not None:
            x = self.transform(x)
        return x, y


# Split helper to enable 70/15/15 ratio for training/validation/test
def split_70_15_15(full_ds: Dataset, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    n = len(full_ds)
    if n < 3:
        raise RuntimeError(f"Zu wenige Samples ({n}) für 70/15/15 Split.")

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val  

    # safety for very small datasets
    if n_val == 0:
        n_val = 1
        n_train = max(1, n_train - 1)
        n_test = n - n_train - n_val
    if n_test == 0:
        n_test = 1
        n_train = max(1, n_train - 1)
        n_val = n - n_train - n_test

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_ds, [n_train, n_val, n_test], generator=g
    )
    return train_subset, val_subset, test_subset


# Utility: optional limiting to control "mix ratio"
def _maybe_limit(samples: List[IMG_SAMPLE], limit: Optional[int], seed: int) -> List[IMG_SAMPLE]:
    """
    If limit is set, take a deterministic subset of that size
    Useful if you want to control mixture ratio by controlling how many images you include
    """
    if limit is None:
        return samples
    if limit <= 0:
        return []
    if len(samples) <= limit:
        return samples

    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]
    return [samples[i] for i in idxs]


# Main entry for the dataloader
def get_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Expects config like:

    config = {
      "seed": 42,
      "dataloader": {
        "batch_size": 64,
        "eval_batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True
      },
      "data": {
        "fer": {
          "enabled": True,
          "root": "/path/to/fer",
          "limit": None
        },
        "rafdb": {
          "enabled": True,
          "images_root": "/path/to/raf/images",
          "csvs": ["/path/to/train.csv", "/path/to/test.csv"],   # optional list; we pool them all
          "limit": None
        },
        "affectnet": {
          "enabled": True,
          "images_root": "/path/to/affect/images",
          "csvs": ["/path/to/train.csv", "/path/to/val.csv", "/path/to/test.csv"],
          "limit": None
        }
      }
    }
    """

    seed = int(cfg.get("seed", 42))
    dl_cfg = cfg.get("dataloader", {})
    data_cfg = cfg.get("data", {})

    batch_size = int(dl_cfg.get("batch_size", 64))
    eval_batch_size = int(dl_cfg.get("eval_batch_size", batch_size))
    num_workers = int(dl_cfg.get("num_workers", 4))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    persistent_workers = bool(dl_cfg.get("persistent_workers", True)) and num_workers > 0
    drop_last = bool(dl_cfg.get("drop_last", True))

    # collect full datasets (no transforms yet) 
    full_datasets: List[Dataset] = []

    # FER Folder:
    fer = data_cfg.get("fer", {})
    if fer.get("enabled", False):
        fer_root = fer.get("root")
        if not fer_root:
            raise RuntimeError("cfg.data.fer.enabled=True aber cfg.data.fer.root fehlt.")

        # Pool everything that exists: training/validation/test (so we can re-split 70/15/15 after concat)
        pooled: List[IMG_SAMPLE] = []
        for sub in ("training", "validation", "test"):
            p = os.path.join(fer_root, sub)
            if os.path.isdir(p):
                pooled.extend(folder_adapter(p))

        pooled = _maybe_limit(pooled, fer.get("limit"), seed=seed)

        if len(pooled) == 0:
            raise RuntimeError(f"FER enabled, aber 0 Samples gefunden unter {fer_root}.")

        full_datasets.append(FER_Dataset(pooled, transform=None))

    # RAF-DB CSV
    raf = data_cfg.get("rafdb", {})
    if raf.get("enabled", False):
        images_root = raf.get("images_root")
        csvs = raf.get("csvs", [])
        if not images_root or not csvs:
            raise RuntimeError("cfg.data.rafdb.enabled=True aber images_root oder csvs fehlt.")

        pooled: List[IMG_SAMPLE] = []
        for csv_path in csvs:
            pooled.extend(rafdb_csv_adapter(images_root, csv_path))

        pooled = _maybe_limit(pooled, raf.get("limit"), seed=seed + 1)

        if len(pooled) == 0:
            raise RuntimeError("RAF-DB enabled, aber 0 Samples nach Adapter gefunden (Pfade/CSV/Labels prüfen).")

        full_datasets.append(FER_Dataset(pooled, transform=None))

    # AffectNet CSV
    aff = data_cfg.get("affectnet", {})
    if aff.get("enabled", False):
        images_root = aff.get("images_root")
        csvs = aff.get("csvs", [])
        if not images_root or not csvs:
            raise RuntimeError("cfg.data.affectnet.enabled=True aber images_root oder csvs fehlt.")

        pooled: List[IMG_SAMPLE] = []
        for csv_path in csvs:
            pooled.extend(affectnet_csv_adapter(images_root, csv_path))

        pooled = _maybe_limit(pooled, aff.get("limit"), seed=seed + 2)

        if len(pooled) == 0:
            raise RuntimeError("AffectNet enabled, aber 0 Samples nach Adapter gefunden (Pfade/CSV/Labels prüfen).")

        full_datasets.append(FER_Dataset(pooled, transform=None))

    if len(full_datasets) == 0:
        raise RuntimeError("Kein Dataset aktiviert. Setze cfg.data.<name>.enabled=True.")

    # concat BEFORE split because the mix of different dataset should be considered one
    full_ds = ConcatDataset(full_datasets)

    # split 70/15/15
    train_subset, val_subset, test_subset = split_70_15_15(full_ds, seed=seed)

    # Compute mean/std ONLY on training subset (after splitting and concat of course)
    mean, std = compute_mean_std(
        train_subset, # mean and std s only calculated on the trainset otherwise information from val and test would get into training (data leakage)
        img_size=64,
        batch_size=256,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("Computed TRAIN mean:", mean.tolist())
    print("Computed TRAIN std:", std.tolist())

    # apply transforms AFTER split
    train_ds = TransformDataset(
    train_subset,
    transform=get_train_transforms(mean, std)
    )

    val_ds = TransformDataset(
        val_subset,
        transform=get_val_transforms(mean, std)
    )

    test_ds = TransformDataset(
        test_subset,
        transform=get_test_transforms(mean, std)
    )

    # deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False, # of course no shuffeling in validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False, # no shuffeling in test as well
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader
