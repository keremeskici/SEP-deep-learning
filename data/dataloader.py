# dataloader is used to create batches from the dataset and to load them during training/validation/testing
# for the dataloader implementation I use the DataLoader class from torch.utils.data
# a worker in Pytorch is a predifined subprocess that is used to load data in parallel to the main process, which can significantly speed up the data loading process, especially when dealing with large datasets or complex data transformations

# since multiple datasets can be combined dataloader is now changed to use mixtures of different datasets
# ConcatDataset takes a list of datasets and concatenates them to one big dataset
# So Dataset can be mixed

# IMPORTANT CHANGE (no leakage / no mixing official test into train):
# - build a TRAIN-POOL (only train data of each dataset)
# - split TRAIN-POOL into train/val (e.g. 85/15)
# - build a TEST-POOL (only official test data) and do NOT random-split it

# for performace it is better to take the exsisting, official train/test set rather than to stick with 70/15/15 ratio

# Then every set chooses the right augmentation from transforms.py
# Also mean/std is computed ONLY on the training subset


from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from data.dataset import FER_SamplesPILDataset
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


# Split helper: create validation from TRAIN-POOL
def split_train_val(train_ds: Dataset, seed: int, val_ratio: float = 0.15):
    n = len(train_ds)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    if n_val <= 0 or n_train <= 0:
        raise RuntimeError(f"Dataset too small for train/val split: n={n}, val_ratio={val_ratio}")

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_ds, [n_train, n_val], generator=g)
    return train_subset, val_subset


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
        "val_ratio": 0.15,

        "fer": {
          "enabled": True,
          "root": "/path/to/FER-2013",
          "limit": None,
          "limit_test": None,
          "include_validation_in_train": True
        },

        "rafdb": {
          "enabled": True,
          "train_images_root": "/path/to/RAF-DB/DATASET/train",
          "test_images_root":  "/path/to/RAF-DB/DATASET/test",
          "train_csvs": ["/path/to/RAF-DB/train_labels.csv"],
          "test_csvs":  ["/path/to/RAF-DB/test_labels.csv"],
          "limit": None,
          "limit_test": None
        },

        "affectnet": {
          "enabled": True,
          "images_root": "/path/to/AffectNet",
          "train_csvs": ["/path/to/AffectNet/labels_train.csv"],
          "test_csvs":  ["/path/to/AffectNet/labels_test.csv"],  # optional; can be []
          "limit": None,
          "limit_test": None
        }
      }
    }
    """

    seed = int(cfg.get("seed", 42))
    dl_cfg = cfg.get("dataloader", {})
    data_cfg = cfg.get("data", {})

    val_ratio = float(data_cfg.get("val_ratio", 0.15))

    batch_size = int(dl_cfg.get("batch_size", 64))
    eval_batch_size = int(dl_cfg.get("eval_batch_size", batch_size))
    num_workers = int(dl_cfg.get("num_workers", 4))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    persistent_workers = bool(dl_cfg.get("persistent_workers", True)) and num_workers > 0
    drop_last = bool(dl_cfg.get("drop_last", True))

    # collect TRAIN datasets and TEST datasets separately (no transforms yet)
    train_datasets: List[Dataset] = []
    test_datasets: List[Dataset] = []

    
    # FER Folder (train/val from training(+optional validation), test from test folder)
    fer = data_cfg.get("fer", {})
    if fer.get("enabled", False):
        fer_root = fer.get("root")
        if not fer_root:
            raise RuntimeError("cfg.data.fer.enabled=True but cfg.data.fer.root is missing.")

        include_validation = bool(fer.get("include_validation_in_train", True))

        pooled_train: List[IMG_SAMPLE] = []
        for sub in ("training", "train"):
            p = os.path.join(fer_root, sub)
            if os.path.isdir(p):
                pooled_train.extend(folder_adapter(p))

        if include_validation:
            p = os.path.join(fer_root, "validation")
            if os.path.isdir(p):
                pooled_train.extend(folder_adapter(p))

        pooled_test: List[IMG_SAMPLE] = []
        p = os.path.join(fer_root, "test")
        if os.path.isdir(p):
            pooled_test.extend(folder_adapter(p))

        pooled_train = _maybe_limit(pooled_train, fer.get("limit"), seed=seed)
        pooled_test = _maybe_limit(pooled_test, fer.get("limit_test"), seed=seed + 100)

        print(f"[FER] train pooled after limit: {len(pooled_train)} (root={fer_root})")
        if pooled_train:
            print("[FER] train example path:", pooled_train[0][0])

        print(f"[FER] test pooled after limit: {len(pooled_test)} (root={fer_root})")
        if pooled_test:
            print("[FER] test example path:", pooled_test[0][0])

        if len(pooled_train) == 0:
            raise RuntimeError(f"FER enabled but 0 TRAIN samples found under {fer_root}.")

        train_datasets.append(FER_SamplesPILDataset(pooled_train))

        # test is optional for FER, but recommended
        if len(pooled_test) > 0:
            test_datasets.append(FER_SamplesPILDataset(pooled_test))

    
    # RAF-DB CSV (train from train_csvs + train_images_root, test from test_csvs + test_images_root)
    raf = data_cfg.get("rafdb", {})
    if raf.get("enabled", False):
        train_images_root = raf.get("train_images_root")
        test_images_root = raf.get("test_images_root")
        train_csvs = raf.get("train_csvs", [])
        test_csvs = raf.get("test_csvs", [])

        if not train_images_root or not train_csvs:
            raise RuntimeError("cfg.data.rafdb.enabled=True but train_images_root or train_csvs missing.")

        pooled_train: List[IMG_SAMPLE] = []
        for csv_path in train_csvs:
            pooled_train.extend(rafdb_csv_adapter(train_images_root, csv_path))

        pooled_train = _maybe_limit(pooled_train, raf.get("limit"), seed=seed + 1)

        print(f"[RAF] train pooled after limit: {len(pooled_train)} (train_images_root={train_images_root})")
        if pooled_train:
            print("[RAF] train example path:", pooled_train[0][0])

        if len(pooled_train) == 0:
            raise RuntimeError("RAF-DB enabled, but 0 TRAIN samples after adapter (check paths/CSV/labels).")

        train_datasets.append(FER_SamplesPILDataset(pooled_train))

        # test optional, but recommended
        if test_images_root and test_csvs:
            pooled_test: List[IMG_SAMPLE] = []
            for csv_path in test_csvs:
                pooled_test.extend(rafdb_csv_adapter(test_images_root, csv_path))

            pooled_test = _maybe_limit(pooled_test, raf.get("limit_test"), seed=seed + 101)

            print(f"[RAF] test pooled after limit: {len(pooled_test)} (test_images_root={test_images_root})")
            if pooled_test:
                print("[RAF] test example path:", pooled_test[0][0])

            if len(pooled_test) > 0:
                test_datasets.append(FER_SamplesPILDataset(pooled_test))

    
    # AffectNet CSV (train_csvs and test_csvs; adapter can resolve Train/Test under images_root)
    aff = data_cfg.get("affectnet", {})
    if aff.get("enabled", False):
        images_root = aff.get("images_root")
        train_csvs = aff.get("train_csvs", [])
        test_csvs = aff.get("test_csvs", [])

        if not images_root or not train_csvs:
            raise RuntimeError("cfg.data.affectnet.enabled=True but images_root or train_csvs missing.")

        pooled_train: List[IMG_SAMPLE] = []
        for csv_path in train_csvs:
            pooled_train.extend(affectnet_csv_adapter(images_root, csv_path))

        pooled_train = _maybe_limit(pooled_train, aff.get("limit"), seed=seed + 2)

        print(f"[AffectNet] train pooled after limit: {len(pooled_train)} (images_root={images_root})")
        if pooled_train:
            print("[AffectNet] train example path:", pooled_train[0][0])

        if len(pooled_train) == 0:
            raise RuntimeError("AffectNet enabled, but 0 TRAIN samples after adapter (check paths/CSV/labels).")

        train_datasets.append(FER_SamplesPILDataset(pooled_train))

        # test is optional (depends if you have labeled AffectNet test)
        if test_csvs:
            pooled_test: List[IMG_SAMPLE] = []
            for csv_path in test_csvs:
                pooled_test.extend(affectnet_csv_adapter(images_root, csv_path))

            pooled_test = _maybe_limit(pooled_test, aff.get("limit_test"), seed=seed + 102)

            print(f"[AffectNet] test pooled after limit: {len(pooled_test)} (images_root={images_root})")
            if pooled_test:
                print("[AffectNet] test example path:", pooled_test[0][0])

            if len(pooled_test) > 0:
                test_datasets.append(FER_SamplesPILDataset(pooled_test))

    if len(train_datasets) == 0:
        raise RuntimeError("No TRAIN dataset activated. Set cfg.data.<name>.enabled=True and provide train data.")

    print("[MIX] number of TRAIN datasets:", len(train_datasets))
    print("[MIX] TRAIN dataset lens:", [len(ds) for ds in train_datasets])
    print("[MIX] total TRAIN before concat:", sum(len(ds) for ds in train_datasets))

    if len(test_datasets) == 0:
        # you can decide to allow this, but most training scripts expect a test_loader
        raise RuntimeError("No TEST dataset found/activated. Provide test data or adapt evaluation.")

    print("[MIX] number of TEST datasets:", len(test_datasets))
    print("[MIX] TEST dataset lens:", [len(ds) for ds in test_datasets])
    print("[MIX] total TEST before concat:", sum(len(ds) for ds in test_datasets))

    # concat BEFORE split because the mix of different TRAIN datasets should be considered one
    train_full = ConcatDataset(train_datasets)

    # split train_full into train/val (val comes from train pool, because many datasets do not have an official val folder)
    train_subset, val_subset = split_train_val(train_full, seed=seed, val_ratio=val_ratio)

    # official test is kept separate (no random split!)
    test_full = ConcatDataset(test_datasets)
    test_subset = test_full

    # Compute mean/std ONLY on training subset (after splitting and concat of course)
    mean, std = compute_mean_std(
        train_subset,  # mean and std is only calculated on the trainset otherwise information from val and test would get into training (data leakage)
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
        transform=get_train_transforms(mean, std),
    )

    val_ds = TransformDataset(
        val_subset,
        transform=get_val_transforms(mean, std),
    )

    test_ds = TransformDataset(
        test_subset,
        transform=get_test_transforms(mean, std),
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
        shuffle=False,  # of course no shuffling in validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,  # no shuffling in test as well
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader