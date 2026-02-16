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

# IMPORTANT ADDITION:
# - optional selection of validation dataset(s) (e.g. only RAF-DB, or only FER, or both, or all)
# - if a chosen dataset has no official test set, we create a holdout from its training pool and remove it from training (no leakage) -> this allows us to still evaluate on that dataset even without an official test set, and to control the mix ratio of that dataset in validation by controlling the holdout ratio and thus the number of eval samples from that dataset


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


def _split_samples_for_eval(
    samples: List[IMG_SAMPLE],
    seed: int,
    holdout_ratio: float,
) -> Tuple[List[IMG_SAMPLE], List[IMG_SAMPLE]]:
    """
    If a dataset has NO official test set (e.g. AffectNet often in our setup),
    we can still evaluate on it without leakage by holding out a deterministic subset
    from its TRAIN-POOL and removing it from training.

    returns: (train_keep, eval_holdout)
    """
    n = len(samples)
    n_eval = int(n * holdout_ratio)
    n_train = n - n_eval
    if n_eval <= 0 or n_train <= 0:
        raise RuntimeError(f"Dataset too small for holdout split: n={n}, holdout_ratio={holdout_ratio}")

    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)

    eval_idxs = set(idxs[:n_eval])
    eval_holdout = [samples[i] for i in range(n) if i in eval_idxs]
    train_keep = [samples[i] for i in range(n) if i not in eval_idxs]
    return train_keep, eval_holdout


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

        # NEW (optional): choose on which dataset(s) validation (= "eval during training") runs.
        # - default behavior (if missing): val is a split from the mixed TRAIN-POOL (like now)
        # - if you set "datasets": ["rafdb", "fer"] then val will be built from those datasets
        #   (prefer official test split; if missing, we make a holdout split from that dataset's TRAIN-POOL)
        #
        # examples:
        #   eval:
        #     datasets: ["rafdb"]                # only RAF-DB
        #   eval:
        #     datasets: ["fer"]                  # only FER
        #   eval:
        #     datasets: ["affectnet"]            # only AffectNet
        #   eval:
        #     datasets: ["rafdb", "fer"]         # RAF-DB + FER
        #   eval:
        #     datasets: ["affectnet", "fer"]     # AffectNet + FER
        #   eval:
        #     datasets: ["affectnet", "rafdb"]   # AffectNet + RAF-DB
        #
        # (optional) if a chosen dataset has no official test pool:
        #   holdout_ratio: 0.15   # defaults to val_ratio
        #
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

    # NEW: optional selection of validation dataset(s)
    eval_cfg = data_cfg.get("eval", {}) or {}
    eval_datasets_cfg = eval_cfg.get("datasets", None)  # e.g. ["rafdb", "fer"] or None
    if isinstance(eval_datasets_cfg, str):
        # allow: "rafdb,fer"
        eval_datasets: Optional[List[str]] = [x.strip() for x in eval_datasets_cfg.split(",") if x.strip()]
    else:
        eval_datasets = list(eval_datasets_cfg) if eval_datasets_cfg else None

    holdout_ratio = float(eval_cfg.get("holdout_ratio", val_ratio))

    batch_size = int(dl_cfg.get("batch_size", 64))
    eval_batch_size = int(dl_cfg.get("eval_batch_size", batch_size))
    num_workers = int(dl_cfg.get("num_workers", 4))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    persistent_workers = bool(dl_cfg.get("persistent_workers", True)) and num_workers > 0
    drop_last = bool(dl_cfg.get("drop_last", True))

    # collect TRAIN datasets and TEST datasets separately (no transforms yet)
    # (we store pools per dataset-name so that we can optionally build val from selected datasets)
    pooled_train_by_name: Dict[str, List[IMG_SAMPLE]] = {}
    pooled_test_by_name: Dict[str, List[IMG_SAMPLE]] = {}

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

        pooled_train_by_name["fer"] = pooled_train
        pooled_test_by_name["fer"] = pooled_test

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

        pooled_test: List[IMG_SAMPLE] = []
        if test_images_root and test_csvs:
            for csv_path in test_csvs:
                pooled_test.extend(rafdb_csv_adapter(test_images_root, csv_path))

            pooled_test = _maybe_limit(pooled_test, raf.get("limit_test"), seed=seed + 101)

            print(f"[RAF] test pooled after limit: {len(pooled_test)} (test_images_root={test_images_root})")
            if pooled_test:
                print("[RAF] test example path:", pooled_test[0][0])

        pooled_train_by_name["rafdb"] = pooled_train
        pooled_test_by_name["rafdb"] = pooled_test

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

        pooled_test: List[IMG_SAMPLE] = []
        if test_csvs:
            for csv_path in test_csvs:
                pooled_test.extend(affectnet_csv_adapter(images_root, csv_path))

            pooled_test = _maybe_limit(pooled_test, aff.get("limit_test"), seed=seed + 102)

            print(f"[AffectNet] test pooled after limit: {len(pooled_test)} (images_root={images_root})")
            if pooled_test:
                print("[AffectNet] test example path:", pooled_test[0][0])

        pooled_train_by_name["affectnet"] = pooled_train
        pooled_test_by_name["affectnet"] = pooled_test

    if len(pooled_train_by_name) == 0:
        raise RuntimeError("No TRAIN dataset activated. Set cfg.data.<name>.enabled=True and provide train data.")

    # Decide how to build VAL:
    #
    # - default: mixed val split from mixed TRAIN-POOL (same as current behavior)
    # - if eval.datasets is set: build val from those dataset(s)
    #     * prefer official TEST-POOL for those datasets
    #     * if a chosen dataset has no TEST-POOL, create a holdout from its TRAIN-POOL and REMOVE it from training (no leakage)
    eval_samples: Optional[List[IMG_SAMPLE]] = None
    if eval_datasets is not None:
        eval_samples = []

        # normalize names
        eval_datasets_norm = [d.strip().lower() for d in eval_datasets if str(d).strip()]
        valid_names = set(pooled_train_by_name.keys())
        unknown = [d for d in eval_datasets_norm if d not in valid_names]
        if unknown:
            raise RuntimeError(
                f"Unknown eval dataset(s) {unknown}. Valid: {sorted(list(valid_names))}"
            )

        for name in eval_datasets_norm:
            test_pool = pooled_test_by_name.get(name, [])
            if test_pool:
                # evaluate on the official test pool (best choice)
                eval_samples.extend(test_pool)
            else:
                # no official test -> create a holdout from TRAIN-POOL and REMOVE it from training
                train_pool = pooled_train_by_name.get(name, [])
                train_keep, eval_holdout = _split_samples_for_eval(
                    train_pool,
                    seed=seed + 500 + hash(name) % 1000,
                    holdout_ratio=holdout_ratio,
                )
                pooled_train_by_name[name] = train_keep
                eval_samples.extend(eval_holdout)

        if len(eval_samples) == 0:
            raise RuntimeError(
                "You selected eval.datasets, but the resulting evaluation pool is empty. "
                "Check your dataset paths / test_csvs / holdout settings."
            )

        print(f"[EVAL] Using selected eval datasets={eval_datasets_norm} with total samples={len(eval_samples)}")
    else:
        print("[EVAL] Using default mixed validation split from TRAIN-POOL (like now).")

    # Build TRAIN datasets (after optional holdout removal above)
    train_datasets: List[Dataset] = []
    for name, samples in pooled_train_by_name.items():
        if len(samples) == 0:
            raise RuntimeError(
                f"Dataset '{name}' is enabled, but after eval holdout it has 0 TRAIN samples left."
            )
        train_datasets.append(FER_SamplesPILDataset(samples))

    # Build TEST datasets (always official test pools across enabled datasets, if they exist)
    test_datasets: List[Dataset] = []
    for name, samples in pooled_test_by_name.items():
        if len(samples) > 0:
            test_datasets.append(FER_SamplesPILDataset(samples))

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

    # VAL: either default split from train_full OR fixed eval pool created above
    if eval_samples is None:
        # split train_full into train/val (val comes from train pool, because many datasets do not have an official val folder)
        train_subset, val_subset = split_train_val(train_full, seed=seed, val_ratio=val_ratio)
    else:
        # no random split: train is full train_full, val is fixed pool (test or holdout)
        train_subset = train_full
        val_subset = FER_SamplesPILDataset(eval_samples)

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