from __future__ import annotations

import os
import random
from typing import Tuple, Optional
import numpy as np
import torch

from torch.utils.data import DataLoader, ConcatDataset

from data.adapters import rafdb_csv_adapter, affectnet_csv_adapter, folder_adapter
from data.dataset import FER_Dataset, FER_DatasetFromSamples


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataset(dataset_cfg: dict, set_type: str) -> Optional[torch.utils.data.Dataset]:
    ds_type = dataset_cfg.get("type", "folder")
    root = dataset_cfg.get("root")
    
    if ds_type == "rafdb":
        # RAF-DB typically has CSV files for splits
        if set_type == "train":
            csv_path = dataset_cfg.get("train_csv")
        elif set_type == "val":
            csv_path = dataset_cfg.get("val_csv")
        elif set_type == "test":
            csv_path = dataset_cfg.get("test_csv")
        else:
            return None
            
        if not csv_path or not os.path.exists(csv_path):
            return None
            
        samples = rafdb_csv_adapter(root, csv_path)
        return FER_DatasetFromSamples(samples, set_type=set_type)

    elif ds_type == "affectnet":
        if set_type == "train":
            csv_path = dataset_cfg.get("train_csv")
        elif set_type == "val":
            csv_path = dataset_cfg.get("val_csv")
        # AffectNet usually validation is the test set for us, or separate
        else:
            return None # Or handle test if needed

        if not csv_path or not os.path.exists(csv_path):
            return None

        samples = affectnet_csv_adapter(root, csv_path)
        return FER_DatasetFromSamples(samples, set_type=set_type)

    elif ds_type == "folder":
        # Folder dataset: root/split/class/image.jpg OR root/class/image.jpg
        # If set_type is provided, try to find a subfolder with that name
        # If not found, and it's train, maybe use the whole root? 
        # For mixed datasets, usually we assume root/train, root/val structure for folder datasets
        
        target_dir = os.path.join(root, set_type) if set_type in ["train", "val", "test"] else root
        if not os.path.isdir(target_dir):
            if set_type == "train":
                # Fallback: if no split folders, assume root is the dataset (only for training?)
                # Or just strictly require 'training', 'validation' folders?
                # Let's check for standard names
                alt_dir = os.path.join(root, "training")
                if os.path.isdir(alt_dir):
                    target_dir = alt_dir
                else:
                    target_dir = root # Last resort: just use root
            elif set_type == "val":
                 alt_dir = os.path.join(root, "validation")
                 if os.path.isdir(alt_dir):
                     target_dir = alt_dir
                 else:
                     return None
            else:
                return None
        
        return FER_Dataset(root_dir=target_dir, set_type=set_type)

    return None


def get_dataloaders(
    datasets_cfg: list,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]: 
   
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for ds_cfg in datasets_cfg:
        train_ds = build_dataset(ds_cfg, "train")
        if train_ds is not None and len(train_ds) > 0:
            train_datasets.append(train_ds)
            
        val_ds = build_dataset(ds_cfg, "val")
        if val_ds is not None and len(val_ds) > 0:
            val_datasets.append(val_ds)

        test_ds = build_dataset(ds_cfg, "test")
        if test_ds is not None and len(test_ds) > 0:
            test_datasets.append(test_ds)

    if not train_datasets:
        raise ValueError("No training datasets found! Check config.")

    # Combine datasets
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets) if val_datasets else None
    combined_test = ConcatDataset(test_datasets) if test_datasets else None

    # Helper to get classes from the first dataset (assuming consistency)
    # We should probably enforce consistency or merge classes, but for now take first
    if hasattr(train_datasets[0], "classes_list"):
         combined_train.classes_list = train_datasets[0].classes_list

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        combined_train, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0, 
    ) if combined_val else None

    test_loader = DataLoader(
        combined_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    ) if combined_test else None

    return train_loader, val_loader, test_loader
