# calculate_mean_std.py
from __future__ import annotations

import platform
from typing import Tuple
import multiprocessing as mp

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WrapperDataset(Dataset):
    """
    Wrap an existing dataset that returns (image, label) and apply a stat_transform
    to the image. Returns (transformed_image, label).
    """

    def __init__(self, ds: Dataset, stat_transform):
        self.ds = ds
        self.stat_transform = stat_transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x = self.ds[idx]
        # Expect (img, y). If your dataset returns dicts, adjust here.
        if isinstance(x, (tuple, list)) and len(x) >= 2:
            img, y = x[0], x[1]
        elif isinstance(x, dict) and "image" in x and "label" in x:
            img, y = x["image"], x["label"]
        else:
            raise TypeError(
                "Dataset must return (image, label) or a dict with keys 'image' and 'label'."
            )

        img = self.stat_transform(img)
        return img, y


@torch.no_grad()
def compute_mean_std(
    dataset: Dataset,
    img_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-channel mean/std on the given dataset.
    Assumes images are converted to tensors in [0,1] via ToTensor().

    Compatible with your call:
        compute_mean_std(train_subset, img_size=64, batch_size=256, num_workers=..., pin_memory=...)
    
    WINDOWS COMPATIBILITY:
    - Always uses num_workers=0 on Windows to avoid multiprocessing deadlocks
    - Uses num_workers parameter only on macOS/Linux
    """
    print(f"[compute_mean_std] Starting with dataset size: {len(dataset)}")

    # Resize ensures consistent H,W across different datasets.
    # The `antialias` kwarg was added in newer torchvision versions; fall back if unavailable.
    resize_steps = []
    if img_size is not None and img_size > 0:
        try:
            resize_steps.append(transforms.Resize((img_size, img_size), antialias=True))
        except TypeError:
            # older torchvision does not accept antialias
            resize_steps.append(transforms.Resize((img_size, img_size)))

    resize_steps.append(transforms.ToTensor())  # outputs float32 [0,1], shape [C,H,W]
    stat_transform = transforms.Compose(resize_steps)
    print("[compute_mean_std] Transforms created successfully")

    # CRITICAL: Windows uses different multiprocessing model - force single-threaded on Windows
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        # Windows: MUST use num_workers=0 to prevent spawn context deadlock
        actual_num_workers = 0
        actual_pin_memory = False
        print("[compute_mean_std] *** WINDOWS DETECTED *** - Using single-threaded DataLoader")
    else:
        # macOS/Linux: can use multiprocessing
        actual_num_workers = num_workers
        actual_pin_memory = bool(pin_memory) and torch.cuda.is_available()
        print(f"[compute_mean_std] Non-Windows system - Using num_workers={actual_num_workers}")

    print(f"[compute_mean_std] Creating WrapperDataset...")
    wrapped_dataset = WrapperDataset(dataset, stat_transform)
    print(f"[compute_mean_std] WrapperDataset created successfully")
    
    print(f"[compute_mean_std] Creating DataLoader with batch_size={batch_size}, num_workers={actual_num_workers}...")
    try:
        loader = DataLoader(
            wrapped_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=actual_num_workers,
            pin_memory=actual_pin_memory,
            drop_last=False,
            persistent_workers=False,
        )
        print("[compute_mean_std] DataLoader created successfully")
    except Exception as e:
        print(f"[compute_mean_std] ERROR creating DataLoader: {e}")
        raise

    channel_sum = None
    channel_sum_sq = None
    num_pixels = 0
    batch_count = 0

    print(f"[compute_mean_std] Starting iteration through DataLoader...")
    try:
        for x, _y in loader:
            batch_count += 1
            if batch_count % 10 == 0 or batch_count == 1:
                print(f"[compute_mean_std] Processing batch {batch_count}... shape={x.shape}")
            
            # x: [B,C,H,W]
            x = x.to(dtype=torch.float64)

            if x.ndim != 4:
                raise ValueError(f"Expected images with shape [B,C,H,W], got {tuple(x.shape)}")

            b, c, h, w = x.shape

            if channel_sum is None:
                channel_sum = torch.zeros(c, dtype=torch.float64)
                channel_sum_sq = torch.zeros(c, dtype=torch.float64)

            channel_sum += x.sum(dim=(0, 2, 3))
            channel_sum_sq += (x * x).sum(dim=(0, 2, 3))
            num_pixels += b * h * w
        
        print(f"[compute_mean_std] Finished iteration. Total batches: {batch_count}")
    except Exception as e:
        print(f"[compute_mean_std] ERROR during iteration: {e}")
        raise

    if channel_sum is None or num_pixels == 0:
        raise RuntimeError("Dataset is empty or no images were loaded; cannot compute mean/std.")

    mean = channel_sum / num_pixels
    var = (channel_sum_sq / num_pixels) - (mean * mean)
    var = torch.clamp(var, min=0.0)  # numerical safety
    std = torch.sqrt(var)

    print(f"[compute_mean_std] Computed mean: {mean.tolist()}")
    print(f"[compute_mean_std] Computed std: {std.tolist()}")

    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32)