from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# to keep everything tidy the mean and std calculation function is kept seperately in this file and can be changed to our liking

@torch.no_grad()
def compute_mean_std(
    dataset: Dataset,
    img_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    stat_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    class _Wrapper(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, y = self.ds[idx]
            img = stat_transform(img)
            return img, y

    loader = DataLoader(
        _Wrapper(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    num_pixels = 0

    for x, _ in loader:
        x = x.to(dtype=torch.float64)
        b, c, h, w = x.shape
        channel_sum += x.sum(dim=(0, 2, 3))
        channel_sum_sq += (x * x).sum(dim=(0, 2, 3))
        num_pixels += b * h * w

    mean = channel_sum / num_pixels
    var = channel_sum_sq / num_pixels - mean * mean
    std = torch.sqrt(var.clamp_min(1e-12))

    return mean.float(), std.float()