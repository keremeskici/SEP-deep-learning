from __future__ import annotations

import platform
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# changed the mean and std calculation to make it windows approved as well, because some errors can appear due to num_workers not being zero

class WrapperDataset(Dataset):
    def __init__(self, ds: Dataset, stat_transform):
        self.ds = ds
        self.stat_transform = stat_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, y = self.ds[idx]
        img = self.stat_transform(img)
        return img, y


def _make_loader(dataset: Dataset, stat_transform, batch_size: int, num_workers: int, pin_memory: bool):
    # macOS: usually spawns as stable fork
    mp_ctx = "spawn" if platform.system() == "Darwin" else None

    return DataLoader(
        WrapperDataset(dataset, stat_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        multiprocessing_context=mp_ctx,
    )


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

    
    if platform.system() == "Darwin":
        pin_memory = False

    def run(loader: DataLoader):
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

    # 1) normal try (with num workers)
    loader = _make_loader(dataset, stat_transform, batch_size, num_workers, pin_memory)
    try:
        return run(loader)
    except RuntimeError as e:
        msg = str(e)
        # 2) special Mac/PyTorch-Error: torch_shm_manager not exceutable
        if "torch_shm_manager" in msg and "Permission denied" in msg:
            print(
                "[compute_mean_std] WARNING: torch_shm_manager permission issue detected.\n"
                "Fix (recommended): chmod +x <venv>/site-packages/torch/bin/torch_shm_manager\n"
                "Fallback: re-running mean/std with num_workers=0 just for stats."
            )
            loader = _make_loader(dataset, stat_transform, batch_size, num_workers=0, pin_memory=False)
            return run(loader) # num_workers are set to 0 in case this error appears (just loacally for mean_std)

        # raise different error
        raise