import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

class FER_SamplesPILDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, torch.tensor(label, dtype=torch.long)

