import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from data.labels import CANONICAL_CLASSES
from data.transforms import get_train_transforms, get_val_transforms, get_test_transforms

class FER_Dataset(Dataset):
    def __init__(self, root_dir, set_type):
        self.root_dir = root_dir
        self.set_type = set_type

        if set_type == 'train':
            self.transform = get_train_transforms()
        elif set_type == 'val':
            self.transform = get_val_transforms()
        elif set_type == 'test':
            self.transform = get_test_transforms()
        else:
            raise ValueError("set_type must be 'train', 'val' or 'test'")

        self.classes_list = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_dict = {cls: i for i, cls in enumerate(self.classes_list)}

        self.samples = []
        for cls in self.classes_list:
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(
                        (os.path.join(cls_dir, img), self.class_dict[cls])
                    )
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. "
                "Check folder structure and file extensions."
            )
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class FER_DatasetFromSamples(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], set_type: str):
        self.samples = samples
        self.set_type = set_type
        self.classes_list = list(CANONICAL_CLASSES)
        if set_type == "train":
            self.transform = get_train_transforms()
        elif set_type == "val":
            self.transform = get_val_transforms()
        elif set_type == "test":
            self.transform = get_test_transforms()
        else:
            raise ValueError("set_type must be 'train', 'val' or 'test'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
