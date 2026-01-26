
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.transforms import get_train_transforms, get_val_transforms, get_test_transforms

# since I wasn´t sure if we need to code the dataloader by ourself or use the built in from pytorch, this file would be the easier version using the built in one

# if this file is used, dataset and dataloader are no longer nessesary to be implemented by ourself
# the only file that would be needed is transforms.py to define the transformations/augmentations
# I think for organisational purposes it would be better to have a separate file for tranformation and hard code them, rather than having them in the config file

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(config):
    """
    depending on the config file, different hyperparameters can be included, such as:
    - data_root
    - batch_size
    - num_workers
    - pin_memory
    - persistent_workers
    - seed
    - (optional) img_size (probably hard coded in transforms.py, but can be overwritten here)
    """
    data_root = config.data_root # path to the root directory of the dataset

    # Since we didn´t discuss wether to take the augmentation data form config or not, I decided to implement both for now
    if hasattr(config, "img_size"): # in case config provides an image size, use it to initialize the transforms
        train_tf = get_train_transforms(config.img_size)
        val_tf   = get_val_transforms(config.img_size)
        test_tf  = get_test_transforms(config.img_size)
    else: # otherwise use the default image size defined in transforms.py
        train_tf = get_train_transforms()
        val_tf   = get_val_transforms()
        test_tf  = get_test_transforms()

    train_dir = f"{data_root}/training" # path to the training data directory
    val_dir   = f"{data_root}/validation" # path to the validation data directory
    test_dir  = f"{data_root}/test" # path to the test data directory

    train_set = ImageFolder(root=train_dir, transform=train_tf) 
    val_set   = ImageFolder(root=val_dir, transform=val_tf)
    test_set  = ImageFolder(root=test_dir, transform=test_tf)

    g = torch.Generator()
    g.manual_seed(config.seed)

# Synthax sugar, thats not necessary but avoids code duplication instead of initializing each dataloader with the same parameters
    constructor_variables = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(config.persistent_workers and config.num_workers > 0),
    )
# DataLoader for each dataset

# train_loader is the only one with shuffle=True, for mixed ordered batches during training
    train_loader = DataLoader(
        train_set,
        shuffle=True,          
        **constructor_variables # uses the parameters defined in common_loader_kwargs
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,         
        **constructor_variables
    )

    test_loader = DataLoader(
        test_set,
        shuffle=False,         
        **constructor_variables
    )

    return train_loader, val_loader, test_loader