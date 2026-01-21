
# dataloader is used to create batches from the dataset and to load them during training/validation/testing
# for the dataloader implementation I use the DataLoader class from torch.utils.data
# a worker in Pytorch is a predifined subprocess that is used to load data in parallel to the main process, which can significantly speed up the data loading process, especially when dealing with large datasets or complex data transformations


from __future__ import annotations

import os
import random
from typing import Tuple, Optional
import numpy as np
import torch

from torch.utils.data import DataLoader
from data.dataset import FER_Dataset

# Method that gives each worker its own deterministic seed derived from the global seed
def seed_worker(worker_id: int) -> None:
    """
    a seed is nessary to ensure reproducibility when using multiple workers in the dataloader
    if a seed is set, every worker will produce the same sequence of random numbers, which can lead to a fixed order of data augmentation transformations being applied to the data
    """
    worker_seed = torch.initial_seed() % 2**32 # worker specific seed 
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    data_root: str = "data",
    batch_size: int = 64, # can be changed depending on the available hardware later on
    num_workers: int = 4, # number of subprocesses to use for data loading, can also be adjusted depending on the hardware
    pin_memory: bool = True, 
    persistent_workers: bool = True, # keep workers alive between epochs to reduce overhead
    seed: int = 42, #global seed (can also be changed, depending on the perfomance when training the model)
) -> Tuple[DataLoader, DataLoader, DataLoader]: 
   
    train_dir = os.path.join(data_root, "training") # path to the training data directory
    val_dir = os.path.join(data_root, "validation") # path to the validation data directory
    test_dir = os.path.join(data_root, "test") # path to the test data directory

    train_ds = FER_Dataset(root_dir=train_dir, set_type="train") # create dataset for training set
    val_ds = FER_Dataset(root_dir=val_dir, set_type="val") # create dataset for validation set
    test_ds = FER_Dataset(root_dir=test_dir, set_type="test") # create dataset for test set

    
    g = torch.Generator() # create a generator for random number generation
    g.manual_seed(seed) #

    train_loader = DataLoader( # DataLoader Constructor for the training set
        train_ds, 
        batch_size=batch_size, # size of each batch can be adjusted depending on the available hardware
        shuffle=True, # shuffle changes the order of the data at every epoch 
        num_workers=num_workers, # subprocesses to load data more efficiently in parallel
        pin_memory=pin_memory, # pin_memory can improve performance when transferring data to GPU
        worker_init_fn=seed_worker, # function to initialize each worker with a seed for reproducibility
        generator=g, # generator with a fixed seed for reproducibility
        persistent_workers=persistent_workers and num_workers > 0, # keep workers alive between epochs to reduce overhead
    )

    val_loader = DataLoader( # DataLoader Constructor for the validation set
        val_ds,
        batch_size=batch_size,
        shuffle=False, # do nut shuffle the validation data, otherwise the evaluation results would not be consistent
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0, 
    )

    test_loader = DataLoader( # DataLoader Constructor for the test set
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    return train_loader, val_loader, test_loader
