

# there are 2 ways to initialize the dataset and dataloader objects:
# 1. directly in a script (like in test_dataset.py)
# 2. using the get_dataloaders function defined in dataloader.py (recommended way)

# Example for 1:
# the dataloader can be initialized differently for every dataset (train, val, test), so different batch sizes or shuffling can be applied if needed

# from torch.utils.data import DataLoader
# from data.dataset import FER_Dataset
# train_ds = FER_Dataset("data/training", "train")
# train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)

# Example for 2:
# all data loaders get initialized with a single function call, therfore with the same values

#from data.dataloader import get_dataloaders
# train_loader, val_loader, test_loader = get_dataloaders(
    #data_root="data",
    #batch_size=64,
    #num_workers=4,
    #pin_memory=True,
    #seed=42
#)

# once initialized, the dataloader can be used in a training loop like this:
    #model.train()
    # for images, labels in train_loader:

# tests
# to start the test use: cd <PROJECT_ROOT>
                        # python -m tests.test_data_loader


import os, sys, random
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data.dataloader
print("USING dataloader file:", data.dataloader.__file__)
from data.dataloader import get_dataloaders


def main():
    print("IN MAIN ✅")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=4,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    images, labels = next(iter(train_loader))
    print("OK batch:", images.shape, labels)

    # Stimmen die Labels mit den Klassen überein?
    train_ds = train_loader.dataset
    print(train_ds.classes_list)
    print(train_ds.class_dict)

    # Zeige 5 zufällige Samples
    for i in random.sample(range(len(train_ds)), k=min(5, len(train_ds))):
        img, label = train_ds[i]
        print(label.item(), train_ds.classes_list[label.item()])

    # zwei Epochen vergleichen (Shuffle)
    batch1 = [lab for _, lab in train_loader]
    batch2 = [lab for _, lab in train_loader]
    print(batch1[0][:10])
    print(batch2[0][:10])

    # ist val nicht geshuffled?
    v1 = [lab for _, lab in val_loader]
    v2 = [lab for _, lab in val_loader]
    print(torch.equal(v1[0], v2[0]))

    # ist augmentation nur aktiv in train?
    img1, _ = train_ds[0]
    img2, _ = train_ds[0]
    print(torch.equal(img1, img2))  # meistens False

    img1, _ = val_loader.dataset[0]
    img2, _ = val_loader.dataset[0]
    print(torch.equal(img1, img2))  # True

    # mehrere Worker testen (WICHTIG: macOS nur im __main__ Kontext!)
    train_loader2, _, _ = get_dataloaders(num_workers=4)
    images2, labels2 = next(iter(train_loader2))
    print("workers batch:", images2.shape)


if __name__ == "__main__":
    print("IN __MAIN__ ✅")
    main()

# test results:
#IN __MAIN__ ✅
#IN MAIN ✅
#OK batch: torch.Size([4, 3, 64, 64]) tensor([4, 5, 0, 1])
#['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
#{'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5}
#3 happiness
#5 surprise
#2 fear
#4 sadness
#0 anger
#tensor([5, 4, 1, 0])
#tensor([2, 4, 5, 0])
#True
#False
#True
