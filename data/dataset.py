# dataset is used to load data from png/jpg files and hand over single samples to the dataloader (that creates batches) after being transformed by the transforms defined in transforms.py

# os module is used to handle file paths since the dataset consists of image files stored in a directory

# PIL is used to open images files since they will be handed over as PIL images to the transforms defined in transforms.py

# I use the implemented interface of torch.util.data.Dataset that my costum dataset class FER_Dataset inherits from

import os
from PIL import Image
import torch    
from torch.utils.data import Dataset
from data.transforms import get_train_transforms, get_val_transforms, get_test_transforms

class FER_Dataset(Dataset):
    def __init__(self, root_dir, set_type): # root_dir and set_type need to be provided when initializing the dataset
        self.root_dir = root_dir # Path to the root directory of the dataset
        self.set_type = set_type # 'train', 'val' or 'test'

# Since they're only 3 data sets (train, val, test) I chose to handle the selection of the correct transforms directly in the dataset class 

        if set_type == 'train':
            self.transform = get_train_transforms()
        elif set_type == 'val':
            self.transform = get_val_transforms()
        elif set_type == 'test':
            self.transform = get_test_transforms()
        else:
            raise ValueError("set_type must be 'train', 'val' or 'test'")

        # Subfolder that represent labels/classes in root_dir
        self.classes_list = sorted([ # list that returns all subfolder names in root_dir and sorts them alphabetically
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d)) # in MacOs sometimes hidden files/folders are created that are not directories, so we check if it is a directory
        ])
        self.class_dict = {cls: i for i, cls in enumerate(self.classes_list)} # turn the list into a dictionary with class names as keys and their corresponding indices as values

        # the same as: 
        # self.class_dict = {} # empty dictionary
        # for i, cls in enumerate(self.classes_list): # every values cls at i in classes_list
            # self.class_dict[cls] = i # assign cls as key and i as value in the dictionary

       # Create a list of (image_path, label) tuples
        self.samples = [] # empty list to store image paths and their corresponding labels
        for cls in self.classes_list: # iterate through every class in the classes_list
            cls_dir = os.path.join(root_dir, cls) # root_dir + class name = path to the class directory
            for img in os.listdir(cls_dir): # iterate through every image in the class directory
                if img.lower().endswith(('.jpg', '.jpeg', '.png')): # check if the file is an image
                    self.samples.append( # add a tuple of (image_path, label) to the samples list
                        (os.path.join(cls_dir, img), self.class_dict[cls]) # image path and corresponding label as number
                    )
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. "
                "Check folder structure and file extensions."
            )
        
    def __len__(self): # always in the interface needs to be implemented
        return len(self.samples)

    def __getitem__(self, idx): # always in the interface needs to be implemented
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform: # apply the transforms defined in transforms.py
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long) # since cross entropy loss needs the target labels as long tensors


