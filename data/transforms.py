
import torch
from torchvision import transforms

# To mitigate overfitting we split the data into training, validation and test set, so the model can be evaluated on unseen data during and after tarining and futhermore we apply data augmentation techniques so the model dosen´t only memorize the training data

def get_train_transforms(mean, std):
    return transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2,contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def get_val_transforms(mean, std):
    return transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def get_test_transforms(mean, std):
    return transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])