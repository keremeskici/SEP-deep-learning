
import torch
from torchvision import transforms

# To mitigate overfitting we split the data into training, validation and test set, so the model can be evaluated on unseen data during and after tarining and futhermore we apply data augmentation techniques so the model dosen´t only memorize the training data

# for testing I chose ImageNet mean and std
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

def get_train_transforms(): # Data augmentation for the training set
    return transforms.Compose([ 
        transforms.Resize((64,64)),
# like proposed in the preliminary report we chose to add random horizontal flip, rotation and color jittering as data augmentation techniques, to increase the diversity of the tarining data
        transforms.RandomHorizontalFlip(p=0.5), # flip every second image horizontally
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,), # small values since facial images should not be heavily altered 
        transforms.ToTensor(),
        transforms.Normalize(mean, std), # TODO: define mean and std, since we didn´t choose a training data set yet, we can´t compute them right now (maybe just try with imagenet values for now)
    ])

def get_val_transforms(): # no data augmentation for the validation test set
    return transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def get_test_transforms(): # no data augmentation for the test set either
    return transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])  