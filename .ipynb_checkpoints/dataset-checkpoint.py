import os
import shutil
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# use pre-computed mean and standard deviation from Tiny ImageNet training set
MEAN = [0.4802, 0.4481, 0.3975]
STD  = [0.2770, 0.2691, 0.2821]

# function: fix_val_folder
# convert file/folder structure to what PyTorch expects (only required for validation folder)
# build a dictionary to map each filename to its class ID

def fix_val_folder(data_dir="./tiny-imagenet-200/"):
    val_dir = os.path.join(data_dir, "val")
    val_images_dir = os.path.join(val_dir, "images")
    val_annot_txt = os.path.join(val_dir, "val_annotations.txt")

    val_img_to_class = {}
    with open(val_annot_txt) as f:
        for line in f:
            parts = line.strip().split('\t')
            val_img_to_class[parts[0]] = parts[1]

    moved = 0
    for img_file, class_id in val_img_to_class.items():
        src = os.path.join(val_images_dir, img_file)
        dst_dir = os.path.join(val_images_dir, class_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, img_file)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            moved += 1
            
    print(f'Moved {moved} images into class subdirectories.')

# function: get_dataloaders
# num_workers -> 2 CPUs load and preprocess data in parallel while GPU is training

def get_dataloaders(data_dir="./tiny-imagenet-200/", batch_size=128, num_workers=2):
    # augment training data:
    # RandomCrop -> pads the data with 8 pixels on each side and randomly crops back to 64x64 (introduces noise)
    # RandomHorizontalFlip -> 50% chance of image flipping, doubles data for the class
    # ColorJitter -> randomly changes brightness, contrast, and saturdation (increases robustness to noise)
    # ToTensor -> converts image to tensorfloat
    train_transforms = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val", "images"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset

# function: denormalise
# used to display the image (converts back from tensorfloat to image)

def denormalise(tensor, mean=mean, std=std):
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
        
    return t