import json, zipfile, sys, shutil, cv2, os, hashlib, random, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from pathlib import Path
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split


# consts
DEVICE = torch.device("gpu" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 244)
B_SIZE = 32
EPOCHS = 10
VALID_SPLIT = 0.2
TARGET_ACC = 0.9


def print_device():
    """prints device type for processing"""
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU: Training will be slower")


def sort_data(folder: str):
    """sorts data from folder for train and validation imgs"""
    # get classes from source
    source = Path(folder)
    classes = [c.name for c in source.iterdir() if c.is_dir()]
    classes.sort()

    print(f"Classes Found: {len(classes)}")

    # counts images
    total_imgs = 0
    for i, c in enumerate(classes):
        img_count = (len(list(source / c).glob("*.[jJ][pP][gG]"))
                     + len(list(source / c).glob("*.[jJ][pP][eE][gG]"))
                     + len(list(source / c).glob("*.[pP][nN][gG]")))
        print(f"{i + 1}.{c}: {img_count} images")
        total_imgs += img_count

    print(f"Total images: {total_imgs}")

    # rmtree for the temp
    temp_folder = Path("temp_dataset")
    if temp_folder.exists():
        shutil.rmtree(temp_folder)

    # create train and validation directories
    train_dir = temp_folder / "train"
    valid_dir = temp_folder / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    # create subdirectories for classes in train/validation dirs
    for c in classes:
        (train_dir / c).mkdir(exist_ok=True)
        (valid_dir / c).mkdir(exist_ok=True)

        class_folder = source / c

        # get using image type (jpg, jpeg, png) capitalization insensitive
        imgs = ((list(source / c).glob("*.[jJ][pP][gG]"))
                     + (list(source / c).glob("*.[jJ][pP][eE][gG]"))
                     + (list(source / c).glob("*.[pP][nN][gG]")))

        # use sklearn train test split on images
        train_imgs, valid_imgs = train_test_split(imgs, test_size=VALID_SPLIT, random_state=40)

        # shutil copy all images in train and validation
        for img in train_imgs:
            shutil.copy(img, train_dir / c / img.name)
        for img in valid_imgs:
            shutil.copy(img, valid_dir / c / img.name)

        print(f"{c}: {len(train_imgs)}\ntrain, {len(valid_imgs)} validation")

    return train_dir, valid_dir, classes


def data_load(train_dir, valid_dir):
    """augmentation + normalization for training data"""
    # transforming data for augmentation and norm
    transforms_data = {
        "train": transforms.Compose([transforms.RandomResizedCrop(IMG_SIZE[0]),
                                     transforms.RandomHorizontalFlip(),
                                    """...."""]),

        "valid": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(IMG_SIZE[0]),
                                    """...."""]),
    }

    # builds datasets from imagefolder
    train_dataset = datasets.ImageFolder(train_dir, transforms_data["train"])
    valid_dataset = datasets.ImageFolder(train_dir, transforms_data["valid"])

    # create dataloader from datasets
    train_loader = DataLoader(train_dataset, batch_size=B_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=B_SIZE, shuffle=True, num_workers=4)

    return train_loader, valid_loader


def create_model(classes):
    """Creates a model using MobileNetV2"""
    # gets the mobilenetv2 model with default weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # freeze base layers
    for p in model.features.parameters():
        p.requires_grad = False

    # replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, classes)

    return model.to(DEVICE)


def train_model(model, train_loader, valid_loader):
    """trains model using train and validation loaders"""
    # get the optimizer for parameters of the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # uses torch to get crossentropyloss
    x_entropy_loss = nn.CrossEntropyLoss()

    # scheduler for the LR -> reduces on conversion plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # metrics to keep during training
    metrics = {"accuracy": [], "val_acc": [], "loss": [], "val_loss": []}

    # counter vars for training
    max_model = copy.deepcopy(model.state_dict())
    max_acc = 0.0
    count_patience = 0
    max_patience = 7
    start_time = time.time()

    # get total passes count
    total_passes = EPOCHS * (len(train_loader) + len(valid_loader))

    # progress bar using tqdm
    progress = tqdm(total=total_passes, unit="step", leave=True)

    for e in range (EPOCHS):
        for phase in ["train", "valid"]:
            if phase == "train":
                dataloader = train_loader
                model.train()
            else:
                dataloader = valid_loader
                model.eval()
            
            loss = 0.0
            







def main() -> int:
    # argv for image folder
    if len(sys.argv) < 2:
        print("Usage: python Train.py <img_folder>")
        return 1

    # check if folder exists
    img_folder = sys.argv[1]
    if not Path(img_folder).exists():
        print(f"Error: {img_folder} doesn't exist")
        return 1

    # show device used for training (either gpu or cpu)
    print_device()

    return 0

if __name__ == "__main__":
    main()
