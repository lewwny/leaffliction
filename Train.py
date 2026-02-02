import json, zipfile, sys, shutil, cv2, os, hashlib, random, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
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
    source = Path(folder)
    classes = [c.name for c in source.iterdir() if c.is_dir()]
    classes.sort()

    print(f"Classes Found: {len(classes)}")

    total_imgs = 0
    for i, c in enumerate(classes):
        img_count = (len(list(source / c).glob("*.[jJ][pP][gG]"))
                     + len(list(source / c).glob("*.[jJ][pP][eE][gG]"))
                     + len(list(source / c).glob("*.[pP][nN][gG]")))
        print(f"{i + 1}.{c}: {img_count} images")
        total_imgs += img_count

    print(f"Total images: {total_imgs}")

    temp_folder = Path("temp_dataset")
    if temp_folder.exists():
        shutil.rmtree(temp_folder)

    train_dir = temp_folder / "train"
    validation_dir = temp_folder / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    for c in classes:
        (train_dir / c).mkdir(exist_ok=True)
        (validation_dir / c).mkdir(exist_ok=True)

        class_folder = source / c

        imgs = ((list(source / c).glob("*.[jJ][pP][gG]"))
                     + (list(source / c).glob("*.[jJ][pP][eE][gG]"))
                     + (list(source / c).glob("*.[pP][nN][gG]")))

        train_imgs, valid_imgs = train_test_split(imgs, test_size=VALID_SPLIT)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python Train.py <img_folder>")
        return 1

    img_folder = sys.argv[1]
    if not Path(img_folder).exists():
        print(f"Error: {img_folder} doesn't exist")
        return 1

    print_method()


if __name__ == "__main__":
    main()
