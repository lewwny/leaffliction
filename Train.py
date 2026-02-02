import json, zipfile, sys, shutil, cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path


# consts
METHOD = torch.device("gpu" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 244)
B_SIZE = 32
EPOCHS = 10
VALID_SPLIT = 0.2
TARGET_ACC = 0.9


def print_method():
    if METHOD.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU: Training will be slower")


def organize_dataset(folder: str):
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
