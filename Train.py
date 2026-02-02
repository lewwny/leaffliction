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
        