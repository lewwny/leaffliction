import json
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any


# consts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)


def load_model(path: str):
    """loads model from path"""
    model_path = Path(path)

    pth
    return


def get_transform() -> transforms.Compose:
    """returns transform pipeline for inference"""
    # imagenet normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE[0]),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])


def predict_image(model: nn.Module, image_path: str, classes: List[str],
                  transform: transforms.Compose):
    # load and transform image

    # run inference

    # predict

    # get probabilities for each class

    # return best probability
    return


def main() -> int:
    """makes predictions using a pre-created leaffliction model"""
    # argv for model dir and image(s)
    if len(sys.argv) < 3:
        print("Usage: python Predict.py <model_dir> <image_path> [image_path2 ...]")
        return 1

    model_dir = sys.argv[1]
    image_paths = sys.argv[2:]

    # check if model dir exists
    if not Path(model_dir).exists():
        print(f"Error: {model_dir} not found")
        return 1

    # try load model
    try:
        model, classes, metadata = load_model(model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # get transform

    # predict each image in each image path
    for image_path in image_paths:
        if not Path(image_path).exists():
            print(f"{image_path} not found")
            continue

        try:
            # predict here
            a = 1
        except FileNotFoundError as e:
            print(f"Error processing {image_path}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
