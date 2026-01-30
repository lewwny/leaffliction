import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

def is_valide_image(file_path):
    """Check if the file at file_path is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, IOError):
        return False

def main():
    """main function for Augmentation.py"""
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: python Augmentation.py <data_directory>")
        data_image_path = sys.argv[1]
        if not os.path.isfile(data_image_path):
            raise FileNotFoundError(f"The file {data_image_path} does not exist.")
        if not is_valide_image(data_image_path):
            raise ValueError(f"The file {data_image_path} is not a valid image.")
        print(f"The file {data_image_path} is a valid image.")
        data_image = Image.open(data_image_path)
        print(f"Image format: {data_image.format}")
        print(f"Image size: {data_image.size}")
        print(f"Image mode: {data_image.mode}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()