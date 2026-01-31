import numpy as np
from load_image import ft_load
import sys
import os
import matplotlib.pyplot as plt
import argparse
from Distribution import analyze_directory
from plantcv import plantcv as pcv
from PIL import Image, UnidentifiedImageError, ImageEnhance, ImageFilter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
	"""main function for Transformation.py"""
	parser = argparse.ArgumentParser(
		description="Perform image transformations using PlantCV.")
	parser.add_argument("image_path", type=str,
						help="Path to the input image file.")
	args = parser.parse_args()
	image_path = args.image_path


if __name__ == "__main__":
	main()