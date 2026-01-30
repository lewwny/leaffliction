import numpy as np
import sys
from load_image import ft_load
import os
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError, ImageEnhance, ImageFilter

def is_valide_image(file_path):
    """Check if the file at file_path is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, IOError):
        return False

def show_image(image_data):
    """Display the image represented by image_data."""
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()

def augment_flip(image_data):
    """Flip the image horizontally."""
    return np.fliplr(image_data)

def augment_rotate(image_data, angle=45):
    """Rotate the image by the given angle."""
    pil_img = Image.fromarray(image_data)
    rotated_img = pil_img.rotate(angle)
    return np.array(rotated_img)

def augment_zoom(image_data, zoom_factor=1.2):
    """Zoom into the image by the given zoom factor."""
    height, width = image_data.shape[:2]
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    start_row, start_col = (height - new_height) // 2, (width - new_width) // 2
    zoomed_img = image_data[start_row:start_row + new_height, start_col:start_col + new_width]
    return np.array(Image.fromarray(zoomed_img).resize((width, height)))

def augment_blur(image_data, blur_radius=2):
    """Apply Gaussian blur to the image."""
    pil_img = Image.fromarray(image_data)
    blurred_img = pil_img.filter(Image.Filter.GaussianBlur(blur_radius))
    return np.array(blurred_img)

def augment_contrast(image_data, factor=1.5):
    """Adjust the contrast of the image."""
    pil_img = Image.fromarray(image_data)
    enhancer = ImageEnhance.Contrast(pil_img)
    contrasted_img = enhancer.enhance(factor)
    return np.array(contrasted_img)

def augment_illumination(image_data, factor=1.2):
    """Adjust the illumination of the image."""
    pil_img = Image.fromarray(image_data)
    enhancer = ImageEnhance.Brightness(pil_img)
    illuminated_img = enhancer.enhance(factor)
    return np.array(illuminated_img)

def augmentation_image(image_data, original_path):
    """Placeholder function for image augmentation."""
    augmentations = {
        "flip": augment_flip,
        "rotate": augment_rotate,
        "zoom": augment_zoom,
        "blur": augment_blur,
        "contrast": augment_contrast,
        "illumination": augment_illumination
    }
    for aug_name, aug_func in augmentations.items():
        augmented_image = aug_func(image_data)
        show_image(augmented_image)
        # save_augmented_image(augmented_image, original_path, aug_name)

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
        data_image = ft_load(data_image_path)
        augmentation_image(data_image, data_image_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()