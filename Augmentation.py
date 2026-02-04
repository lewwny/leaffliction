import numpy as np
import sys
from load_image import ft_load
import os
import matplotlib.pyplot as plt
import argparse
from Distribution import analyze_directory
from PIL import Image, UnidentifiedImageError, ImageEnhance, ImageFilter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    zoomed_img = image_data[start_row:start_row +
                            new_height, start_col:start_col + new_width]
    return np.array(Image.fromarray(zoomed_img).resize((width, height)))


def augment_blur(image_data, blur_radius=2):
    """Apply Gaussian blur to the image."""
    pil_img = Image.fromarray(image_data)
    blurred_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))
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


def save_augmented_image(image_data, original_path, augmentation_type):
    """Save the augmented image to disk."""
    new_folder = "augmented_directory"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    new_file_name = f"{name}_{augmentation_type}{ext}"
    new_file_path = os.path.join(new_folder, new_file_name)
    pil_img = Image.fromarray(image_data)
    pil_img.save(new_file_path)
    print(f"Saved augmented image: {new_file_path}")


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
        # show_image(augmented_image)
        save_augmented_image(augmented_image, original_path, aug_name)


def upgrade_data(data, directory_path):
    """Augment images in the directory to balance the dataset."""
    max_images = max(data.values())
    use_flag = False
    for class_names, num_images in data.items():
        class_folder = os.path.join(directory_path, class_names)
        images_needed = max_images - num_images
        if images_needed < 6:
            continue
        image_files = [f for f in os.listdir(class_folder)
                       if is_valide_image(os.path.join(class_folder, f))]
        augmentations = [augment_flip, augment_rotate, augment_zoom,
                         augment_blur, augment_contrast, augment_illumination]
        aug_index = 0
        while images_needed > 0:
            for image_file in image_files:
                if images_needed <= 0:
                    break
                image_path = os.path.join(class_folder, image_file)
                image_data = ft_load(image_path)
                augmented_image = augmentations[aug_index %
                                                len(augmentations)](image_data)
                pil_img = Image.fromarray(augmented_image)
                new_file_name = (
                    f"{os.path.splitext(image_file)[0]}_aug{aug_index}"
                    f"{os.path.splitext(image_file)[1]}"
                )
                pil_img.save(os.path.join(class_folder, new_file_name))
                print(f"Saved augmented image: {new_file_name} "
                      f"in {class_folder}")
                images_needed -= 1
                aug_index += 1
        


def main():
    """main function for Augmentation.py"""
    try:
        parser = argparse.ArgumentParser(description=""
                                         "Image Augmentation Script")
        parser.add_argument("image_path", nargs='?',
                            help="Path to a single image file")
        parser.add_argument("--augment", type=str, metavar="DIRECTORY",
                            help="Path to a directory to balance (augment)")
        args = parser.parse_args()
        if args.augment:
            if not os.path.isdir(args.augment):
                raise NotADirectoryError(f"The directory "
                                         f"{args.augment} does not exist.")
            directory_path = args.augment
            data = analyze_directory(directory_path)
            if not data:
                raise ValueError("No valid images found in "
                                 "the specified directory.")
            upgrade_data(data, directory_path)
        elif args.image_path:
            data_image_path = args.image_path
            if not os.path.isfile(data_image_path):
                raise FileNotFoundError(f"The file "
                                        f"{data_image_path} does not exist.")
            if not is_valide_image(data_image_path):
                raise ValueError(f"The file {data_image_path} "
                                 "is not a valid image.")
            print(f"The file {data_image_path} is a valid image.")
            data_image = ft_load(data_image_path)
            augmentation_image(data_image, data_image_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
