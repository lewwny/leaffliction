import numpy as np
import sys
import argparse
import cv2 
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from load_image import ft_load
from Augmentation import is_valide_image, show_image

def show_image_greyscale(image, title="Greyscale Image"):
    """Displays a 2D image (mask) in greyscale."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image_color(image, title="Color Image"):
    """Displays a 3D image (color)."""
    plt.figure(figsize=(6, 6))
    if hasattr(image, 'ndim') and image.ndim == 3:
        plt.imshow(image) 
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def check_plantcv_output(output):
    """Extracts the image if PlantCV returns a tuple."""
    if isinstance(output, tuple):
        return output[0]
    return output

def transform_image(image: np.ndarray) -> None:
    pcv.params.debug = None 

    show_image_color(image, "0. Original Image")

    s_channel = pcv.rgb2gray_hsv(image, 's')
    s_blurred = pcv.gaussian_blur(s_channel, (11,11))
    
    mask = pcv.threshold.binary(gray_img=s_blurred, threshold=85, object_type='light')
    mask_clean = pcv.fill(mask, 200)
    
    show_image_greyscale(mask_clean, "1. Binary Mask")
    
    masked_image = pcv.apply_mask(image, mask_clean, 'white')
    show_image_color(masked_image, "2. Masked Image")

    h, w = image.shape[:2]
    roi_contour = pcv.roi.rectangle(img=image, x=0, y=0, h=h, w=w)
    kept_mask = pcv.roi.filter(mask=mask_clean, roi=roi_contour, roi_type='partial')
    
    shape_result = pcv.analyze.size(img=image, labeled_mask=kept_mask)
    shape_img = check_plantcv_output(shape_result)
    show_image_color(shape_img, "3. Shape Analysis")
    
    try:
        results = pcv.homology.x_axis_pseudolandmarks(img=image, mask=kept_mask)
        
        found_image = None
        found_points = None

        if isinstance(results, tuple):
            for item in results:
                if isinstance(item, np.ndarray):
                    if item.ndim == 3 and item.shape[2] == 3:
                        found_image = item
                    elif item.ndim == 3 and item.shape[2] == 2:
                        found_points = item
                    elif item.ndim == 2 and item.shape[1] == 2:
                        found_points = item

        if found_image is not None:
            show_image_color(found_image, "4. Pseudolandmarks (Generated)")
            
        elif found_points is not None:
            img_with_points = image.copy()
            
            for point in found_points:
                x, y = point[0] if len(point) == 1 else point
                cv2.circle(img_with_points, (int(x), int(y)), 10, (255, 0, 0), -1)
                
            show_image_color(img_with_points, "4. Pseudolandmarks (Manual Draw)")
            
        else:
            print("Warning: Could not find valid image or points in results.")

    except Exception as e:
        print(f"Error processing landmarks: {e}")

def main():
    try:
        parser = argparse.ArgumentParser(description="PlantCV Pipeline Transformation")
        parser.add_argument("image_path", type=str, help="Path to the input image file.")
        args = parser.parse_args()
        
        if not is_valide_image(args.image_path):
            raise ValueError("The provided file is not a valid image.")
            
        image = ft_load(args.image_path)
        if image is None:
            raise ValueError("Failed to load the image.")
            
        transform_image(image)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()