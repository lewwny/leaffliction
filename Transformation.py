import numpy as np
import sys
import argparse
import cv2 
import rembg
import os
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from load_image import ft_load
from Augmentation import is_valide_image

def is_in_circle(x, y, center_x, center_y, radius):
    if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
        return True
    return False

def plot_stat_hist(label, sc=1):
    # Using 'default' because label="default" is passed to analyze.color
    try:
        observations = pcv.outputs.observations['default']
        # Depending on PlantCV version, structure might differ slightly
        if 'color_analysis' in observations:
            data = observations['color_analysis'][label]
        else:
            data = observations[label]
            
        y = data['histogram']
        x = [i * sc for i in range(len(y))]
        
        if label == "hue_frequencies":
            x = x[:int(255 / 2)]
            y = y[:int(255 / 2)]
        if label == "blue-yellow_frequencies" or label == "green-magenta_frequencies":
            x = [val + 128 for val in x]
            
        plt.plot(x, y, label=label)
    except KeyError:
        pass # Skip missing labels safely

def is_roi_border(x, y, roi_start_x, roi_start_y, roi_h, roi_w, roi_line_w):
    return (
        (roi_start_x <= x <= roi_start_x + roi_w and roi_start_y <= y <= roi_start_y + roi_line_w) or
        (roi_start_x <= x <= roi_start_x + roi_w and roi_start_y + roi_h - roi_line_w <= y <= roi_start_y + roi_h) or
        (roi_start_x <= x <= roi_start_x + roi_line_w and roi_start_y <= y <= roi_start_y + roi_h) or
        (roi_start_x + roi_w - roi_line_w <= x <= roi_start_x + roi_w and roi_start_y <= y <= roi_start_y + roi_h)
    )

def create_roi_image(image, masked, filled):
    roi_start_x = 0
    roi_start_y = 0
    roi_w = image.shape[0]
    roi_h = image.shape[1]
    roi_line_w = 5

    roi = pcv.roi.rectangle(img=masked, x=roi_start_x, y=roi_start_y, w=roi_w, h=roi_h)
    kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type='partial')

    roi_image = image.copy()
    roi_image[kept_mask != 0] = (0, 255, 0)
    
    # Simple border drawing (could be optimized with cv2.rectangle)
    cv2.rectangle(roi_image, (roi_start_y, roi_start_x), (roi_start_y + roi_h, roi_start_x + roi_w), (255, 0, 0), roi_line_w)

    return roi_image, kept_mask

def plot_histogram(image, kept_mask, save_path=None):
    dict_label = {
        "blue": 1,
        "green": 1,
        "green-magenta": 1,
        "lightness": 2.55,
        "red": 1,
        "blue-yellow": 1,
        "hue": 1,
        "saturation": 2.55,
        "value": 2.55
    }

    # Ensure mask is labeled
    if np.max(kept_mask) > 0:
        labels_mask, _ = pcv.create_labels(mask=kept_mask)
    else:
        labels_mask = kept_mask

    pcv.outputs.clear()
    pcv.analyze.color(rgb_img=image, colorspaces="all", labeled_mask=labels_mask, label="default")

    plt.figure(figsize=(16, 9))
    for key, val in dict_label.items():
        plot_stat_hist(key, val)

    plt.legend()
    plt.title("Color Histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels")
    plt.grid(visible=True, which='major', axis='both', linestyle='--')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def draw_pseudolandmarks(image, pseudolandmarks, color, radius):
    if pseudolandmarks is None:
        return image
        
    for i in range(len(pseudolandmarks)):
        point = None
        # Handle different output formats from PlantCV versions
        if isinstance(pseudolandmarks[i], np.ndarray):
            point = pseudolandmarks[i].flatten()
        elif isinstance(pseudolandmarks[i], (list, tuple)):
            point = pseudolandmarks[i]
            
        if point is not None and len(point) >= 2:
            center_x = int(point[1])
            center_y = int(point[0])
            cv2.circle(image, (center_y, center_x), radius, color, -1)
    return image

def create_pseudolandmarks_image(image, kept_mask):
    pseudolandmarks = image.copy()
    try:
        top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(
            img=pseudolandmarks, mask=kept_mask, label='default'
        )
        pseudolandmarks = draw_pseudolandmarks(pseudolandmarks, top_x, (0, 0, 255), 5)
        pseudolandmarks = draw_pseudolandmarks(pseudolandmarks, bottom_x, (255, 0, 255), 5)
        pseudolandmarks = draw_pseudolandmarks(pseudolandmarks, center_v_x, (255, 0, 0), 5)
    except Exception:
        pass # Handle cases where landmarks cannot be found
    return pseudolandmarks

def process_single_image(image_path, dst_folder=None):
    # Load and process
    image_bgr = ft_load(image_path)
    if image_bgr is None:
        return
        
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Remove background
    image_without_bg = rembg.remove(image)

    # Pre-processing
    l_grayscale = pcv.rgb2gray_lab(rgb_img=image_without_bg, channel='l')
    l_thresh = pcv.threshold.binary(gray_img=l_grayscale, threshold=35, object_type='light')
    filled = pcv.fill(bin_img=l_thresh, size=200)
    gaussian_blur = pcv.gaussian_blur(img=filled, ksize=(3, 3))
    masked = pcv.apply_mask(img=image, mask=gaussian_blur, mask_color='black')

    # Features
    roi_image, kept_mask = create_roi_image(image, masked, filled)
    
    analysis_image = pcv.analyze.size(img=image, labeled_mask=kept_mask)
    if not isinstance(analysis_image, np.ndarray):
         analysis_image = image # Fallback if analyze.size returns non-image object

    pseudolandmarks = create_pseudolandmarks_image(image, kept_mask)
    pseudowithoutbg = create_pseudolandmarks_image(masked, kept_mask)

    # Doublewithoutbg: concatenation horizontale de Mask + Pseudowithoutbg (utilisé pour le training)
    doublewithoutbg = np.concatenate((masked, pseudowithoutbg), axis=1)

    images = {
        "Original": image,
        "Gaussian_blur": cv2.cvtColor(gaussian_blur, cv2.COLOR_GRAY2RGB),
        "Mask": masked,
        "ROI_Objects": roi_image,
        "Analyze_object": analysis_image,
        "Pseudolandmarks": pseudolandmarks,
        "Pseudowithoutbg": pseudowithoutbg,
        "Doublewithoutbg": doublewithoutbg
    }

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if dst_folder:
        # Save mode
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
            
        for label, img in images.items():
            # Convert back to BGR for OpenCV saving
            save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(dst_folder, f"{base_name}_{label}.jpg")
            cv2.imwrite(save_path, save_img)
        
        # Save Histogram
        plot_histogram(image, kept_mask, save_path=os.path.join(dst_folder, f"{base_name}_histogram.png"))
        print(f"Processed and saved: {base_name}")

    else:
        # Display mode
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(16, 9))
        fig.suptitle(f"Transformation of {image_path}")
        
        display_keys = ["Original", "Gaussian_blur", "Mask", "ROI_Objects", "Analyze_object", "Pseudolandmarks"]
        
        for i, key in enumerate(display_keys):
            if key in images:
                ax.flat[i].imshow(images[key])
                ax.flat[i].set_title(key)
                ax.flat[i].axis('off')
                
        plt.show()
        plt.close()
        plot_histogram(image, kept_mask)

def process_doublewithoutbg(image_path, dst_folder):
    """Process image and save only Doublewithoutbg (for training)."""
    image_bgr = ft_load(image_path)
    if image_bgr is None:
        return
        
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Remove background
    image_without_bg = rembg.remove(image)

    # Pre-processing
    l_grayscale = pcv.rgb2gray_lab(rgb_img=image_without_bg, channel='l')
    l_thresh = pcv.threshold.binary(gray_img=l_grayscale, threshold=35, object_type='light')
    filled = pcv.fill(bin_img=l_thresh, size=200)
    gaussian_blur = pcv.gaussian_blur(img=filled, ksize=(3, 3))
    masked = pcv.apply_mask(img=image, mask=gaussian_blur, mask_color='black')

    # ROI for pseudolandmarks
    roi = pcv.roi.rectangle(img=masked, x=0, y=0, w=image.shape[0], h=image.shape[1])
    kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type='partial')

    # Pseudolandmarks on masked image
    pseudowithoutbg = create_pseudolandmarks_image(masked, kept_mask)

    # Doublewithoutbg: concatenation horizontale de Mask + Pseudowithoutbg
    doublewithoutbg = np.concatenate((masked, pseudowithoutbg), axis=1)

    # Save
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_img = cv2.cvtColor(doublewithoutbg, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(dst_folder, f"{base_name}_Doublewithoutbg.jpg")
    cv2.imwrite(save_path, save_img)
    print(f"Saved: {save_path}")


def main():
    try:
        parser = argparse.ArgumentParser(description="PlantCV Pipeline Transformation")
        parser.add_argument("image_path", type=str, nargs='?', help="Path to the input image file.")
        parser.add_argument("-src", type=str, help="Source directory for images.")
        parser.add_argument("-dst", type=str, help="Destination directory for output.")
        parser.add_argument("--double", action="store_true", help="Save only Doublewithoutbg images (for training).")
        args = parser.parse_args()
        
        # Batch Mode
        if args.src:
            if not args.dst:
                raise ValueError("Destination directory (-dst) is required when using source directory (-src).")
            
            if not os.path.exists(args.src):
                raise ValueError(f"Source directory {args.src} does not exist.")
                
            valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
            
            # Recursive walk or simple listdir based on needs. Using listdir for simplicity.
            for filename in os.listdir(args.src):
                if filename.endswith(valid_exts):
                    file_path = os.path.join(args.src, filename)
                    try:
                        if args.double:
                            process_doublewithoutbg(file_path, dst_folder=args.dst)
                        else:
                            process_single_image(file_path, dst_folder=args.dst)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

        # Single Image Mode
        elif args.image_path:
            if not is_valide_image(args.image_path):
                raise ValueError("The provided file is not a valid image.")
            process_single_image(args.image_path)
            
        else:
            parser.print_help()
            sys.exit(1)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()