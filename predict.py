import json
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import rembg
from plantcv import plantcv as pcv


# consts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)


def load_model(path: str):
    """loads model from path"""
    # find files in path
    model_path = Path(path)
    model_files = list(model_path.glob("*.pth"))
    metadata_files = list(model_path.glob("*_metadata.json"))

    # verify needed files exist and are found
    if not model_files or not metadata_files:
        raise FileNotFoundError(f"files missing for prediction")

    # load metadata
    with open(metadata_files[0], 'r') as file:
        metadata = json.load(file)

    classes = metadata["classes"]
    class_count = metadata["class_count"]

    # create model architecture using mobilenetv2
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, class_count)

    # load trained weights to overwrite classifier
    model.load_state_dict(torch.load(model_files[0], map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model, classes, metadata


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


def apply_mask_transform(image_path: str) -> Image.Image:
    """Apply mask transformation (leaf on black background) to image"""
    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # remove background
    image_without_bg = rembg.remove(image)
    
    # pre-processing with plantCV
    l_grayscale = pcv.rgb2gray_lab(rgb_img=image_without_bg, channel='l')
    l_thresh = pcv.threshold.binary(gray_img=l_grayscale, threshold=35, object_type='light')
    filled = pcv.fill(bin_img=l_thresh, size=200)
    gaussian_blur = pcv.gaussian_blur(img=filled, ksize=(3, 3))
    masked = pcv.apply_mask(img=image, mask=gaussian_blur, mask_color='black')
    
    # convert to PIL Image
    return Image.fromarray(masked)


def predict_image(model: nn.Module, image_path: str, classes: List[str],
                  transform: transforms.Compose, use_mask: bool = False):
    """makes a class prediction for an image"""
    # load and transform image
    if use_mask:
        image = apply_mask_transform(image_path)
    else:
        image = Image.open(image_path).convert("RGB")
    
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probas = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predindex = torch.max(probas, 1)

    # predict
    p_class = classes[predindex.item()]
    p_confidence = confidence.item()

    # get probabilities for each class
    probabilities = {classes[i]: probas[0][i].item() for i in range (len(classes))}

    # return all metrics
    return p_class, p_confidence, probabilities


def print_pred(image_path: str, p_class: str, confidence: float,
               probas: Dict[str, float]) -> None:
    """prints the results of predictions"""
    print(f"Image: {image_path}\nPredicted class: {p_class}")
    print(f"Confidence: {confidence:.2%}\nAll probabilities:")
    for c, proba in sorted (probas.items()):
        print(f"{c:30s} {proba:.2%}")
    return

def draw_pred(image_path: str, p_class: str, confidence: float, metadata: Dict[str, Any]) -> None:
    """draws prediction on image"""
    # load image
    image = cv2.imread(image_path)

    
    # draw text on image
    text = f"Predicted: {p_class} ({confidence:.2%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 255, 0)
    thickness = 2
    position = (10, 30)
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    # show image in a window
    window_name = "Prediction"
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> int:
    """makes predictions using a pre-created leaffliction model"""
    # argv for model dir and image(s)
    if len(sys.argv) < 3:
        print("Usage: python Predict.py <model_dir> <image_path> [image_path2 ...] [--mask]")
        print("  --mask: apply mask transformation (use if model was trained on masked images)")
        return 1

    # Check for --mask flag
    use_mask = "--mask" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--mask"]
    
    model_dir = args[0]
    image_paths = args[1:]

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
    transform = get_transform()

    # predict each image in each image path
    for image_path in image_paths:
        if not Path(image_path).exists():
            print(f"{image_path} not found")
            continue

        try:
            # predict here
            p_class, confidence, probas = predict_image(model, image_path, classes, transform, use_mask)
            print_pred(image_path, p_class, confidence, probas)
            draw_pred(image_path, p_class, confidence, metadata)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
