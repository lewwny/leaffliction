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


def predict_image(model: nn.Module, image_path: str, classes: List[str],
                  transform: transforms.Compose):
    """makes a class prediction for an image"""
    # load and transform image
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
    transform = get_transform()

    # predict each image in each image path
    for image_path in image_paths:
        if not Path(image_path).exists():
            print(f"{image_path} not found")
            continue

        try:
            # predict here
            p_class, confidence, probas = predict_image(model, image_path, classes, transform)
            print_pred(image_path, p_class, confidence, probas)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
