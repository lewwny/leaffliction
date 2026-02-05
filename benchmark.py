import sys
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from predict import load_model, get_transform, predict_image


IMAGES_DIR = Path("images")
SAMPLES = 20


def collect_samples(classes: list) -> tuple:
    """collect random samples from each class directory"""
    images, labels = [], []
    for cls in classes:
        cls_dir = IMAGES_DIR / cls
        if not cls_dir.exists():
            continue
        all_images = list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        samples = random.sample(all_images, min(SAMPLES, len(all_images)))
        images.extend([str(img) for img in samples])
        labels.extend([cls] * len(samples))
    return images, labels


def run_benchmark(model, model_name: str, classes, images, true_labels, use_mask: bool = False):
    """run benchmark on model and plot results"""
    # get transforms
    transform = get_transform()

    # run predictions
    print(f"Running predictions on {len(images)} images (mask={use_mask})")
    predictions = []
    for i, img_path in enumerate(images):
        try:
            pred_class, _, _ = predict_image(model, img_path, classes, transform, use_mask)
            predictions.append(pred_class)
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            predictions.append(None)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(images)}")

    # filter out failed predictions
    valid = [(t, p) for t, p in zip(true_labels, predictions) if p is not None]
    true_labels, predictions = zip(*valid) if valid else ([], [])

    # calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    # plot results
    mode = "masked" if use_mask else "normal"
    title = f"{model_name} ({mode})"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # metrics bar chart
    ax1 = axes[0]
    metrics = {'Accuracy': accuracy, 'F1 Score': f1}
    bars = ax1.bar(metrics.keys(), metrics.values(), color=['steelblue', 'darkorange'])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.set_title(f'{title}\nAccuracy: {accuracy:.2%} | F1: {f1:.2%}')
    for bar, val in zip(bars, metrics.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2%}', ha='center', fontweight='bold')

    # confusion matrix
    ax2 = axes[1]
    cm = confusion_matrix(true_labels, predictions, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax2, cmap='Blues', values_format='d', xticks_rotation=45)
    ax2.set_title('Confusion Matrix')

    plt.tight_layout()

    # save
    output_file = f"benchmark_{model_name}_{mode}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

    # print metrics
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1 Score: {f1:.2%}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmark.py <model1_dir> <model2_dir")
        return 1

    model_path1 = sys.argv[1]

    if not Path(model_path1).exists():
        print(f"Error: {model_path1} not found")
        return 1
    
    model_path2 = sys.argv[2]

    if not Path(model_path2).exists():
        print(f"Error: {model_path2} not found")
        return 1

    random.seed(42)
    model1, classes1, _ = load_model(model_path1)
    model2, _, _ = load_model(model_path1)
    images, true_labels = collect_samples(classes1)
    run_benchmark(model1, "model_v1", classes1, images, true_labels, False)
    run_benchmark(model2, "model_v2", classes1, images, true_labels, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
