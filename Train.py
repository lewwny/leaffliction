import json
import sys
import shutil
import hashlib
import copy
import time
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# consts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
B_SIZE = 32
EPOCHS = 10
VALID_SPLIT = 0.2
TARGET_ACC = 0.9


def print_device() -> None:
    """prints device type for processing"""
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU: Training will be slower")


def time_format(seconds: float) -> str:
    """formats from time() to readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sort_data(folder: str) -> Tuple[Path, Path, List[str]]:
    """sorts data from folder for train and validation imgs"""
    # get classes from source
    source = Path(folder)
    classes = [c.name for c in source.iterdir() if c.is_dir()]
    classes.sort()

    print(f"Classes Found: {len(classes)}")

    # counts images
    total_imgs = 0
    for i, c in enumerate(classes):
        class_path = source / c
        img_count = (len(list(class_path.glob("*.[jJ][pP][gG]")))
                     + len(list(class_path.glob("*.[jJ][pP][eE][gG]")))
                     + len(list(class_path.glob("*.[pP][nN][gG]"))))
        print(f"{i + 1}.{c}: {img_count} images")
        total_imgs += img_count

    print(f"Total images: {total_imgs}")

    # rmtree for the temp
    temp_folder = Path("temp_dataset")
    if temp_folder.exists():
        # recursive delete
        shutil.rmtree(temp_folder)

    # create train and validation directories
    train_dir = temp_folder / "train"
    valid_dir = temp_folder / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    # create subdirectories for classes in train/validation dirs
    for c in classes:
        (train_dir / c).mkdir(exist_ok=True)
        (valid_dir / c).mkdir(exist_ok=True)

        class_folder = source / c

        # get using image type (jpg, jpeg, png) capitalization insensitive
        imgs = (list(class_folder.glob("*.[jJ][pP][gG]"))
                     + list(class_folder.glob("*.[jJ][pP][eE][gG]"))
                     + list(class_folder.glob("*.[pP][nN][gG]")))

        # use sklearn train test split on images
        train_imgs, valid_imgs = train_test_split(imgs, test_size=VALID_SPLIT, random_state=42)

        # shutil copy all images in train and validation
        for img in train_imgs:
            shutil.copy(img, train_dir / c / img.name)
        for img in valid_imgs:
            shutil.copy(img, valid_dir / c / img.name)

        print(f"{c}: {len(train_imgs)} train, {len(valid_imgs)} validation")

    return train_dir, valid_dir, classes


def data_load(train_dir: Path, valid_dir: Path) -> Tuple[DataLoader, DataLoader]:
    """augmentation + normalization for training data"""
    # imagenet normalization values (mean/std used by pretrained models)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # transforming data for augmentation and norm
    transforms_data = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),

        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE[0]),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }

    # builds datasets from imagefolder
    train_dataset = datasets.ImageFolder(train_dir, transforms_data["train"])
    valid_dataset = datasets.ImageFolder(valid_dir, transforms_data["valid"])

    # create dataloader from datasets
    train_loader = DataLoader(train_dataset, batch_size=B_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=B_SIZE, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def create_model(num_classes: int) -> nn.Module:
    """creates a model using MobileNetV2"""
    # gets the mobilenetv2 model with default weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # freeze base layers
    for p in model.features.parameters():
        p.requires_grad = False

    # replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model.to(DEVICE)


def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader
) -> Tuple[Dict[str, Any], Dict[str, List[float]], float, float]:
    """trains model using train and validation loaders"""
    # get the optimizer for parameters of the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # uses torch to get crossentropyloss
    xel = nn.CrossEntropyLoss()

    # scheduler for the LR -> reduces on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # metrics to keep during training
    metrics = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }

    # counter vars for training
    best_model = copy.deepcopy(model.state_dict())
    max_acc = 0.0
    count_patience = 0
    max_patience = 7
    start_time = time.time()

    # get total passes count
    total_steps = EPOCHS * (len(train_loader) + len(valid_loader))

    # progress bar using tqdm
    progress = tqdm(total=total_steps, unit="batch", leave=True)

    # iterate over epochs in batches
    for e in range (EPOCHS):
        # go through both training and validation phases
        for phase in ["train", "valid"]:
            if phase == "train":
                dataloader = train_loader
                model.train()
            else:
                dataloader = valid_loader
                model.eval()

            # metrics over epochs
            samples = 0
            corrections = 0
            loss = 0.0

            # go through data and send to device
            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # reset grad for the optimizers (avoids accumulation from previous batch)
                optimizer.zero_grad()

                # set training phase grad on torch (disabled during validation for speed)
                with torch.set_grad_enabled(phase == "train"):
                    outs = model(inputs)
                    _, preds = torch.max(outs, 1)
                    local_loss = xel(outs, labels)

                    # if phase is training (backpdrop and update weights)
                    if phase == "train":
                        local_loss.backward()
                        optimizer.step()

                # update metrics
                samples += inputs.size(0)
                corrections += torch.sum(preds == labels.data).item()
                loss += local_loss.item() * inputs.size(0)

                # update progress bar
                progress.update(1)

            # get epoch loss and acc
            e_loss = loss / samples
            e_acc = corrections / samples

            # store metrics
            if phase == "train":
                metrics["train_acc"].append(e_acc)
                metrics["train_loss"].append(e_loss)
                metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])
            else:
                metrics["val_acc"].append(e_acc)
                metrics["val_loss"].append(e_loss)

                # move scheduler using epoch acc
                scheduler.step(e_acc)

                # deep copy model if better acc
                if e_acc > max_acc:
                    max_acc = e_acc
                    best_model = copy.deepcopy(model.state_dict())
                    count_patience = 0
                    progress.write(f"Epoch {e+1}: New best accuracy: {e_acc:.4f}")
                else:
                    # add to patience counter to stop training uselessly
                    count_patience += 1

        # early stop checker (patience limit reached)
        if count_patience >= max_patience:
            progress.write("Max patience reached - Stopping")
            break

    # close progress bar
    progress.close()

    # print total training time
    total_time = time.time() - start_time
    print(f"Total training time: {time_format(total_time)}")
    print(f"Best validation accuracy: {max_acc:.4f}")

    # return best model and metrics
    return best_model, metrics, max_acc, total_time


def save_model(path: str, model_state: Dict[str, Any], classes: List[str],
               metrics: Dict[str, List[float]], folder: str) -> None:
    """saves trained model into file"""
    # save model
    save_dir = Path(Path(folder).name + "_model")
    save_dir.mkdir(exist_ok=True)

    model_path = save_dir / f"{path}.pth"
    torch.save(model_state, model_path)
    print(f"model saved to {model_path}")

    # make archive
    archive_path = shutil.make_archive(path, "zip", save_dir)
    print(f"archive saved to {archive_path}")

    # create sha1 signature
    sig = hashlib.sha1()
    with open(archive_path, "rb") as file:
        while chunk := file.read(8192):
            sig.update(chunk)

    # save signature to file
    with open("signature.txt", "w") as file:
        file.write(f"{sig.hexdigest()}  {archive_path}\n")

    # load metadata and save to file
    model_meta = {
        "classes": classes,
        "image_size": IMG_SIZE,
        "class_count": len(classes),
        "epochs": len(metrics["train_acc"]),
        "train_accuracy": float(metrics["train_acc"][-1]),
        "valid_accuracy": float(metrics["val_acc"][-1]),
        "train_loss": float(metrics["train_loss"][-1]),
        "valid_loss": float(metrics["val_loss"][-1]),
    }

    meta_path = save_dir / f"{path}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(model_meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def plot_metrics(metrics: Dict[str, List[float]], save_path: str = "training_metrics.png") -> None:
    """plots history for training"""
    epochs = list(range(1, len(metrics["train_acc"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

    # accuracy over epochs (train vs validation)
    ax1 = axes[0, 0]
    ax1.plot(epochs, metrics["train_acc"], 'b-o', label="Train Accuracy", linewidth=2, markersize=4)
    ax1.plot(epochs, metrics["val_acc"], 'r-o', label="Validation Accuracy", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # loss over epochs (train vs validation)
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics["train_loss"], 'b-o', label="Train Loss", linewidth=2, markersize=4)
    ax2.plot(epochs, metrics["val_loss"], 'r-o', label="Validation Loss", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Model Loss (CrossEntropy)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, metrics["learning_rate"], 'g-o', label="Learning Rate", linewidth=2, markersize=4)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    # overfitting detection (gap between train and val)
    ax4 = axes[1, 1]
    acc_gap = [t - v for t, v in zip(metrics["train_acc"], metrics["val_acc"])]
    loss_gap = [v - t for t, v in zip(metrics["train_loss"], metrics["val_loss"])]
    ax4.bar([e - 0.2 for e in epochs], acc_gap, 0.4, label="Acc Gap (Train-Val)", color="blue", alpha=0.7)
    ax4.bar([e + 0.2 for e in epochs], loss_gap, 0.4, label="Loss Gap (Val-Train)", color="red", alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Gap")
    ax4.set_title("Overfitting Detection (positive = overfitting)")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> int:
    """uses an image folder to train a model for leaf recognition ML"""
    # argv for image folder
    if len(sys.argv) < 2:
        print("Usage: python Train.py <img_folder>")
        return 1

    # path for where we'll save the final trained model
    save_path = "leaffliction_model"

    # check if folder exists
    img_folder = sys.argv[1]
    if not Path(img_folder).exists():
        print(f"Error: {img_folder} doesn't exist")
        return 1

    # show device used for training (either gpu or cpu)
    print_device()

    # get sorted data and convert to dataloaders
    train_dir, valid_dir, classes = sort_data(img_folder)
    train_loader, valid_loader = data_load(train_dir, valid_dir)

    # create model and train, save metrics
    model = create_model(len(classes))
    best_model, metrics, best_acc, train_time = train_model(model, train_loader, valid_loader)

    # save model
    save_model(save_path, best_model, classes, metrics, img_folder)

    # show results
    plot_metrics(metrics)

    # cleanup
    temp_folder = Path("temp_dataset")
    if temp_folder.exists():
        shutil.rmtree(temp_folder)

    return 0


if __name__ == "__main__":
    sys.exit(main())
