# Training Deep Learning Models for Image Classification

This document explains all the core concepts, steps, and techniques used in `Train.py` for training a convolutional neural network to classify leaf diseases.

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware: CPU vs GPU](#hardware-cpu-vs-gpu)
3. [Data Preparation](#data-preparation)
   - [Train/Validation Split](#trainvalidation-split)
   - [Why Split Data?](#why-split-data)
4. [Data Loading Pipeline](#data-loading-pipeline)
   - [Transforms](#transforms)
   - [Data Augmentation](#data-augmentation)
   - [Normalization](#normalization)
   - [ImageFolder](#imagefolder)
   - [DataLoader](#dataloader)
5. [Transfer Learning](#transfer-learning)
   - [What is Transfer Learning?](#what-is-transfer-learning)
   - [MobileNetV2](#mobilenetv2)
   - [Freezing Layers](#freezing-layers)
   - [Replacing the Classifier](#replacing-the-classifier)
6. [The Training Loop](#the-training-loop)
   - [Epochs](#epochs)
   - [Batches](#batches)
   - [Forward Pass](#forward-pass)
   - [Loss Function (CrossEntropyLoss)](#loss-function-crossentropyloss)
   - [Backward Pass (Backpropagation)](#backward-pass-backpropagation)
   - [Optimizer (Adam)](#optimizer-adam)
   - [Gradient Management](#gradient-management)
7. [Training vs Evaluation Mode](#training-vs-evaluation-mode)
8. [Learning Rate Scheduling](#learning-rate-scheduling)
9. [Early Stopping](#early-stopping)
10. [Model Saving](#model-saving)
    - [state_dict()](#state_dict)
    - [Checkpointing](#checkpointing)
11. [Metrics and Visualization](#metrics-and-visualization)
12. [Key Libraries](#key-libraries)
13. [Hyperparameters Summary](#hyperparameters-summary)

---

## Overview

Training a neural network is an iterative optimization process where the model learns to map inputs (images) to outputs (class labels) by adjusting its internal parameters (weights and biases).

```
Input Image → Neural Network → Predicted Class
                    ↑
              Adjust weights based on errors
```

The training pipeline consists of:
1. **Data Preparation** - Split and organize images
2. **Data Loading** - Transform and batch images efficiently
3. **Model Creation** - Initialize a pretrained network
4. **Training Loop** - Iteratively update weights
5. **Evaluation** - Validate on unseen data
6. **Saving** - Export the trained model

---

## Hardware: CPU vs GPU

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### CPU (Central Processing Unit)
- General-purpose processor
- Sequential processing
- Slower for neural networks (hours to days)

### GPU (Graphics Processing Unit)
- Specialized for parallel operations
- Thousands of cores
- 10-100x faster for deep learning

**Why GPUs?** Neural networks involve massive matrix multiplications. GPUs can perform thousands of these operations simultaneously.

```python
# moving data to GPU
inputs = inputs.to(DEVICE)
model = model.to(DEVICE)
```

---

## Data Preparation

### Train/Validation Split

```python
train_imgs, valid_imgs = train_test_split(imgs, test_size=0.2, random_state=42)
```

Data is split into:
- **Training set (80%)** - Used to update model weights
- **Validation set (20%)** - Used to evaluate generalization

### Why Split Data?

| Set | Purpose | Used for Weight Updates? |
|-----|---------|-------------------------|
| Training | Learn patterns | Yes |
| Validation | Check generalization | No |
| Test | Final evaluation | No |

**Overfitting** occurs when a model memorizes training data but fails on new data. The validation set helps detect this.

```
Training Accuracy: 99%  }
                        } Large gap = Overfitting!
Validation Accuracy: 70%}
```

### sklearn's train_test_split

```python
from sklearn.model_selection import train_test_split
```

- `test_size=0.2` → 20% for validation
- `random_state=42` → Seed for reproducibility (same split every run)
- Returns two lists: training samples and validation samples

---

## Data Loading Pipeline

### Transforms

Transforms are preprocessing operations applied to images before feeding them to the model.

```python
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

`Compose` chains multiple transforms together, applied in order.

### Data Augmentation

Artificially increases dataset diversity by applying random transformations:

| Transform | Effect | Purpose |
|-----------|--------|---------|
| `RandomResizedCrop(224)` | Random crop + resize | Scale invariance |
| `RandomHorizontalFlip()` | 50% chance flip | Orientation invariance |
| `RandomRotation(15)` | ±15° rotation | Rotation invariance |
| `ColorJitter(0.2, 0.2)` | Random brightness/contrast | Lighting invariance |

**Why augment?**
- Prevents overfitting
- Simulates real-world variations
- Effectively multiplies dataset size

### Normalization

```python
imagenet_mean = [0.485, 0.456, 0.406]  # RGB channel means
imagenet_std = [0.229, 0.224, 0.225]   # RGB channel stds

transforms.Normalize(imagenet_mean, imagenet_std)
```

Normalization standardizes pixel values:
$$x_{normalized} = \frac{x - \mu}{\sigma}$$

**Why normalize?**
- Pretrained models expect ImageNet-normalized inputs
- Helps gradient descent converge faster
- Centers data around zero

### ToTensor

```python
transforms.ToTensor()
```

Converts PIL Image to PyTorch tensor:
- Changes shape: `(H, W, C)` → `(C, H, W)`
- Changes range: `[0, 255]` → `[0.0, 1.0]`

### ImageFolder

```python
train_dataset = datasets.ImageFolder(train_dir, transform)
```

Automatically creates a dataset from directory structure:
```
train_dir/
├── Apple_scab/
│   ├── img001.jpg
│   └── img002.jpg
├── Black_rot/
│   └── img003.jpg
└── Healthy/
    └── img004.jpg
```

Each subdirectory becomes a class label (0, 1, 2...).

### DataLoader

```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

Wraps a dataset for efficient batch loading:

| Parameter | Purpose |
|-----------|---------|
| `batch_size=32` | Process 32 images at once |
| `shuffle=True` | Randomize order each epoch |
| `num_workers=4` | Parallel data loading threads |

**Why batches?**
- GPU memory is limited
- Stochastic updates help escape local minima
- Faster than processing one image at a time

```python
for inputs, labels in train_loader:
    # inputs.shape = (32, 3, 224, 224) → 32 images, 3 channels, 224x224
    # labels.shape = (32,) → 32 class indices
```

---

## Transfer Learning

### What is Transfer Learning?

Instead of training from scratch, we start with a model pretrained on a large dataset (ImageNet: 1.2M images, 1000 classes).

```
ImageNet Knowledge → Your Task
    (general)         (specific)
```

**Benefits:**
- Faster training (hours instead of days)
- Better results with less data
- Lower computational requirements

### MobileNetV2

```python
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
```

MobileNetV2 is a lightweight CNN designed for mobile devices:
- ~3.4M parameters (vs ResNet50's 25M)
- Uses depthwise separable convolutions
- Good accuracy/speed tradeoff

Architecture:
```
Input (224x224x3)
    ↓
[Feature Extractor] ← Pretrained, frozen
    - Convolutional layers
    - Learns edges, textures, shapes
    ↓
[Classifier] ← Replaced, trainable
    - Fully connected layer
    - Maps features to classes
    ↓
Output (num_classes)
```

### Freezing Layers

```python
for p in model.features.parameters():
    p.requires_grad = False
```

**Freezing** prevents weight updates during training:
- `requires_grad = False` → No gradients computed
- Preserves pretrained knowledge
- Reduces training time and memory

### Replacing the Classifier

```python
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
```

The original classifier outputs 1000 ImageNet classes. We replace it with a new layer for our classes (e.g., 4 leaf diseases).

```
Original: features → Linear(1280, 1000) → 1000 classes
Modified: features → Linear(1280, 4)    → 4 classes
```

---

## The Training Loop

### Epochs

```python
for epoch in range(EPOCHS):  # EPOCHS = 10
```

One **epoch** = one complete pass through the entire training dataset.

```
Epoch 1: See all images → Update weights
Epoch 2: See all images → Update weights
...
Epoch 10: Final pass
```

More epochs = more learning opportunities, but risk of overfitting.

### Batches

Each epoch processes data in batches:

```python
for inputs, labels in dataloader:
    # Process one batch of 32 images
```

One **iteration** = one batch processed.

```
Dataset: 1000 images
Batch size: 32
Iterations per epoch: ⌈1000/32⌉ = 32
```

### Forward Pass

```python
outputs = model(inputs)  # Shape: (batch_size, num_classes)
_, preds = torch.max(outputs, 1)  # Get predicted class indices
```

Data flows through the network:
```
Image → Conv layers → Features → Classifier → Class scores
```

The output is a vector of scores (logits) for each class.

### Loss Function (CrossEntropyLoss)

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

CrossEntropyLoss measures prediction error:

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

Where:
- $y_i$ = true label (one-hot encoded)
- $\hat{y}_i$ = predicted probability

| Prediction | Loss |
|------------|------|
| Correct (high confidence) | Low |
| Correct (low confidence) | Medium |
| Wrong | High |

CrossEntropyLoss combines:
1. **Softmax** - Converts logits to probabilities
2. **Negative Log Likelihood** - Penalizes wrong predictions

### Backward Pass (Backpropagation)

```python
loss.backward()
```

Computes gradients using the chain rule:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial w}$$

Gradients indicate how much each weight contributed to the error.

```
Loss
  ↓ (backpropagate)
Classifier weights ← gradient
  ↓
Feature weights ← gradient (if not frozen)
```

### Optimizer (Adam)

```python
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
optimizer.step()
```

Updates weights based on gradients:
$$w_{new} = w_{old} - \eta \cdot \nabla L$$

**Adam** (Adaptive Moment Estimation):
- Combines momentum and RMSprop
- Adapts learning rate per parameter
- Generally works well out-of-the-box

| Optimizer | Description |
|-----------|-------------|
| SGD | Basic gradient descent |
| SGD + Momentum | Accelerates in consistent directions |
| Adam | Adaptive learning rates |

### Gradient Management

```python
optimizer.zero_grad()  # Reset gradients before backward pass
```

PyTorch **accumulates** gradients by default. Without zeroing:
```
Batch 1: gradient = 0.5
Batch 2: gradient = 0.5 + 0.3 = 0.8  ← Wrong!
```

Must reset before each batch.

---

## Training vs Evaluation Mode

```python
model.train()  # Training mode
model.eval()   # Evaluation mode
```

| Behavior | train() | eval() |
|----------|---------|--------|
| Dropout | Active (random neurons disabled) | Disabled |
| BatchNorm | Uses batch statistics | Uses running averages |
| Purpose | Learning with regularization | Consistent predictions |

```python
with torch.set_grad_enabled(phase == "train"):
    # Gradients computed only during training
    # Saves memory during validation
```

---

## Learning Rate Scheduling

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)
scheduler.step(validation_accuracy)
```

**ReduceLROnPlateau**:
- Monitors a metric (validation accuracy)
- If no improvement for `patience` epochs → reduce LR
- `factor=0.5` → multiply LR by 0.5

```
Epoch 1-3: lr = 0.001
Epoch 4-6: lr = 0.001 (no improvement)
Epoch 7:   lr = 0.0005 (reduced!)
```

**Why reduce LR?**
- Large LR for fast initial progress
- Small LR for fine-tuning near optimum
- Helps escape plateaus

---

## Early Stopping

```python
if count_patience >= max_patience:
    print("Max patience reached - Stopping")
    break
```

Stops training when validation accuracy stops improving:

```
Epoch 5: val_acc = 0.85 (best)
Epoch 6: val_acc = 0.84 (patience = 1)
Epoch 7: val_acc = 0.83 (patience = 2)
...
Epoch 12: val_acc = 0.82 (patience = 7 → STOP)
```

**Benefits:**
- Prevents overfitting
- Saves training time
- Returns best model, not final model

---

## Model Saving

### state_dict()

```python
best_model = copy.deepcopy(model.state_dict())
torch.save(model_state, "model.pth")
```

`state_dict()` is a Python dictionary containing:
- All learnable parameters (weights, biases)
- No model architecture

```python
{
    'classifier.1.weight': tensor([...]),
    'classifier.1.bias': tensor([...]),
    ...
}
```

### Why deepcopy?

```python
best_model = copy.deepcopy(model.state_dict())
```

Without deepcopy, you save a **reference** that changes as training continues.

### Checkpointing

Best practice: save during training, not just at the end.

```python
if e_acc > max_acc:
    max_acc = e_acc
    best_model = copy.deepcopy(model.state_dict())
```

This ensures you keep the best-performing model even if later epochs overfit.

---

## Metrics and Visualization

### Tracked Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | correct / total | % of correct predictions |
| Loss | CrossEntropyLoss | Prediction error magnitude |
| Learning Rate | optimizer.param_groups[0]["lr"] | Current step size |

### Overfitting Detection

```python
acc_gap = train_acc - val_acc
```

| Gap | Interpretation |
|-----|----------------|
| ~0 | Good generalization |
| > 0.1 | Mild overfitting |
| > 0.2 | Significant overfitting |

### Visualization

The 2x2 metrics plot shows:
1. **Accuracy curves** - Should both increase, stay close
2. **Loss curves** - Should both decrease, stay close
3. **Learning rate** - Shows scheduler behavior
4. **Gap analysis** - Quantifies overfitting

---

## Key Libraries

### shutil (Shell Utilities)

```python
import shutil

shutil.copy(src, dst)        # Copy file
shutil.rmtree(path)          # Delete directory recursively (rm -rf)
shutil.make_archive(...)     # Create ZIP/TAR archive
```

### torch (PyTorch)

Core deep learning framework:
- Tensors (GPU-accelerated arrays)
- Automatic differentiation
- Neural network modules

### torchvision

Computer vision utilities:
- Pretrained models (`models.mobilenet_v2`)
- Image transforms (`transforms.Compose`)
- Dataset loaders (`datasets.ImageFolder`)

### sklearn (scikit-learn)

Machine learning utilities:
- `train_test_split` - Data splitting
- Preprocessing, metrics, etc.

### tqdm

Progress bar for loops:
```python
for batch in tqdm(dataloader):
    ...
```

---

## Hyperparameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | (224, 224) | Input image dimensions |
| `B_SIZE` | 32 | Batch size |
| `EPOCHS` | 10 | Maximum training epochs |
| `VALID_SPLIT` | 0.2 | Validation set ratio |
| `lr` | 0.001 | Initial learning rate |
| `max_patience` | 7 | Early stopping patience |

### Tuning Guidelines

| Problem | Solution |
|---------|----------|
| Underfitting (low train acc) | Increase epochs, unfreeze layers, larger model |
| Overfitting (high gap) | More augmentation, dropout, early stopping |
| Slow training | Larger batch size, higher LR, GPU |
| Out of memory | Smaller batch size, smaller model |

---

## Training Pipeline Summary

```
1. Load and split data
   └─ train_test_split → 80% train, 20% validation

2. Create data pipeline
   └─ transforms → augmentation + normalization
   └─ ImageFolder → dataset from directories
   └─ DataLoader → batched iteration

3. Initialize model
   └─ Load pretrained MobileNetV2
   └─ Freeze feature layers
   └─ Replace classifier

4. Training loop
   └─ For each epoch:
       └─ Training phase:
           └─ Forward pass → predictions
           └─ Loss calculation
           └─ Backward pass → gradients
           └─ Optimizer step → update weights
       └─ Validation phase:
           └─ Forward pass only
           └─ Calculate metrics
           └─ Update scheduler
           └─ Save best model

5. Save and visualize
   └─ Save state_dict to .pth
   └─ Save metadata to JSON
   └─ Plot metrics
```
