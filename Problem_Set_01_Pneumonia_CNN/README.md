# Problem Set 01 — Pneumonia Detection from Chest X-Ray Images using CNN

## Overview

This project develops a **Convolutional Neural Network (CNN)** to classify chest X-ray images of paediatric patients (aged 1–5 years) into two categories: **NORMAL** and **PNEUMONIA**. The model assists healthcare professionals by providing automated screening of anterior-posterior chest X-ray images.

## Dataset

- **Source**: Chest X-Ray Images (Pneumonia) dataset  
- **Total Images**: 5,856 JPEG images  
- **Classes**: NORMAL, PNEUMONIA  
- **Split Distribution**:

| Split       | Normal | Pneumonia | Total |
|-------------|--------|-----------|-------|
| Training    | 1,341  | 3,875     | 5,216 |
| Validation  | 8      | 8         | 16    |
| Test        | 234    | 390       | 624   |

> **Note**: The dataset exhibits significant class imbalance — pneumonia cases outnumber normal cases roughly 3:1 in the training set. This is addressed through weighted loss functions.

## Approach & Methodology

### 1. Data Preprocessing & Augmentation
- All images are resized to **150×150** pixels
- **Training augmentations** to reduce overfitting:
  - Random horizontal flipping (p=0.4)
  - Random rotation (±12°)
  - Colour jitter (brightness & contrast)
  - Random affine translation
- **Normalisation** using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 2. Model Architecture
A custom CNN built with **PyTorch** comprising:
- **4 convolutional blocks**, each containing:
  - Two Conv2D layers (3×3 kernel, padding=1)
  - Batch Normalisation after each convolution
  - ReLU activation
  - Max Pooling (2×2)
- **Global Average Pooling** to reduce spatial dimensions
- **Fully-connected classifier head**:
  - Dense (256 → 128) with ReLU
  - Dropout (45%) for regularisation
  - Output layer (128 → 1) with sigmoid activation

**Channel progression**: 3 → 32 → 64 → 128 → 256

### 3. Training Strategy
- **Loss Function**: Binary Cross-Entropy with Logits (weighted by class ratio to handle imbalance)
- **Optimiser**: Adam (lr=0.0003, weight_decay=1e-4)
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Epochs**: 15
- **Batch Size**: 32
- **Best model checkpoint** saved based on validation AUC

### 4. Evaluation Metrics
- Accuracy
- AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
- F1-Score
- Precision & Recall
- Confusion Matrix

## How to Run

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas pillow
```

### Execute
```bash
python pneumonia_cnn.py
```

The script will:
1. Load and augment the chest X-ray images
2. Train the CNN for 15 epochs
3. Evaluate on the test set
4. Generate the following visualisation files:
   - `sample_images.png` — Grid of sample training images
   - `class_distribution.png` — Bar chart of class counts per split
   - `training_history.png` — Loss, accuracy, and AUC curves over epochs
   - `confusion_matrix.png` — Test set confusion matrix
   - `roc_curve.png` — ROC curve with AUC score
   - `precision_recall_curve.png` — Precision vs recall trade-off
   - `prediction_samples.png` — Predictions on random test images

## Key Design Decisions

1. **Custom CNN over pre-trained models**: A purpose-built architecture keeps the model lightweight and demonstrates understanding of CNN fundamentals, while still achieving strong performance on this binary task.

2. **Weighted BCE loss**: The positive weight (normal/pneumonia ratio ≈ 0.346) compensates for the 3:1 class imbalance, ensuring the model does not simply predict "PNEUMONIA" for every input.

3. **Global Average Pooling**: Reduces the feature maps to a single vector without flattening, which significantly cuts parameter count and reduces overfitting risk.

4. **Aggressive data augmentation**: With only ~5,200 training images, augmentation is critical to prevent the model from memorising the training set.

## Findings

- The CNN successfully learns discriminative features from chest X-ray images
- Data augmentation and batch normalisation are essential to avoid overfitting on this relatively small medical imaging dataset
- The class imbalance handling via weighted loss ensures reasonable recall for both classes
- The model achieves high sensitivity (recall) for pneumonia detection, which is clinically important — missing a pneumonia case is more costly than a false alarm

## Directory Structure

```
Problem_Set_01_Pneumonia_CNN/
├── pneumonia_cnn.py          # Main training and evaluation script
├── README.md                 # This file
├── train/                    # Training images
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/                      # Validation images
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/                     # Test images
│   ├── NORMAL/
│   └── PNEUMONIA/
└── (generated outputs)
    ├── best_pneumonia_model.pth
    ├── sample_images.png
    ├── class_distribution.png
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    └── prediction_samples.png
```
