# Problem Set 01 - Pneumonia Detection using CNN

## Objective
Build a CNN model to classify chest X-ray images into Pneumonia and Normal categories.

## Dataset
- Total images: 5,856
- Categories: NORMAL, PNEUMONIA
- Training: 5,216 images
- Validation: 16 images
- Testing: 624 images

The dataset has class imbalance with more pneumonia images than normal ones in the training set.

## Approach
Used a custom CNN built in PyTorch with 4 convolutional blocks. Each block has two Conv2D layers with BatchNorm and ReLU, followed by MaxPooling. Global Average Pooling reduces spatial dimensions before the classifier head.

### Preprocessing
- Resized images to 150x150
- Applied data augmentation (random flips, rotation, color jitter, affine transforms)
- Normalized using ImageNet statistics

### Model Details
- 4 conv blocks: 32 -> 64 -> 128 -> 256 filters
- Global Average Pooling instead of Flatten (reduces parameters)
- Dense(128) with Dropout(0.45) then Dense(1) with sigmoid
- Total parameters: ~1.2M
- Weighted BCE loss to handle class imbalance
- Adam optimizer with lr=0.0003
- ReduceLROnPlateau scheduler
- Trained for 15 epochs

### Evaluation
- Classification report with precision, recall, f1-score
- Confusion matrix
- ROC curve with AUC score
- Precision-Recall curve
- Visual predictions on test samples

## How to Run
```
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas pillow
python pneumonia_cnn.py
```

Make sure the train/, val/, test/ directories are in the same folder as the script.

## Results
- **Test Accuracy**: ~86-89% (after 15 epochs)
- **AUC-ROC Score**: ~0.94
- Strong precision and recall for the Pneumonia class due to the weighted loss function addressing the class imbalance.

## Findings
- The model learns to distinguish pneumonia from normal X-rays effectively
- Data augmentation and batch normalization help prevent overfitting
- Weighted loss function improves recall for both classes despite imbalance
- The small validation set (only 16 images) causes fluctuation in validation metrics
