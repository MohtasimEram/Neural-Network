import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMAGE_SIZE = 150
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.0003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation(degrees=12),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

class_names = train_dataset.classes
print(f"[INFO] Classes detected: {class_names}")
print(f"[INFO] Training samples  : {len(train_dataset)}")
print(f"[INFO] Validation samples: {len(val_dataset)}")
print(f"[INFO] Test samples      : {len(test_dataset)}")


def show_sample_images(dataset, class_names, num_cols=6):
    fig, axes = plt.subplots(2, num_cols, figsize=(16, 6))
    fig.suptitle("Sample Chest X-Ray Images", fontsize=16, fontweight="bold")
    indices = random.sample(range(len(dataset)), 2 * num_cols)
    for idx, ax in zip(indices, axes.flat):
        img, label = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(class_names[label], fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "sample_images.png"), dpi=120)
    plt.close()
    print("[INFO] Saved sample_images.png")


show_sample_images(train_dataset, class_names)


class PneumoniaDetector(nn.Module):
    def __init__(self):
        super(PneumoniaDetector, self).__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.block1 = conv_block(3, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


model = PneumoniaDetector().to(DEVICE)
print(f"\n[INFO] Model architecture:\n{model}\n")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] Total parameters     : {total_params:,}")
print(f"[INFO] Trainable parameters : {trainable_params:,}\n")

num_normal = len([s for s in train_dataset.samples if s[1] == 0])
num_pneumonia = len([s for s in train_dataset.samples if s[1] == 1])
pos_weight = torch.tensor([num_normal / num_pneumonia]).to(DEVICE)
print(f"[INFO] Class counts - Normal: {num_normal}, Pneumonia: {num_pneumonia}")
print(f"[INFO] Positive weight for BCE: {pos_weight.item():.4f}\n")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                  factor=0.5, patience=3)


def run_epoch(model, loader, criterion, optimizer=None, is_training=False):
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            running_loss += loss.item() * images.size(0)
            correct += (preds == labels.long()).sum().item()
            total += images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return epoch_loss, epoch_acc, epoch_auc


history = {"train_loss": [], "train_acc": [], "train_auc": [],
           "val_loss": [], "val_acc": [], "val_auc": []}

best_val_auc = 0.0
best_model_path = os.path.join(BASE_DIR, "best_pneumonia_model.pth")

print("=" * 70)
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Train AUC':>9} | "
      f"{'Val Loss':>8} | {'Val Acc':>7} | {'Val AUC':>7}")
print("-" * 70)

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion,
                                         optimizer, is_training=True)
    vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader, criterion)

    scheduler.step(vl_loss)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["train_auc"].append(tr_auc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)
    history["val_auc"].append(vl_auc)

    if vl_auc > best_val_auc:
        best_val_auc = vl_auc
        torch.save(model.state_dict(), best_model_path)

    print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.2%} | {tr_auc:>9.4f} | "
          f"{vl_loss:>8.4f} | {vl_acc:>6.2%} | {vl_auc:>7.4f}")

print("=" * 70)
print(f"[INFO] Best validation AUC: {best_val_auc:.4f}")
print(f"[INFO] Best model saved to {best_model_path}\n")


def plot_training_history(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, history["train_loss"], "o-", label="Train Loss")
    axes[0].plot(epochs_range, history["val_loss"], "s-", label="Val Loss")
    axes[0].set_title("Loss over Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["train_acc"], "o-", label="Train Accuracy")
    axes[1].plot(epochs_range, history["val_acc"], "s-", label="Val Accuracy")
    axes[1].set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history["train_auc"], "o-", label="Train AUC")
    axes[2].plot(epochs_range, history["val_auc"], "s-", label="Val AUC")
    axes[2].set_title("AUC-ROC over Epochs", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "training_history.png"), dpi=120)
    plt.close()
    print("[INFO] Saved training_history.png")


plot_training_history(history)


model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
model.eval()

all_test_labels = []
all_test_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)
        all_test_labels.extend(labels.numpy())
        all_test_probs.extend(probs.cpu().numpy())

all_test_labels = np.array(all_test_labels)
all_test_probs = np.array(all_test_probs)
all_test_preds = (all_test_probs >= 0.5).astype(int)

print("\n" + "=" * 60)
print("          TEST SET CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(all_test_labels, all_test_preds,
                            target_names=class_names))

test_auc = roc_auc_score(all_test_labels, all_test_probs)
test_f1 = f1_score(all_test_labels, all_test_preds)
test_acc = np.mean(all_test_preds == all_test_labels)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test AUC-ROC  : {test_auc:.4f}")
print(f"Test F1-Score : {test_f1:.4f}")
print("=" * 60)


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix - Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()
    print("[INFO] Saved confusion_matrix.png")


plot_confusion_matrix(all_test_labels, all_test_preds, class_names)


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
            label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "roc_curve.png"), dpi=120)
    plt.close()
    print("[INFO] Saved roc_curve.png")


plot_roc_curve(all_test_labels, all_test_probs)


def plot_precision_recall(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="green", lw=2, label="Precision-Recall")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve - Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "precision_recall_curve.png"), dpi=120)
    plt.close()
    print("[INFO] Saved precision_recall_curve.png")


plot_precision_recall(all_test_labels, all_test_probs)


def visualise_predictions(model, dataset, class_names, num_images=8):
    indices = random.sample(range(len(dataset)), num_images)
    fig, axes = plt.subplots(2, num_images // 2, figsize=(16, 8))

    model.eval()
    for i, ax in zip(indices, axes.flat):
        img, true_label = dataset[i]
        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(DEVICE)).squeeze()
            prob = torch.sigmoid(logit).item()
        pred_label = 1 if prob >= 0.5 else 0

        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        color = "green" if pred_label == true_label else "red"
        ax.imshow(img_np)
        ax.set_title(f"True: {class_names[true_label]}\n"
                     f"Pred: {class_names[pred_label]} ({prob:.2f})",
                     fontsize=10, color=color, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Model Predictions on Test Images", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "prediction_samples.png"), dpi=120)
    plt.close()
    print("[INFO] Saved prediction_samples.png")


visualise_predictions(model, test_dataset, class_names)


def plot_class_distribution():
    splits = {"Train": TRAIN_DIR, "Validation": VAL_DIR, "Test": TEST_DIR}
    data = {"Split": [], "Class": [], "Count": []}

    for split_name, split_dir in splits.items():
        for cls in class_names:
            count = len(os.listdir(os.path.join(split_dir, cls)))
            data["Split"].append(split_name)
            data["Class"].append(cls)
            data["Count"].append(count)

    import pandas as pd
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(splits))
    width = 0.35

    normal_counts = df[df["Class"] == "NORMAL"]["Count"].values
    pneumonia_counts = df[df["Class"] == "PNEUMONIA"]["Count"].values

    bars1 = ax.bar(x_pos - width / 2, normal_counts, width, label="NORMAL",
                   color="#4a90d9", edgecolor="black")
    bars2 = ax.bar(x_pos + width / 2, pneumonia_counts, width, label="PNEUMONIA",
                   color="#e74c3c", edgecolor="black")

    ax.bar_label(bars1, padding=3, fontsize=10)
    ax.bar_label(bars2, padding=3, fontsize=10)

    ax.set_xlabel("Dataset Split", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution Across Splits", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(splits.keys()))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "class_distribution.png"), dpi=120)
    plt.close()
    print("[INFO] Saved class_distribution.png")


plot_class_distribution()

print("\n[DONE] All tasks completed successfully!")
print("   Check the generated PNG files for visualisations.")
