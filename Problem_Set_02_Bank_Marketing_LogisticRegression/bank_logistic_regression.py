import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "bank-data", "bank-full.csv")
df = pd.read_csv(DATA_PATH, sep=";")

print("=" * 65)
print("       BANK MARKETING - EXPLORATORY DATA ANALYSIS")
print("=" * 65)
print(f"\nDataset shape : {df.shape}")
print(f"Features      : {df.shape[1] - 1}")
print(f"Target column : 'y' (yes / no)\n")
print(df.info())
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Descriptive Statistics (Numerical) ---")
print(df.describe())

print("\n--- Categorical Feature Unique Values ---")
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
for col in cat_cols:
    print(f"  {col:12s} -> {df[col].nunique()} unique: {df[col].unique()[:8].tolist()}")

print(f"\n--- Missing values ---\n{df.isnull().sum().sum()} missing values found.")

target_counts = df["y"].value_counts()
print(f"\nTarget distribution:\n{target_counts}")
print(f"Subscription rate: {target_counts['yes'] / len(df) * 100:.2f}%\n")

fig, ax = plt.subplots(figsize=(6, 4))
colours = ["#e74c3c", "#2ecc71"]
target_counts.plot(kind="bar", color=colours, edgecolor="black", ax=ax)
for i, (label, count) in enumerate(target_counts.items()):
    ax.text(i, count + 200, str(count), ha="center", fontsize=12, fontweight="bold")
ax.set_title("Term Deposit Subscription Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Subscribed (y)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_xticklabels(["No", "Yes"], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "target_distribution.png"))
plt.close()
print("[INFO] Saved target_distribution.png")

fig, ax = plt.subplots(figsize=(9, 5))
df[df["y"] == "no"]["age"].hist(bins=40, alpha=0.6, color="#e74c3c", label="No", ax=ax)
df[df["y"] == "yes"]["age"].hist(bins=40, alpha=0.6, color="#2ecc71", label="Yes", ax=ax)
ax.set_title("Age Distribution by Subscription Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "age_distribution.png"))
plt.close()
print("[INFO] Saved age_distribution.png")

fig, ax = plt.subplots(figsize=(12, 5))
job_sub = df.groupby(["job", "y"]).size().unstack(fill_value=0)
job_sub.plot(kind="bar", stacked=True, color=colours, edgecolor="black", ax=ax)
ax.set_title("Job Category vs Subscription", fontsize=14, fontweight="bold")
ax.set_xlabel("Job")
ax.set_ylabel("Count")
ax.legend(title="Subscribed")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "job_distribution.png"))
plt.close()
print("[INFO] Saved job_distribution.png")

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, center=0, linewidths=0.5)
ax.set_title("Correlation Heatmap - Numerical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
plt.close()
print("[INFO] Saved correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="y", y="balance", data=df, palette=colours, ax=ax,
            showfliers=False)
ax.set_title("Account Balance by Subscription Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Subscribed")
ax.set_ylabel("Balance")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "balance_boxplot.png"))
plt.close()
print("[INFO] Saved balance_boxplot.png")

fig, ax = plt.subplots(figsize=(9, 5))
df[df["y"] == "no"]["duration"].hist(bins=50, alpha=0.6, color="#e74c3c", label="No", ax=ax)
df[df["y"] == "yes"]["duration"].hist(bins=50, alpha=0.6, color="#2ecc71", label="Yes", ax=ax)
ax.set_title("Call Duration Distribution by Subscription", fontsize=14, fontweight="bold")
ax.set_xlabel("Duration (seconds)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "duration_distribution.png"))
plt.close()
print("[INFO] Saved duration_distribution.png")

fig, ax = plt.subplots(figsize=(8, 5))
edu_sub = pd.crosstab(df["education"], df["y"], normalize="index") * 100
edu_sub.plot(kind="bar", color=colours, edgecolor="black", ax=ax)
ax.set_title("Subscription Rate by Education Level", fontsize=14, fontweight="bold")
ax.set_xlabel("Education")
ax.set_ylabel("Percentage (%)")
ax.legend(title="Subscribed")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "education_subscription.png"))
plt.close()
print("[INFO] Saved education_subscription.png")


print("\n" + "=" * 65)
print("       DATA PREPROCESSING")
print("=" * 65)

df_processed = df.copy()

df_processed["y"] = df_processed["y"].map({"no": 0, "yes": 1})

label_encoders = {}
for col in cat_cols:
    if col == "y":
        continue
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nProcessed dataset shape: {df_processed.shape}")
print(df_processed.head())

X = df_processed.drop("y", axis=1)
y = df_processed["y"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape : {y.shape}")
print(f"Target distribution : {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[INFO] Features scaled using StandardScaler")

print("\n" + "=" * 65)
print("       LOGISTIC REGRESSION - TRAINING & EVALUATION")
print("=" * 65)

logreg = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    random_state=42,
)

logreg.fit(X_train_scaled, y_train)
print("\n[INFO] Model trained successfully.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=cv, scoring="accuracy")
cv_auc = cross_val_score(logreg, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

print(f"\n5-Fold Cross-Validation Results:")
print(f"  Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  AUC-ROC  : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("\n" + "=" * 65)
print("       TEST SET - EVALUATION RESULTS")
print("=" * 65)

test_accuracy = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred_proba)
test_f1 = f1_score(y_test, y_pred)

print(f"\n  Accuracy  : {test_accuracy:.4f}")
print(f"  AUC-ROC   : {test_auc:.4f}")
print(f"  F1-Score  : {test_f1:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix - Logistic Regression", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"))
plt.close()
print("[INFO] Saved confusion_matrix.png")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="darkorange", lw=2,
        label=f"Logistic Regression (AUC = {test_auc:.4f})")
ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
        label="Random Baseline (AUC = 0.50)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve - Logistic Regression", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"))
plt.close()
print("[INFO] Saved roc_curve.png")

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall, precision, color="green", lw=2, label="Precision-Recall Curve")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "precision_recall_curve.png"))
plt.close()
print("[INFO] Saved precision_recall_curve.png")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logreg.coef_[0],
    "Abs_Coefficient": np.abs(logreg.coef_[0]),
}).sort_values("Abs_Coefficient", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colours_bar = ["#e74c3c" if c < 0 else "#2ecc71"
               for c in feature_importance["Coefficient"].values]
ax.barh(feature_importance["Feature"], feature_importance["Coefficient"],
        color=colours_bar, edgecolor="black", height=0.6)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient Value", fontsize=12)
ax.set_title("Logistic Regression Feature Coefficients", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"))
plt.close()
print("[INFO] Saved feature_importance.png")

print("\n--- Threshold Optimisation ---")
best_threshold = 0.5
best_f1_val = 0.0

for thresh in np.arange(0.1, 0.9, 0.01):
    preds_thresh = (y_pred_proba >= thresh).astype(int)
    f1_val = f1_score(y_test, preds_thresh)
    if f1_val > best_f1_val:
        best_f1_val = f1_val
        best_threshold = thresh

print(f"  Optimal threshold : {best_threshold:.2f}")
print(f"  Best F1-Score     : {best_f1_val:.4f}")

y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
print(f"\n--- Results with Optimal Threshold ({best_threshold:.2f}) ---")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_optimal):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred_optimal):.4f}")
print(classification_report(y_test, y_pred_optimal, target_names=["No", "Yes"]))

fig, ax = plt.subplots(figsize=(7, 5))
marital_sub = pd.crosstab(df["marital"], df["y"], normalize="index") * 100
marital_sub.plot(kind="bar", color=colours, edgecolor="black", ax=ax)
ax.set_title("Subscription Rate by Marital Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Marital Status")
ax.set_ylabel("Percentage (%)")
ax.legend(title="Subscribed")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "marital_subscription.png"))
plt.close()
print("[INFO] Saved marital_subscription.png")

fig, ax = plt.subplots(figsize=(12, 5))
month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month_sub = pd.crosstab(df["month"], df["y"])
month_sub = month_sub.reindex(month_order)
month_sub.plot(kind="bar", color=colours, edgecolor="black", ax=ax)
ax.set_title("Subscription by Month of Contact", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Count")
ax.legend(title="Subscribed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "month_subscription.png"))
plt.close()
print("[INFO] Saved month_subscription.png")

print("\n" + "=" * 65)
print("       FINAL SUMMARY")
print("=" * 65)
print(f"""
  Dataset              : Bank Marketing ({df.shape[0]} records, {df.shape[1]} attributes)
  Model                : Logistic Regression (class_weight='balanced')
  Train/Test Split     : 80/20 (stratified)
  Feature Scaling      : StandardScaler
  
  Test Accuracy        : {test_accuracy:.4f}
  Test AUC-ROC         : {test_auc:.4f}
  Test F1-Score        : {test_f1:.4f}
  
  Optimal Threshold    : {best_threshold:.2f}
  F1 at Opt. Threshold : {best_f1_val:.4f}
  
  5-Fold CV Accuracy   : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}
  5-Fold CV AUC        : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}
""")

print("[DONE] All tasks completed! Check the 'figures/' directory for visualisations.")
