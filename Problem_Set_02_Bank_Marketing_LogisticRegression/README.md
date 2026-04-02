# Problem Set 02 — Bank Term Deposit Subscription Prediction using Logistic Regression

## Overview

This project builds a **Logistic Regression** model to predict whether a bank customer will subscribe to a **term deposit** ('yes') or not ('no'), based on their demographics, account details, and past campaign interactions. The model helps the banking institution target marketing efforts more effectively.

## Dataset

- **Source**: Bank Marketing Data Set  
- **Records**: 45,211 customer entries  
- **Attributes**: 17 features (7 numerical + 9 categorical + 1 target)  
- **Target Variable**: `y` (yes / no — whether the customer subscribed)

### Feature Descriptions

| Feature    | Type        | Description                                      |
|------------|-------------|--------------------------------------------------|
| age        | Numerical   | Customer age                                     |
| job        | Categorical | Type of job                                      |
| marital    | Categorical | Marital status (married, single, divorced)        |
| education  | Categorical | Education level                                  |
| default    | Categorical | Has credit in default? (yes/no)                  |
| balance    | Numerical   | Average yearly balance in euros                  |
| housing    | Categorical | Has housing loan? (yes/no)                       |
| loan       | Categorical | Has personal loan? (yes/no)                      |
| contact    | Categorical | Contact communication type                       |
| day        | Numerical   | Last contact day of the month                    |
| month      | Categorical | Last contact month of the year                   |
| duration   | Numerical   | Last contact duration in seconds                 |
| campaign   | Numerical   | Number of contacts during this campaign           |
| pdays      | Numerical   | Days since last contact from previous campaign    |
| previous   | Numerical   | Number of contacts before this campaign           |
| poutcome   | Categorical | Outcome of the previous marketing campaign        |
| y          | Binary      | **Target** — subscribed to term deposit?          |

### Class Distribution
- **No**: 39,922 (88.3%)
- **Yes**: 5,289 (11.7%)

> The dataset is highly imbalanced — only ~12% of customers subscribed. This is handled using `class_weight='balanced'` in the model.

## Approach & Methodology

### 1. Exploratory Data Analysis (EDA)
- Examined distributions of all 17 features
- Analysed subscription rates across job categories, education levels, marital status, and months
- Visualised correlations between numerical features using a heatmap
- Investigated call duration patterns for subscribers vs non-subscribers
- Generated 8+ visualisation charts for deeper insights

### 2. Data Preprocessing
- **Target encoding**: Mapped 'yes' → 1, 'no' → 0
- **Categorical encoding**: Applied **Label Encoding** to all categorical features (job, marital, education, default, housing, loan, contact, month, poutcome)
- **Missing values**: No missing values found in the dataset
- **Feature scaling**: Applied **StandardScaler** (zero mean, unit variance) to all features — essential for Logistic Regression which is sensitive to feature magnitudes

### 3. Model Training
- **Algorithm**: Logistic Regression (scikit-learn)
- **Solver**: L-BFGS (efficient for small-to-medium datasets)
- **Regularisation**: C=1.0 (L2 penalty by default)
- **Class balancing**: `class_weight='balanced'` to automatically adjust weights inversely proportional to class frequencies
- **Train/Test split**: 80/20 with stratified sampling to preserve class ratios
- **Cross-validation**: 5-fold Stratified K-Fold for robust performance estimation

### 4. Evaluation Metrics
- **Accuracy**: Overall correctness
- **AUC-ROC**: Ability to distinguish between classes across all thresholds
- **F1-Score**: Harmonic mean of precision and recall (especially important for imbalanced data)
- **Precision & Recall**: Per-class performance
- **Confusion Matrix**: Breakdown of true/false positives and negatives

### 5. Threshold Optimisation
- Conducted a sweep across thresholds from 0.1 to 0.9 to find the value that maximises the F1-score, rather than using the default 0.5 cutoff

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execute
```bash
python bank_logistic_regression.py
```

The script will:
1. Load and explore the dataset (EDA)
2. Preprocess and encode features
3. Train the Logistic Regression model
4. Perform 5-fold cross-validation
5. Evaluate on the test set
6. Find the optimal classification threshold
7. Generate the following visualisations in the `figures/` directory:
   - `target_distribution.png` — Bar chart of yes/no counts
   - `age_distribution.png` — Age histogram by subscription status
   - `job_distribution.png` — Job categories vs subscription
   - `correlation_heatmap.png` — Numerical feature correlations
   - `balance_boxplot.png` — Account balance comparison
   - `duration_distribution.png` — Call duration patterns
   - `education_subscription.png` — Subscription rate by education
   - `confusion_matrix.png` — Test set confusion matrix
   - `roc_curve.png` — ROC curve with AUC
   - `precision_recall_curve.png` — Precision-recall trade-off
   - `feature_importance.png` — Logistic regression coefficients
   - `marital_subscription.png` — Subscription by marital status
   - `month_subscription.png` — Subscription by contact month

## Key Design Decisions

1. **Logistic Regression as specified**: Despite the dataset's complexity, Logistic Regression provides an interpretable baseline and is well-suited for binary classification. Its coefficients directly reveal feature importance.

2. **class_weight='balanced'**: With only 11.7% positive labels, a naive classifier could achieve 88% accuracy by always predicting "no". The balanced weighting ensures the model pays equal attention to both classes.

3. **Label Encoding over One-Hot**: Used Label Encoding for categorical variables to keep the feature count manageable. While One-Hot encoding is often preferred for Logistic Regression, Label Encoding works adequately here and avoids dimensionality explosion.

4. **StandardScaler**: Logistic Regression with L-BFGS relies on gradient-based optimisation, which converges faster when features are on similar scales. Scaling prevents features like `balance` (thousands) from dominating `age` (tens).

5. **Threshold optimisation**: The default 0.5 threshold is rarely optimal for imbalanced datasets. Sweeping thresholds and selecting the one maximising F1 provides a more practical decision boundary.

## Findings

- **Duration** is the strongest predictor — longer call durations correlate strongly with subscription. However, note that in practice this feature is only known after the call ends and should not be used for truly predictive (pre-call) modelling.
- **Previous campaign outcome** (`poutcome`) has significant predictive power — customers who previously said "yes" are likely to subscribe again.
- **Month of contact** matters — certain months (e.g., March, September, October) show higher subscription rates.
- **Age** shows a bimodal pattern — very young (<25) and older (>60) customers are more likely to subscribe.
- The **class imbalance** is the biggest challenge — balanced class weights significantly improve recall for the "yes" class at a moderate cost to overall accuracy.
- **F1-score improvement** with threshold optimisation demonstrates the importance of not relying on default classification boundaries.

## Directory Structure

```
Problem_Set_02_Bank_Marketing_LogisticRegression/
├── bank_logistic_regression.py    # Main analysis and modelling script
├── README.md                      # This file
├── bank-data/
│   └── bank-full.csv              # Dataset (semicolon-separated)
└── figures/                       # Generated visualisations
    ├── target_distribution.png
    ├── age_distribution.png
    ├── job_distribution.png
    ├── correlation_heatmap.png
    ├── balance_boxplot.png
    ├── duration_distribution.png
    ├── education_subscription.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── feature_importance.png
    ├── marital_subscription.png
    └── month_subscription.png
```
