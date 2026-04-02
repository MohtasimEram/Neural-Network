# Problem Set 02 - Bank Term Deposit Prediction using Logistic Regression

## Objective
Predict whether a bank customer will subscribe to a term deposit using Logistic Regression.

## Dataset
- Bank Marketing Dataset with 45,211 records and 17 attributes
- Target variable: y (yes/no - subscribed to term deposit)
- Class distribution: No = 39,922 (88.3%), Yes = 5,289 (11.7%)
- The dataset is imbalanced which affects model performance

## Approach
Used Logistic Regression from scikit-learn with balanced class weights to handle the imbalance.

### Preprocessing
- Checked for missing values (none found)
- Encoded categorical variables using LabelEncoder
- Encoded target variable (yes=1, no=0)
- Applied StandardScaler for feature scaling
- 80/20 train-test split with stratification

### Model Configuration
- Solver: lbfgs
- Max iterations: 1000
- Regularization: C=1.0
- Class weight: balanced
- 5-Fold Stratified Cross-Validation for performance estimation

### Evaluation
- Accuracy, AUC-ROC, F1-Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature coefficient analysis
- Threshold optimization to find best F1 cutoff

## How to Run
```
pip install pandas numpy matplotlib seaborn scikit-learn
python bank_logistic_regression.py
```

Make sure the bank-data/ folder with bank-full.csv is in the same directory.

## Results
- Test Accuracy: ~81%
- AUC-ROC: ~0.876
- Optimized threshold improves F1-score over the default 0.5

## Findings
- Duration of the last call is the strongest predictor of subscription
- Previous campaign outcome also has strong predictive power
- The class imbalance is the biggest challenge - using balanced weights improves recall for the minority class
- Threshold optimization helps get better F1-score than the default 0.5 cutoff
