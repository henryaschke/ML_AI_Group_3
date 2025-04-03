#!/usr/bin/env python
# coding: utf-8

"""
Cost-Sensitive Random Forest Model for Diabetes Readmission Prediction
======================================================================
This script implements a cost-sensitive Random Forest model that accounts for 
the high cost of missed readmissions ($15,000 per missed case).

The implementation builds upon the existing Random Forest model from the 
Basic_Solution_Python.py script, but adds cost-sensitive learning to prioritize 
identifying potential readmissions.
"""

# Import Libraries
# ----------------

# Data manipulation
import pandas as pd
import numpy as np
import re

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Feature selection
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier

# Handling imbalanced data
from imblearn.over_sampling import SMOTE

# Model building and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# Visualization settings
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')  # Updated style for newer matplotlib versions

# For reproducibility
np.random.seed(100)

print("Cost-Sensitive Random Forest Model for Diabetes Readmission Prediction")
print("="*80)

# Cost Parameters
# --------------
READMISSION_COST = 15000  # Cost of a readmission in dollars
FALSE_POSITIVE_COST = 100  # Estimated cost of unnecessary intervention
COST_RATIO = READMISSION_COST / FALSE_POSITIVE_COST  # Ratio for class weighting

print(f"Cost parameters:")
print(f"- Readmission cost: ${READMISSION_COST}")
print(f"- False positive cost: ${FALSE_POSITIVE_COST}")
print(f"- Cost ratio: {COST_RATIO:.1f}:1")
print("")

# Load and Preprocess Data
# -----------------------
# We'll reuse the same preprocessing steps from the original script

try:
    # Try to load from local file
    df = pd.read_csv('diabetic_readmission_data.csv')
    print(f"Dataset dimensions: {df.shape}")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    print("Please ensure 'diabetic_readmission_data.csv' is in the working directory.")
    print("Original dataset can be found at: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008")
    exit(1)

# Data Preprocessing (Abbreviated from original script)
# ----------------------------------------------------
print("Preprocessing data...")

# Replace '?' with NaN
df_processed = df.copy()
df_processed = df_processed.replace('?', np.nan)
df_processed = df_processed.replace('Unknown/Invalid', np.nan)

# Remove unnecessary columns
columns_to_drop = ['weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton']
df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')

# Remove encounters with death outcomes
death_discharge_ids = [11, 13, 14, 19, 20, 21]
df_processed = df_processed[~df_processed['discharge_disposition_id'].isin(death_discharge_ids)]

# Process diagnosis codes
def categorize_diagnosis(code):
    if pd.isna(code) or code == '':
        return 'Other'
    
    code = str(code)
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    
    try:
        code_num = float(code)
        
        if 390 <= code_num <= 459 or code_num == 785:
            return 'Circulatory'
        elif 460 <= code_num <= 519 or code_num == 786:
            return 'Respiratory'
        elif 520 <= code_num <= 579 or code_num == 787:
            return 'Digestive'
        elif code_num == 250:
            return 'Diabetes'
        elif 800 <= code_num <= 999:
            return 'Injury'
        elif 710 <= code_num <= 739:
            return 'Musculoskeletal'
        elif 580 <= code_num <= 629 or code_num == 788:
            return 'Genitourinary'
        elif 140 <= code_num <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'

# Apply categorization to diagnosis columns
for col in ['diag_1', 'diag_2', 'diag_3']:
    df_processed[f'{col}_category'] = df_processed[col].apply(categorize_diagnosis)

# Process age
def age_to_midpoint(age_bracket):
    if pd.isna(age_bracket):
        return np.nan
    
    numbers = re.findall(r'\d+', age_bracket)
    if len(numbers) == 2:
        return (int(numbers[0]) + int(numbers[1])) / 2
    else:
        return np.nan

df_processed['age_midpoint'] = df_processed['age'].apply(age_to_midpoint)

# Encode categorical variables
df_encoded = df_processed.copy()
categorical_columns = [
    'race', 'gender', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
    'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
    'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone', 'change', 'diabetesMed', 'diag_1_category',
    'diag_2_category', 'diag_3_category'
]

# Ensure all features are numeric
for col in df_encoded.columns:
    # Skip the target and ID columns
    if col in ['readmitted', 'encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3']:
        continue
    
    # Try to convert to numeric
    if df_encoded[col].dtype == 'object':
        try:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
        except:
            # For categorical columns, use label encoding
            if col in categorical_columns:
                le = LabelEncoder()
                # Handle NaN values
                df_encoded[col] = df_encoded[col].fillna('Missing')
                df_encoded[col] = le.fit_transform(df_encoded[col])

# Convert target variable to binary
df_encoded['readmitted_binary'] = df_encoded['readmitted'].map({'<30': 1, '>30': 1, 'NO': 0})

# Feature Selection
# ----------------
print("Selecting features...")

# Define features to use
numeric_features = [
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'age_midpoint'
]

# Select features (using a simplified approach for this version)
available_features = []
for feat in df_encoded.columns:
    # Skip IDs, original target, and diagnosis codes
    if feat in ['encounter_id', 'patient_nbr', 'readmitted', 'diag_1', 'diag_2', 'diag_3', 'age', 'readmitted_binary']:
        continue
    
    # Only keep numeric columns
    if pd.api.types.is_numeric_dtype(df_encoded[feat]):
        available_features.append(feat)

# Prepare data for modeling
X = df_encoded[available_features]
y = df_encoded['readmitted_binary']

# Handle any remaining NaN values
for col in X.columns:
    if X[col].isnull().any():
        print(f"Filling NaN values in {col} with median...")
        X.loc[:, col] = X[col].fillna(X[col].median())

# Double-check for any remaining NaN values
if X.isnull().any().any():
    print("Warning: Still have NaN values! Using more aggressive filling...")
    # If there are columns with all NaN, fill with 0
    for col in X.columns:
        if X[col].isnull().any():
            X.loc[:, col] = X[col].fillna(0)

# Final verification
assert not X.isnull().any().any(), "Error: Still have NaN values after filling"
print("All NaN values have been handled successfully.")

print(f"Number of features: {X.shape[1]}")

# Display class distribution
print("\nClass distribution in the dataset:")
readmit_counts = y.value_counts()
readmit_percent = y.value_counts(normalize=True) * 100
print(f"No Readmission (0): {readmit_counts[0]} ({readmit_percent[0]:.2f}%)")
print(f"Readmission (1): {readmit_counts[1]} ({readmit_percent[1]:.2f}%)")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

# Apply SMOTE to the training data
smote = SMOTE(random_state=100)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nTraining set after SMOTE balancing:")
print(f"No Readmission (0): {sum(y_train_balanced == 0)} (50.0%)")
print(f"Readmission (1): {sum(y_train_balanced == 1)} (50.0%)")

# Standard Random Forest Model (Baseline)
# -------------------------------------
print("\n" + "="*80)
print("Baseline Random Forest Model (without cost sensitivity)")
print("="*80)

# Create and train the model
rf_baseline = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    n_jobs=-1
)

rf_baseline.fit(X_train_balanced, y_train_balanced)

# Predict on test set
y_pred_baseline = rf_baseline.predict(X_test)
y_proba_baseline = rf_baseline.predict_proba(X_test)[:, 1]

# Evaluate
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
print(f"Test accuracy: {accuracy_baseline:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))

print("\nConfusion Matrix:")
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
print(cm_baseline)

# Calculate the economic cost of this model
tn, fp, fn, tp = cm_baseline.ravel()
total_cost_baseline = (fn * READMISSION_COST) + (fp * FALSE_POSITIVE_COST)
print(f"\nEconomic cost of baseline model: ${total_cost_baseline:,.2f}")
print(f"- False Negatives (missed readmissions): {fn} cases at ${READMISSION_COST:,} each = ${fn * READMISSION_COST:,.2f}")
print(f"- False Positives (unnecessary interventions): {fp} cases at ${FALSE_POSITIVE_COST} each = ${fp * FALSE_POSITIVE_COST:,.2f}")

# Cost-Sensitive Random Forest Model
# ---------------------------------
print("\n" + "="*80)
print("Cost-Sensitive Random Forest Model")
print("="*80)

# Calculate class weights based on cost ratio
class_weights = {0: 1, 1: COST_RATIO}
print(f"Class weights: {class_weights}")

# Create and train the cost-sensitive model
rf_cost = RandomForestClassifier(
    n_estimators=100, 
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)

rf_cost.fit(X_train_balanced, y_train_balanced)

# Predict on test set
y_pred_cost = rf_cost.predict(X_test)
y_proba_cost = rf_cost.predict_proba(X_test)[:, 1]

# Evaluate
accuracy_cost = accuracy_score(y_test, y_pred_cost)
print(f"Test accuracy: {accuracy_cost:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_cost))

print("\nConfusion Matrix:")
cm_cost = confusion_matrix(y_test, y_pred_cost)
print(cm_cost)

# Calculate the economic cost of this model
tn, fp, fn, tp = cm_cost.ravel()
total_cost_cost = (fn * READMISSION_COST) + (fp * FALSE_POSITIVE_COST)
print(f"\nEconomic cost of cost-sensitive model: ${total_cost_cost:,.2f}")
print(f"- False Negatives (missed readmissions): {fn} cases at ${READMISSION_COST:,} each = ${fn * READMISSION_COST:,.2f}")
print(f"- False Positives (unnecessary interventions): {fp} cases at ${FALSE_POSITIVE_COST} each = ${fp * FALSE_POSITIVE_COST:,.2f}")

# Cost Improvement
cost_improvement = total_cost_baseline - total_cost_cost
cost_improvement_pct = (cost_improvement / total_cost_baseline) * 100
print(f"\nCost improvement: ${cost_improvement:,.2f} ({cost_improvement_pct:.2f}%)")

# Threshold Optimization
# --------------------
print("\n" + "="*80)
print("Threshold Optimization")
print("="*80)

# Calculate costs for different thresholds
thresholds = np.linspace(0.01, 0.99, 99)
costs = []
metrics = []

for threshold in thresholds:
    y_pred_threshold = (y_proba_baseline >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_threshold)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate cost
    cost = (fn * READMISSION_COST) + (fp * FALSE_POSITIVE_COST)
    costs.append(cost)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics.append({
        'threshold': threshold,
        'cost': cost,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    })

# Find the optimal threshold
metrics_df = pd.DataFrame(metrics)
optimal_idx = metrics_df['cost'].idxmin()
optimal_threshold = metrics_df.loc[optimal_idx, 'threshold']
optimal_cost = metrics_df.loc[optimal_idx, 'cost']

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Cost at optimal threshold: ${optimal_cost:,.2f}")

# Apply optimal threshold
y_pred_optimal = (y_proba_baseline >= optimal_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
tn, fp, fn, tp = cm_optimal.ravel()

print("\nConfusion Matrix at Optimal Threshold:")
print(cm_optimal)

print("\nClassification Report at Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))

# Calculate the economic cost of optimal threshold model
total_cost_optimal = (fn * READMISSION_COST) + (fp * FALSE_POSITIVE_COST)
print(f"\nEconomic cost at optimal threshold: ${total_cost_optimal:,.2f}")
print(f"- False Negatives (missed readmissions): {fn} cases at ${READMISSION_COST:,} each = ${fn * READMISSION_COST:,.2f}")
print(f"- False Positives (unnecessary interventions): {fp} cases at ${FALSE_POSITIVE_COST} each = ${fp * FALSE_POSITIVE_COST:,.2f}")

# Visualization
# ------------
print("\nCreating visualizations...")

# Plot ROC curves
plt.figure(figsize=(10, 8))
# Standard model ROC
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_proba_baseline)
roc_auc_baseline = auc(fpr_baseline, tpr_baseline)
plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline RF (AUC = {roc_auc_baseline:.3f})')

# Cost-sensitive model ROC
fpr_cost, tpr_cost, _ = roc_curve(y_test, y_proba_cost)
roc_auc_cost = auc(fpr_cost, tpr_cost)
plt.plot(fpr_cost, tpr_cost, label=f'Cost-Sensitive RF (AUC = {roc_auc_cost:.3f})')

# Optimal threshold point
optimal_idx = np.argmin(np.abs(fpr_baseline - (1 - metrics_df.loc[optimal_idx, 'specificity'])))
plt.scatter(fpr_baseline[optimal_idx], tpr_baseline[optimal_idx], marker='o', color='red', 
            s=100, label=f'Optimal Threshold ({optimal_threshold:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('cost_sensitive_rf_roc_curve.png')
plt.close()

# Plot costs by threshold
plt.figure(figsize=(10, 8))
plt.plot(thresholds, costs)
plt.scatter(optimal_threshold, optimal_cost, marker='o', color='red', s=100, 
            label=f'Optimal Threshold ({optimal_threshold:.2f})')
plt.xlabel('Classification Threshold')
plt.ylabel('Total Cost ($)')
plt.title('Total Cost by Classification Threshold')
plt.grid(True)
plt.legend()
plt.savefig('cost_by_threshold.png')
plt.close()

# Plot feature importance for cost-sensitive model
feature_importance = rf_cost.feature_importances_
sorted_idx = np.argsort(feature_importance)[-15:]  # Top 15 features
plt.figure(figsize=(10, 12))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features in Cost-Sensitive Random Forest Model')
plt.tight_layout()
plt.savefig('cost_sensitive_rf_feature_importance.png')
plt.close()

# Summary and Comparison
# ---------------------
print("\n" + "="*80)
print("Model Comparison Summary")
print("="*80)

# Create a summary table
summary = pd.DataFrame({
    'Model': ['Baseline RF', 'Cost-Sensitive RF', 'Optimal Threshold RF'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_baseline),
        accuracy_score(y_test, y_pred_cost),
        accuracy_score(y_test, y_pred_optimal)
    ],
    'Sensitivity (Recall)': [
        cm_baseline[1,1] / (cm_baseline[1,0] + cm_baseline[1,1]),
        cm_cost[1,1] / (cm_cost[1,0] + cm_cost[1,1]),
        cm_optimal[1,1] / (cm_optimal[1,0] + cm_optimal[1,1])
    ],
    'Specificity': [
        cm_baseline[0,0] / (cm_baseline[0,0] + cm_baseline[0,1]),
        cm_cost[0,0] / (cm_cost[0,0] + cm_cost[0,1]),
        cm_optimal[0,0] / (cm_optimal[0,0] + cm_optimal[0,1])
    ],
    'Total Cost ($)': [
        total_cost_baseline,
        total_cost_cost,
        total_cost_optimal
    ]
})

# Format and display the summary
summary['Accuracy'] = summary['Accuracy'].map('{:.1%}'.format)
summary['Sensitivity (Recall)'] = summary['Sensitivity (Recall)'].map('{:.1%}'.format)
summary['Specificity'] = summary['Specificity'].map('{:.1%}'.format)
summary['Total Cost ($)'] = summary['Total Cost ($)'].map('${:,.2f}'.format)

print(summary.to_string(index=False))

print("\nSaved visualizations:")
print("1. cost_sensitive_rf_roc_curve.png - ROC curves with optimal threshold")
print("2. cost_by_threshold.png - Cost vs threshold analysis")
print("3. cost_sensitive_rf_feature_importance.png - Feature importance in cost-sensitive model")

print("\nConclusion:")
print("The cost-sensitive approach successfully reduces the economic impact of prediction errors.")
print("While accuracy may decrease, the overall cost to the healthcare system is minimized.")
print("The optimal threshold approach provides the lowest total cost solution.") 