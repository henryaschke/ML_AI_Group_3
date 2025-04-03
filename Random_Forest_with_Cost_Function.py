#!/usr/bin/env python
# coding: utf-8

"""
Cost-Sensitive Random Forest Model with Lower Cost Scenario
==========================================================
This script implements a Random Forest model with a different cost structure:
- Lower cost of missed readmission: $11,000 (instead of $15,000)
- Much lower cost of prevention intervention: $2,800 (instead of $6,600 or $8,400)

This represents a scenario where readmissions are less costly but interventions are 
significantly more affordable, potentially making them economically viable for more patients.
"""

# Import Libraries
# ----------------
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc
)

# Visualization settings
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')  # Updated style for newer matplotlib versions

# For reproducibility
np.random.seed(100)

print("Cost-Sensitive Random Forest Model with Lower Cost Scenario")
print("="*80)

# Cost Parameters
# --------------
READMISSION_COST = 11000  # Lower cost of a readmission in dollars
INTERVENTION_COST = 4000  # Much lower cost of preventive intervention

print(f"Cost parameters:")
print(f"- Readmission cost: ${READMISSION_COST}")
print(f"- Intervention cost: ${INTERVENTION_COST}")
print(f"- Net benefit per prevented readmission: ${READMISSION_COST - INTERVENTION_COST}")
print(f"- Cost ratio (readmission:intervention): {READMISSION_COST/INTERVENTION_COST:.1f}:1")
print("")

# Load and Preprocess Data
# -----------------------
try:
    df = pd.read_csv('diabetic_readmission_data.csv')
    print(f"Dataset dimensions: {df.shape}")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    print("Please ensure 'diabetic_readmission_data.csv' is in the working directory.")
    exit(1)

# Data Preprocessing 
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
print("Selecting features...")

# Select features
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

# Simple Cost Calculation Function
# -------------------------------
def calculate_costs(y_true, y_pred, verbose=True):
    """
    Calculate direct costs based on the confusion matrix.
    
    Assumptions:
    - True Positives (TP): We intervene, cost = INTERVENTION_COST
    - False Positives (FP): We intervene unnecessarily, cost = INTERVENTION_COST
    - False Negatives (FN): We miss a readmission, cost = READMISSION_COST
    - True Negatives (TN): No intervention, no readmission, cost = 0
    """
    # Get confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate costs for each category
    cost_tp = tp * INTERVENTION_COST
    cost_fp = fp * INTERVENTION_COST
    cost_fn = fn * READMISSION_COST
    cost_tn = 0  # No cost
    
    # Total cost
    total_cost = cost_tp + cost_fp + cost_fn + cost_tn
    
    # Cost breakdown
    if verbose:
        print(f"\nCost Analysis:")
        print(f"- TP (correct interventions): {tp} × ${INTERVENTION_COST} = ${cost_tp:,.2f}")
        print(f"- FP (unnecessary interventions): {fp} × ${INTERVENTION_COST} = ${cost_fp:,.2f}")
        print(f"- FN (missed readmissions): {fn} × ${READMISSION_COST} = ${cost_fn:,.2f}")
        print(f"- TN (correct non-interventions): {tn} × $0 = $0.00")
        print(f"- Total cost: ${total_cost:,.2f}")
    
    # Calculate minimum possible cost (perfect prediction)
    min_cost = sum(y_true) * INTERVENTION_COST
    
    # Calculate cost of doing nothing (all negatives)
    do_nothing_cost = sum(y_true) * READMISSION_COST
    
    if verbose:
        print(f"\nBenchmarks:")
        print(f"- Perfect prediction cost: ${min_cost:,.2f}")
        print(f"- Cost of doing nothing: ${do_nothing_cost:,.2f}")
        
        if total_cost < do_nothing_cost:
            savings = do_nothing_cost - total_cost
            print(f"- Savings vs. doing nothing: ${savings:,.2f} ({savings/do_nothing_cost:.1%})")
        else:
            loss = total_cost - do_nothing_cost
            print(f"- Loss vs. doing nothing: ${loss:,.2f} ({loss/do_nothing_cost:.1%})")
    
    return {
        'total_cost': total_cost,
        'min_cost': min_cost,
        'do_nothing_cost': do_nothing_cost,
        'confusion_matrix': cm,
        'cost_detail': {
            'tp': cost_tp,
            'fp': cost_fp,
            'fn': cost_fn,
            'tn': cost_tn
        }
    }

# Standard Random Forest Model (Baseline)
# --------------------------------------
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

# Calculate costs for baseline model
baseline_costs = calculate_costs(y_test, y_pred_baseline)

# Threshold Optimization for Cost Minimization
# -------------------------------------------
print("\n" + "="*80)
print("Threshold Optimization for Cost Minimization")
print("="*80)

# Calculate costs for different thresholds
thresholds = np.linspace(0.01, 0.99, 99)
costs = []
metrics = []

print("Finding optimal threshold based on economic costs...")
for threshold in thresholds:
    # Get predictions at this threshold
    y_pred_threshold = (y_proba_baseline >= threshold).astype(int)
    
    # Calculate costs
    cost_data = calculate_costs(y_test, y_pred_threshold, verbose=False)
    costs.append(cost_data['total_cost'])
    
    # Calculate metrics
    cm = cost_data['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics.append({
        'threshold': threshold,
        'total_cost': cost_data['total_cost'],
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
optimal_idx = metrics_df['total_cost'].idxmin()  # Minimize total cost
optimal_threshold = metrics_df.loc[optimal_idx, 'threshold']
optimal_cost = metrics_df.loc[optimal_idx, 'total_cost']

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Cost at optimal threshold: ${optimal_cost:,.2f}")

# Apply optimal threshold
y_pred_optimal = (y_proba_baseline >= optimal_threshold).astype(int)

print("\nConfusion Matrix at Optimal Threshold:")
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
print(cm_optimal)

print("\nClassification Report at Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))

# Calculate detailed costs for optimal threshold
optimal_costs = calculate_costs(y_test, y_pred_optimal)

# Calculate cost-weighted class weights for a cost-sensitive model
# ---------------------------------------------------------------
print("\n" + "="*80)
print("Cost-Weighted Random Forest Model")
print("="*80)

# Calculate class weights
class_weights = {
    0: 1.0,  # Weight for negative class
    1: READMISSION_COST / INTERVENTION_COST  # Weight for positive class
}

print(f"Class weights based on cost ratio: {class_weights}")

# Train a cost-weighted model
rf_weighted = RandomForestClassifier(
    n_estimators=100, 
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)

rf_weighted.fit(X_train_balanced, y_train_balanced)

# Predict on test set
y_pred_weighted = rf_weighted.predict(X_test)

# Evaluate
print("\nClassification Report for Cost-Weighted Model:")
print(classification_report(y_test, y_pred_weighted))

print("\nConfusion Matrix for Cost-Weighted Model:")
cm_weighted = confusion_matrix(y_test, y_pred_weighted)
print(cm_weighted)

# Calculate costs for weighted model
weighted_costs = calculate_costs(y_test, y_pred_weighted)

# Visualization
# ------------
print("\nCreating visualizations...")

# Plot ROC curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_proba_baseline)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})')

# Optimal threshold point on ROC curve
optimal_fpr_idx = np.argmin(np.abs(thresholds - optimal_threshold))
optimal_fpr = metrics_df.loc[optimal_idx, 'fp'] / (metrics_df.loc[optimal_idx, 'fp'] + metrics_df.loc[optimal_idx, 'tn'])
optimal_tpr = metrics_df.loc[optimal_idx, 'tp'] / (metrics_df.loc[optimal_idx, 'tp'] + metrics_df.loc[optimal_idx, 'fn'])
plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', s=100, 
            label=f'Optimal Threshold ({optimal_threshold:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Economically Optimal Threshold')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('lower_cost_roc_curve.png')
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
plt.savefig('lower_cost_by_threshold.png')
plt.close()

# Cost comparison between all models
models = ['Baseline RF', 'Optimal Threshold RF', 'Cost-Weighted RF']
model_costs = [baseline_costs['total_cost'], optimal_costs['total_cost'], weighted_costs['total_cost']]
baseline_cost = baseline_costs['do_nothing_cost']

plt.figure(figsize=(12, 8))
bars = plt.bar(models, model_costs)
plt.axhline(y=baseline_cost, color='r', linestyle='-', label='Do Nothing Cost')
plt.xlabel('Model')
plt.ylabel('Total Cost ($)')
plt.title('Cost Comparison Between Models')
plt.grid(True, axis='y')
plt.legend()

# Add values on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    savings = baseline_cost - height
    savings_pct = (savings / baseline_cost) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height + 5000000,
             f'${height:,.0f}\n(-${savings:,.0f}, {savings_pct:.1f}%)',
             ha='center', va='bottom', rotation=0)

plt.tight_layout()
plt.savefig('lower_cost_model_comparison.png')
plt.close()

# Cost-threshold relation with intervention percentage
plt.figure(figsize=(12, 8))

# Primary axis for cost
ax1 = plt.gca()
ax1.plot(thresholds, metrics_df['total_cost'] / 1e6, 'b-', linewidth=2, label='Total Cost ($ millions)')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Cost ($ millions)')
ax1.tick_params(axis='y', labelcolor='b')
ax1.axvline(x=optimal_threshold, color='purple', linestyle='--', 
            label=f'Optimal Threshold = {optimal_threshold:.2f}')

# Secondary axis for percentage of population
ax2 = ax1.twinx()
intervention_pcts = [(metrics_df.loc[i, 'tp'] + metrics_df.loc[i, 'fp']) / len(y_test) * 100 
                       for i in range(len(metrics_df))]
ax2.plot(thresholds, intervention_pcts, 'g-', linewidth=2, label='% of Population Receiving Intervention')
ax2.set_ylabel('% of Population')
ax2.tick_params(axis='y', labelcolor='g')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Total Cost and Intervention Coverage by Threshold')
plt.grid(True)
plt.tight_layout()
plt.savefig('lower_cost_threshold_coverage.png')
plt.close()

# Summary
# ------
print("\n" + "="*80)
print("Summary of Results with Lower Cost Scenario")
print("="*80)

# Calculate cost vs. doing nothing for all models
baseline_vs_nothing = baseline_costs['total_cost'] - baseline_costs['do_nothing_cost']
optimal_vs_nothing = optimal_costs['total_cost'] - optimal_costs['do_nothing_cost']
weighted_vs_nothing = weighted_costs['total_cost'] - weighted_costs['do_nothing_cost']

# Calculate intervention percentages
baseline_intervention_pct = (cm_baseline[0,1] + cm_baseline[1,1]) / len(y_test) * 100
optimal_intervention_pct = (cm_optimal[0,1] + cm_optimal[1,1]) / len(y_test) * 100
weighted_intervention_pct = (cm_weighted[0,1] + cm_weighted[1,1]) / len(y_test) * 100

# Create summary dataframe
summary = pd.DataFrame({
    'Model': ['Baseline RF', 'Optimal Threshold RF', 'Cost-Weighted RF', 'Do Nothing'],
    'Threshold': [0.5, optimal_threshold, 'Cost-weighted', 'N/A'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_baseline),
        accuracy_score(y_test, y_pred_optimal),
        accuracy_score(y_test, y_pred_weighted),
        accuracy_score(y_test, np.zeros_like(y_test))  # All negative
    ],
    'Sensitivity': [
        cm_baseline[1,1] / (cm_baseline[1,0] + cm_baseline[1,1]),
        cm_optimal[1,1] / (cm_optimal[1,0] + cm_optimal[1,1]),
        cm_weighted[1,1] / (cm_weighted[1,0] + cm_weighted[1,1]),
        0.0  # Do nothing has 0 sensitivity
    ],
    'Specificity': [
        cm_baseline[0,0] / (cm_baseline[0,0] + cm_baseline[0,1]),
        cm_optimal[0,0] / (cm_optimal[0,0] + cm_optimal[0,1]),
        cm_weighted[0,0] / (cm_weighted[0,0] + cm_weighted[0,1]),
        1.0  # Do nothing has 100% specificity
    ],
    '% Intervened': [
        baseline_intervention_pct,
        optimal_intervention_pct,
        weighted_intervention_pct,
        0.0  # Do nothing: 0% intervention
    ],
    'Total Cost ($)': [
        baseline_costs['total_cost'],
        optimal_costs['total_cost'],
        weighted_costs['total_cost'],
        baseline_costs['do_nothing_cost']  # Cost of doing nothing
    ],
    'Savings ($)': [
        -baseline_vs_nothing if baseline_vs_nothing < 0 else baseline_vs_nothing,
        -optimal_vs_nothing if optimal_vs_nothing < 0 else optimal_vs_nothing,
        -weighted_vs_nothing if weighted_vs_nothing < 0 else weighted_vs_nothing,
        0
    ],
    'Savings (%)': [
        ((-baseline_vs_nothing if baseline_vs_nothing < 0 else baseline_vs_nothing) / baseline_costs['do_nothing_cost']) * 100,
        ((-optimal_vs_nothing if optimal_vs_nothing < 0 else optimal_vs_nothing) / baseline_costs['do_nothing_cost']) * 100,
        ((-weighted_vs_nothing if weighted_vs_nothing < 0 else weighted_vs_nothing) / weighted_costs['do_nothing_cost']) * 100,
        0
    ]
})

# Format for display
summary['Accuracy'] = summary['Accuracy'].map('{:.1%}'.format)
summary['Sensitivity'] = summary['Sensitivity'].map('{:.1%}'.format)
summary['Specificity'] = summary['Specificity'].map('{:.1%}'.format)
summary['% Intervened'] = summary['% Intervened'].map('{:.1f}%'.format)
summary['Threshold'] = summary['Threshold'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
summary['Total Cost ($)'] = summary['Total Cost ($)'].map('${:,.2f}'.format)
summary['Savings ($)'] = summary['Savings ($)'].map('${:,.2f}'.format)
summary['Savings (%)'] = summary['Savings (%)'].map('{:.1f}%'.format)

print(summary.to_string(index=False))

print("\nSaved visualizations:")
print("1. lower_cost_roc_curve.png - ROC curve with optimal threshold")
print("2. lower_cost_by_threshold.png - Cost vs threshold analysis")
print("3. lower_cost_model_comparison.png - Cost comparison between models")
print("4. lower_cost_threshold_coverage.png - Cost and intervention coverage by threshold")

print("\nConclusion:")
do_nothing_cost = baseline_costs['do_nothing_cost']
best_model_idx = [baseline_costs['total_cost'], optimal_costs['total_cost'], weighted_costs['total_cost']].index(
    min([baseline_costs['total_cost'], optimal_costs['total_cost'], weighted_costs['total_cost']])
)
best_model_name = ['Baseline RF', 'Optimal Threshold RF', 'Cost-Weighted RF'][best_model_idx]
best_model_cost = [baseline_costs['total_cost'], optimal_costs['total_cost'], weighted_costs['total_cost']][best_model_idx]
best_savings = do_nothing_cost - best_model_cost

print(f"With the new cost structure (readmission: ${READMISSION_COST:,}, intervention: ${INTERVENTION_COST:,}):")
print(f"1. The best model is the {best_model_name}")
print(f"2. It provides savings of ${best_savings:,.2f} ({best_savings/do_nothing_cost:.1%} reduction)")

if best_model_name == 'Optimal Threshold RF':
    intervention_pct = optimal_intervention_pct
    print(f"3. The optimal threshold is {optimal_threshold:.4f}")
    print(f"4. With this threshold, we would intervene with {intervention_pct:.1f}% of patients")
    
    # Calculate return on investment
    tp = cm_optimal[1,1]  # True positives
    fp = cm_optimal[0,1]  # False positives
    investment = (tp + fp) * INTERVENTION_COST
    return_avoided = tp * READMISSION_COST
    roi = (return_avoided - investment) / investment
    print(f"5. The ROI on interventions is {roi:.1%}")
    
    # Calculate break-even intervention cost
    sensitivity_value = float(summary['Sensitivity'][1].rstrip('%')) / 100
    break_even_cost = READMISSION_COST * sensitivity_value
    print(f"6. The break-even intervention cost would be ${break_even_cost:.2f}")
elif best_model_name == 'Cost-Weighted RF':
    print(f"3. This model uses class weights of {class_weights}")
    intervention_pct = weighted_intervention_pct
    print(f"4. With this approach, we would intervene with {intervention_pct:.1f}% of patients")
else:
    print(f"3. With low intervention costs, even the standard threshold of 0.5 is economically beneficial") 