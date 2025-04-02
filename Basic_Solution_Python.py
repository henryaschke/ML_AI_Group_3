#!/usr/bin/env python
# coding: utf-8

"""
Predictive Modeling for Hospital Readmission of Diabetic Patients
=================================================================
This script is a Python translation of the R-based Basic_Solution notebook.
The objective is to predict the likelihood of diabetic patients being readmitted to hospitals.
We'll analyze data from 130 US hospitals over a 10-year period.
"""

# %%
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
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Model building and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Visualization settings
sns.set(style="whitegrid")
# Use a valid style from the available list
plt.style.use('seaborn-v0_8-whitegrid')

# For reproducibility
np.random.seed(100)

# %%
# Dataset Description
# ------------------
# - Collected from 130 hospitals in the U.S. over 10 years (1999-2008)
# - 101,766 instances, each representing one hospital encounter
# - Focused on diabetic patients
# - 50 attributes including demographics, diagnoses, procedures, lab tests, medication, etc.
# - Target variable: 'readmitted' (whether a patient was readmitted within 30 days, after 30 days, or not readmitted)

# %%
# Load the dataset
try:
    # Try to load from local file first
    df = pd.read_csv('diabetic_readmission_data.csv')
    print(f"Dataset dimensions: {df.shape}")
    print("\nDataset information:")
    print(df.info())
except FileNotFoundError:
    # If not found, provide instructions
    print("Please make sure the diabetic_readmission_data.csv file is in the working directory.")
    print("Original dataset can be found at: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008")

# %%
# Display summary statistics
print("Summary statistics:")
print(df.describe(include='all').T)

# %%
# Check the distribution of the target variable
readmission_counts = df['readmitted'].value_counts()
print("Readmission Counts:")
print(readmission_counts)

# Visualize the distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='readmitted', data=df)
plt.title('Distribution of Readmission')
plt.xlabel('Readmission Category')
plt.ylabel('Count')
plt.show()

# %%
# Dataset Preparation
# ------------------

# Check for missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(f"Total columns with missing values: {len(missing_values)}")
print("\nColumns with missing values:")
print(missing_values)

# Also check for '?' values which are common placeholders for missing data
question_mark_values = {col: (df[col] == '?').sum() for col in df.select_dtypes(include=['object']).columns}
question_mark_values = {k: v for k, v in question_mark_values.items() if v > 0}
print("\nColumns with '?' values:")
print(question_mark_values)

# %%
# Handle Missing Values and Transform Data Types
# Make a copy of the original dataframe
df_processed = df.copy()

# Replace '?' with NaN
df_processed = df_processed.replace('?', np.nan)

# Replace 'Unknown/Invalid' with NaN
df_processed = df_processed.replace('Unknown/Invalid', np.nan)

# Check missing values after replacement
missing_after = df_processed.isnull().sum()
missing_after = missing_after[missing_after > 0]
print("Columns with missing values after replacing '?' and 'Unknown/Invalid':")
print(missing_after)

# %%
# Remove Unnecessary Columns
# Based on the original R analysis, we'll remove columns that don't contribute significantly to the model
columns_to_drop = ['weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton']
df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')

print(f"Shape after dropping unnecessary columns: {df_processed.shape}")

# %%
# Remove Encounters with Death Outcomes
# Since the objective is to predict readmissions, we'll exclude patients who died during their hospital stay
death_discharge_ids = [11, 13, 14, 19, 20, 21]
df_processed = df_processed[~df_processed['discharge_disposition_id'].isin(death_discharge_ids)]

print(f"Shape after removing death-related encounters: {df_processed.shape}")

# %%
# Collapse Diagnosis Codes into Categories
# Following the R analysis, we'll categorize the ICD-9 diagnosis codes into meaningful disease categories
def categorize_diagnosis(code):
    if pd.isna(code) or code == '':
        return 'Other'
    
    # Convert to string if it's not already
    code = str(code)
    
    # Check if code starts with 'V' or 'E'
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    
    # Try to convert to number for range checks
    try:
        code_num = float(code)
        
        # Categorize based on ICD-9 ranges
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

# Show distribution of primary diagnosis categories
diag_1_distribution = df_processed['diag_1_category'].value_counts()
print("Primary Diagnosis Distribution:")
print(diag_1_distribution)

# Visualize
plt.figure(figsize=(12, 6))
sns.countplot(x='diag_1_category', data=df_processed, order=diag_1_distribution.index)
plt.title('Distribution of Primary Diagnosis Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Handle Age Variable
# Convert age brackets to numeric midpoints for easier analysis
def age_to_midpoint(age_bracket):
    if pd.isna(age_bracket):
        return np.nan
    
    # Extract numbers from the bracket
    numbers = re.findall(r'\d+', age_bracket)
    if len(numbers) == 2:
        return (int(numbers[0]) + int(numbers[1])) / 2
    else:
        return np.nan

# Apply conversion
df_processed['age_midpoint'] = df_processed['age'].apply(age_to_midpoint)

# Visualize age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_processed['age_midpoint'], kde=True, bins=10)
plt.title('Distribution of Patient Ages')
plt.xlabel('Age (Midpoint of Bracket)')
plt.ylabel('Frequency')
plt.show()

# %%
# Convert Categorical Variables for Modeling
# Prepare categorical variables for machine learning by encoding them appropriately
categorical_columns = [
    'race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
    'diag_1_category', 'diag_2_category', 'diag_3_category'
]

# Create a copy for encoding
df_encoded = df_processed.copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    if col in df_encoded.columns:
        le = LabelEncoder()
        # Handle NaN values
        df_encoded[col] = df_encoded[col].fillna('Missing')
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

# Convert target variable: 'readmitted' to binary (was the patient readmitted within 30 days or not)
df_encoded['readmitted_binary'] = df_encoded['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})

# After processing but before feature selection, ensure all columns are numeric
print("\nEnsuring all features are numeric...")
for col in df_encoded.columns:
    # Skip the target variable and columns we'll exclude anyway
    if col in ['readmitted', 'readmitted_binary', 'encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3']:
        continue
    
    # If column is object type, try to convert to numeric
    if df_encoded[col].dtype == 'object':
        try:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
            print(f"Converted {col} to numeric.")
        except:
            print(f"Could not convert {col} to numeric. Will be excluded from analysis.")
    
    # Fill any remaining NaN values with column median for numeric columns
    if pd.api.types.is_numeric_dtype(df_encoded[col]):
        if df_encoded[col].isna().any():
            median_val = df_encoded[col].median()
            df_encoded[col] = df_encoded[col].fillna(median_val)
            print(f"Filled NaN values in {col} with median: {median_val}")

# Show data after preprocessing
print(f"\nFinal dataset shape after ensuring numeric values: {df_encoded.shape}")
print("Data types after conversion:")
print(df_encoded.dtypes.value_counts())

# %%
# Feature Selection
# ----------------
# In the original R notebook, a Boruta algorithm was used for feature selection. 
# In Python, we'll use a similar approach with Random Forest feature importance and 
# Recursive Feature Elimination (RFE)

# Prepare data for feature selection
# Define numeric features that are already in the right format
numeric_features = [
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'age_midpoint'
]

# Make sure all categorical features are properly encoded as integers
categorical_features = [
    'race', 'gender', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
    'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
    'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone', 'change', 'diabetesMed', 'diag_1_category',
    'diag_2_category', 'diag_3_category'
]

# Verify all features are available and are integers after encoding
available_features = []
for feat in numeric_features:
    if feat in df_encoded.columns and pd.api.types.is_numeric_dtype(df_encoded[feat]):
        available_features.append(feat)
    elif feat in df_encoded.columns:
        print(f"Converting {feat} to numeric...")
        df_encoded[feat] = pd.to_numeric(df_encoded[feat], errors='coerce')
        df_encoded[feat] = df_encoded[feat].fillna(df_encoded[feat].median())
        available_features.append(feat)
    else:
        print(f"Feature {feat} not found in dataframe, skipping.")

for feat in categorical_features:
    if feat in df_encoded.columns and pd.api.types.is_integer_dtype(df_encoded[feat]):
        available_features.append(feat)
    elif feat in df_encoded.columns:
        print(f"Encoding {feat} as integer...")
        # If it's a categorical variable, encode it
        df_encoded[feat] = pd.Categorical(df_encoded[feat]).codes
        available_features.append(feat)
    else:
        print(f"Feature {feat} not found in dataframe, skipping.")

# Create feature matrix and target vector
X = df_encoded[available_features]
y = df_encoded['readmitted_binary']

print(f"Number of features before selection: {X.shape[1]}")
print("Feature columns:", available_features)

# Check for any remaining issues
print("\nChecking for any remaining issues...")
null_counts = X.isnull().sum()
if null_counts.sum() > 0:
    print("Columns with null values:")
    print(null_counts[null_counts > 0])
    # Fill nulls with median values
    for col in X.columns[X.isnull().any()]:
        X[col] = X[col].fillna(X[col].median())
    print("Null values filled with median values.")
else:
    print("No null values found in feature set.")

# %%
# Random Forest Feature Importance
print("\nStarting Random Forest feature importance calculation...")

# Final sanity check for any non-numeric values
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        print(f"Warning: Column {col} is not numeric. Converting...")
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    if X[col].isnull().any():
        print(f"Warning: Column {col} has null values. Filling with median...")
        X[col] = X[col].fillna(X[col].median())

# Convert X to numpy array and check for any remaining issues
print("Converting to numpy array...")
X_array = X.to_numpy()
if np.isnan(X_array).any():
    print("Warning: NaN values found in the array. Replacing with 0...")
    X_array = np.nan_to_num(X_array)

print("Initializing RandomForestClassifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)

print("Fitting RandomForestClassifier...")
rf.fit(X_array, y)
print("RandomForestClassifier fitting complete!")

# Get feature importances
print("Calculating feature importances...")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Create a dataframe with feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Display top 15 features
print("Top 15 features by importance:")
print(feature_importance_df.head(15))

# %%
# Recursive Feature Elimination (RFE)
# Select features equivalent to the Boruta algorithm in the R notebook
# For simplicity, we'll select top 20 features
rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    n_features_to_select=20,
    step=1,
    verbose=0
)

rfe.fit(X, y)

# Get selected features
selected_features = [X.columns[i] for i in range(len(X.columns)) if rfe.support_[i]]
print(f"Number of features selected: {len(selected_features)}")
print("Selected features:")
print(selected_features)

# Create dataset with selected features
X_selected = X[selected_features]

# Display distribution of target variable
print("\nTarget distribution:")
print(y.value_counts())
print(f"Class 1 percentage: {y.mean() * 100:.2f}%")

# %%
# Analytical Techniques
# --------------------

# Split the Dataset into Training and Test Data
# We'll use an 80:20 train-test split with stratification to ensure balanced class distribution
np.random.seed(100)

# Split the data into training and test sets (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=100, stratify=y
)

# Check class distribution in training set
print("Training set target distribution:")
print(pd.Series(y_train).value_counts())
print(f"Class 1 percentage in training set: {pd.Series(y_train).mean() * 100:.2f}%")

# Check class distribution in test set
print("\nTest set target distribution:")
print(pd.Series(y_test).value_counts())
print(f"Class 1 percentage in test set: {pd.Series(y_test).mean() * 100:.2f}%")

# %%
# Data Balancing with SMOTE
# Since the classes are imbalanced, we'll use SMOTE (Synthetic Minority Over-sampling Technique)
# to balance the training data. This is equivalent to the ROSE method used in the R notebook
smote = SMOTE(random_state=100)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Check balanced class distribution
print("Balanced training set target distribution:")
print(pd.Series(y_train_balanced).value_counts())
print(f"Class 1 percentage in balanced training set: {pd.Series(y_train_balanced).mean() * 100:.2f}%")

# Show shape comparison
print(f"\nOriginal training set shape: {X_train.shape}")
print(f"Balanced training set shape: {X_train_balanced.shape}")

# %%
# K-Fold Cross-Validation Setup
# We'll use 10-fold cross-validation to evaluate our models, consistent with the R notebook approach
n_folds = 10
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=100)

# Function to evaluate model with cross-validation
def evaluate_model_cv(model, X, y, cv):
    # Calculate cross-validated scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Print mean and standard deviation of cross-validation scores
    print(f"Cross-Validation Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
    
    return {
        'cv_accuracy_mean': cv_accuracy.mean(),
        'cv_accuracy_std': cv_accuracy.std(),
        'cv_accuracy_scores': cv_accuracy
    }

# %%
# Model Training and Evaluation
# We'll train four models similar to the R notebook:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. Naive Bayes

# %%
# 1. Logistic Regression with 10-fold Cross-Validation
print("\n" + "="*80)
print("Logistic Regression Model")
print("="*80)

# Train Logistic Regression model
lr_model = LogisticRegression(random_state=100, max_iter=1000, solver='liblinear')

# Cross-validation on balanced training data
print("Logistic Regression - Cross-Validation Results:")
lr_cv_results = evaluate_model_cv(lr_model, X_train_balanced, y_train_balanced, cv)

# Train the model on the entire balanced training set
lr_model.fit(X_train_balanced, y_train_balanced)

# Predict on test set
lr_y_pred = lr_model.predict(X_test)

# Evaluate
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f"\nTest set accuracy: {lr_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, lr_y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
lr_cm = confusion_matrix(y_test, lr_y_pred)
print(lr_cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Early Readmission', 'Early Readmission'],
            yticklabels=['No Early Readmission', 'Early Readmission'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression - Confusion Matrix')
plt.show()

# Calculate ROC curve and AUC for Logistic Regression
lr_probs = lr_model.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
lr_auc = auc(lr_fpr, lr_tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression - ROC Curve')
plt.legend(loc='lower right')
plt.show()

# %%
# 2. Decision Tree with 10-fold Cross-Validation
print("\n" + "="*80)
print("Decision Tree Model")
print("="*80)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=100, min_samples_split=10)

# Cross-validation on balanced training data
print("Decision Tree - Cross-Validation Results:")
dt_cv_results = evaluate_model_cv(dt_model, X_train_balanced, y_train_balanced, cv)

# Train the model on the entire balanced training set
dt_model.fit(X_train_balanced, y_train_balanced)

# Predict on test set
dt_y_pred = dt_model.predict(X_test)

# Evaluate
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print(f"\nTest set accuracy: {dt_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, dt_y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
dt_cm = confusion_matrix(y_test, dt_y_pred)
print(dt_cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Early Readmission', 'Early Readmission'],
            yticklabels=['No Early Readmission', 'Early Readmission'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree - Confusion Matrix')
plt.show()

# Calculate ROC curve and AUC for Decision Tree
dt_probs = dt_model.predict_proba(X_test)[:, 1]
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
dt_auc = auc(dt_fpr, dt_tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree - ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Visualize the Decision Tree (simplified version for readability)
from sklearn import tree
plt.figure(figsize=(20, 10))
tree.plot_tree(dt_model, max_depth=3, feature_names=X_train.columns, 
               class_names=['No Early Readmission', 'Early Readmission'], filled=True)
plt.title('Decision Tree (Limited to Depth 3 for Visibility)')
plt.show()

# %%
# 3. Random Forest with 10-fold Cross-Validation
print("\n" + "="*80)
print("Random Forest Model")
print("="*80)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', 
                                 min_samples_split=10, random_state=100)

# Cross-validation on balanced training data
print("Random Forest - Cross-Validation Results:")
rf_cv_results = evaluate_model_cv(rf_model, X_train_balanced, y_train_balanced, cv)

# Train the model on the entire balanced training set
rf_model.fit(X_train_balanced, y_train_balanced)

# Predict on test set
rf_y_pred = rf_model.predict(X_test)

# Evaluate
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"\nTest set accuracy: {rf_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, rf_y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
rf_cm = confusion_matrix(y_test, rf_y_pred)
print(rf_cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Early Readmission', 'Early Readmission'],
            yticklabels=['No Early Readmission', 'Early Readmission'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest - Confusion Matrix')
plt.show()

# Feature importance
plt.figure(figsize=(12, 6))
features = X_train.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Calculate ROC curve and AUC for Random Forest
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest - ROC Curve')
plt.legend(loc='lower right')
plt.show()

# %%
# 4. Naive Bayes with 10-fold Cross-Validation
print("\n" + "="*80)
print("Naive Bayes Model")
print("="*80)

# Naive Bayes Model
nb_model = GaussianNB()

# Cross-validation on balanced training data
print("Naive Bayes - Cross-Validation Results:")
nb_cv_results = evaluate_model_cv(nb_model, X_train_balanced, y_train_balanced, cv)

# Train the model on the entire balanced training set
nb_model.fit(X_train_balanced, y_train_balanced)

# Predict on test set
nb_y_pred = nb_model.predict(X_test)

# Evaluate
nb_accuracy = accuracy_score(y_test, nb_y_pred)
print(f"\nTest set accuracy: {nb_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, nb_y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
nb_cm = confusion_matrix(y_test, nb_y_pred)
print(nb_cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Early Readmission', 'Early Readmission'],
            yticklabels=['No Early Readmission', 'Early Readmission'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes - Confusion Matrix')
plt.show()

# Calculate ROC curve and AUC for Naive Bayes
nb_probs = nb_model.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
nb_auc = auc(nb_fpr, nb_tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(nb_fpr, nb_tpr, label=f'Naive Bayes (AUC = {nb_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes - ROC Curve')
plt.legend(loc='lower right')
plt.show()

# %%
# Model Comparison
# ---------------
print("\n" + "="*80)
print("Model Comparison")
print("="*80)

# Collect all results
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes']
cv_accuracy = [lr_cv_results['cv_accuracy_mean'], dt_cv_results['cv_accuracy_mean'], 
               rf_cv_results['cv_accuracy_mean'], nb_cv_results['cv_accuracy_mean']]
test_accuracy = [lr_accuracy, dt_accuracy, rf_accuracy, nb_accuracy]

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': models,
    'CV Accuracy': cv_accuracy,
    'Test Accuracy': test_accuracy
})

print("Model Performance Comparison:")
print(comparison_df.sort_values('Test Accuracy', ascending=False))

# Visualize model comparison
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(models))

plt.bar(index, cv_accuracy, bar_width, label='Cross-Validation Accuracy', color='skyblue')
plt.bar(index + bar_width, test_accuracy, bar_width, label='Test Accuracy', color='lightcoral')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(index + bar_width/2, models, rotation=30, ha='right')
plt.legend()
plt.tight_layout()
plt.ylim(0, 1.0)
plt.show()

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))

# Logistic Regression
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', linestyle='-')

# Decision Tree
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.3f})', linestyle='--')

# Random Forest
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', linestyle='-.')

# Naive Bayes
plt.plot(nb_fpr, nb_tpr, label=f'Naive Bayes (AUC = {nb_auc:.3f})', linestyle=':')

# Reference line
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# %%
# Conclusions
# ----------
print("\n" + "="*80)
print("Conclusions")
print("="*80)

# Determine the best model
best_model_idx = test_accuracy.index(max(test_accuracy))
best_model = models[best_model_idx]

print(f"1. {best_model} performs the best with an accuracy of {max(test_accuracy)*100:.2f}%.")
print("2. The most important features for predicting hospital readmission are:")

# Get top 5 features from Random Forest
rf_importances = rf_model.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]
rf_features = X_train.columns

for i in range(5):
    feature_idx = rf_indices[i]
    print(f"   - {rf_features[feature_idx]}: {rf_importances[feature_idx]:.4f}")

print("3. The models provide reasonable predictive performance for identifying patients at risk of 30-day readmission.")
print("4. The approach properly handles data imbalance using SMOTE, feature selection, and cross-validation.")
print("5. This Python implementation demonstrates the same methodology as the original R notebook but with scikit-learn and imbalanced-learn libraries.")
print("="*80)

print("\nAnalysis complete!") 