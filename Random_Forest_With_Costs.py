#!/usr/bin/env python
# coding: utf-8

"""
Cost-Optimized Random Forest Model for Hospital Readmission Prediction
======================================================================
This script implements a Random Forest model for predicting diabetic patient readmissions
with an emphasis on cost optimization.

Key features:
- Classification categories: No readmission, readmission <30 days, readmission >30 days
- Cost structure:
  * Readmission <30 days: $13,000
  * Readmission >30 days: $11,000
  * Prevention intervention cost: $5,000 (assumed to be 100% effective)
"""

# Import Libraries
# ----------------

# Data manipulation
import pandas as pd
import numpy as np
import re
import time

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Feature selection
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier

# Handling imbalanced data
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek

# Model building and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, make_scorer, f1_score, recall_score

# Custom modules
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# Visualization settings
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# For reproducibility
np.random.seed(100)

# Cost Parameters
# ---------------
COST_READMIT_LESS_30 = 13000  # Cost of readmission within 30 days
COST_READMIT_MORE_30 = 11000  # Cost of readmission after 30 days
COST_PREVENTION = 5000        # Cost of preventive intervention

# %%
# Load the dataset
def load_and_prepare_data(file_path='diabetic_readmission_data.csv'):
    """
    Load and prepare the diabetic readmission dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset dimensions: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        print("Original dataset can be found at: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008")
        return None
    
    # Display summary statistics
    print("Summary statistics:")
    print(df.describe(include='all').T)
    
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
    
    return df

# %%
# Dataset Preparation
def preprocess_data(df):
    """
    Preprocess the data for model training, following the approach in Basic_Solution_Python.py
    """
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
    
    # Remove Unnecessary Columns
    columns_to_drop = ['weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton']
    df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"Shape after dropping unnecessary columns: {df_processed.shape}")
    
    # Remove Encounters with Death Outcomes
    death_discharge_ids = [11, 13, 14, 19, 20, 21]
    df_processed = df_processed[~df_processed['discharge_disposition_id'].isin(death_discharge_ids)]
    
    print(f"Shape after removing death-related encounters: {df_processed.shape}")
    
    # Categorize diagnosis codes
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df_processed[f'{col}_category'] = df_processed[col].apply(categorize_diagnosis)
    
    # Handle Age Variable
    df_processed['age_midpoint'] = df_processed['age'].apply(age_to_midpoint)
    
    # Create a copy for encoding
    df_encoded = df_processed.copy()
    
    # Define categorical columns for encoding
    categorical_columns = [
        'race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
        'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
        'diag_1_category', 'diag_2_category', 'diag_3_category'
    ]
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Handle NaN values
            df_encoded[col] = df_encoded[col].fillna('Missing')
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    
    # Create three-class target variable (0: No readmission, 1: >30 days, 2: <30 days)
    readmission_mapping = {'NO': 0, '>30': 1, '<30': 2}
    df_encoded['readmitted_encoded'] = df_encoded['readmitted'].map(readmission_mapping)
    
    # Ensure all features are numeric
    for col in df_encoded.columns:
        # Skip the target variable and columns we'll exclude anyway
        if col in ['readmitted', 'readmitted_encoded', 'encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3']:
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
    
    return df_encoded

def categorize_diagnosis(code):
    """
    Categorize ICD-9 diagnosis codes into meaningful disease categories
    """
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

def age_to_midpoint(age_bracket):
    """
    Convert age brackets to numeric midpoints for easier analysis
    """
    if pd.isna(age_bracket):
        return np.nan
    
    # Extract numbers from the bracket
    numbers = re.findall(r'\d+', age_bracket)
    if len(numbers) == 2:
        return (int(numbers[0]) + int(numbers[1])) / 2
    else:
        return np.nan

# %%
# Feature Selection
def select_features(df_encoded):
    """
    Select important features for the model using the same approach as Basic_Solution_Python.py
    """
    # Define numeric features that are already in the right format
    numeric_features = [
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'age_midpoint'
    ]
    
    # Define categorical features
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
    y = df_encoded['readmitted_encoded']
    
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
    
    # Random Forest Feature Importance
    print("\nCalculating Random Forest feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
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
    
    # Recursive Feature Elimination (RFE) - Select top 20 features
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
    print(f"Class percentages: \n{y.value_counts(normalize=True) * 100}")
    
    return X_selected, y, selected_features

# %%
# Cost Calculation Functions
def visualize_cost_matrix():
    """
    Visualize the cost matrix for different prediction scenarios
    """
    # Create cost matrix
    cost_matrix = np.array([
        [0, COST_READMIT_MORE_30, COST_READMIT_LESS_30],  # No prevention
        [COST_PREVENTION, COST_PREVENTION, COST_PREVENTION]  # With prevention
    ])
    
    # Calculate savings
    savings_matrix = np.array([
        [0, 0, 0],  # No prevention -> no savings
        [0, COST_READMIT_MORE_30 - COST_PREVENTION, COST_READMIT_LESS_30 - COST_PREVENTION]  # Prevention savings
    ])
    
    # Create plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot cost matrix
    sns.heatmap(cost_matrix, annot=True, fmt='.0f', cmap='Blues', ax=axs[0],
                xticklabels=['No Readmission', '>30 Days', '<30 Days'],
                yticklabels=['No Prevention', 'Prevention'])
    axs[0].set_title('Cost Matrix ($)')
    
    # Plot savings matrix
    sns.heatmap(savings_matrix, annot=True, fmt='.0f', cmap='RdYlGn', ax=axs[1],
                xticklabels=['No Readmission', '>30 Days', '<30 Days'],
                yticklabels=['No Prevention', 'Prevention'])
    axs[1].set_title('Savings Matrix ($)')
    
    plt.tight_layout()
    plt.show()
    
    # Display optimal strategy based on cost matrix
    print("\nOptimal Decision Strategy:")
    print("- For patients predicted to have no readmission: No prevention")
    print(f"- For patients predicted to have >30 days readmission: Prevention (saves ${COST_READMIT_MORE_30-COST_PREVENTION:,.2f} per patient)")
    print(f"- For patients predicted to have <30 days readmission: Prevention (saves ${COST_READMIT_LESS_30-COST_PREVENTION:,.2f} per patient)")

def calculate_costs(y_true, y_pred):
    """
    Calculate costs based on predictions and the cost matrix
    Assumes 100% effective prevention
    
    Parameters:
    -----------
    y_true : array-like
        True readmission values (0: no readmission, 1: >30 days, 2: <30 days)
    y_pred : array-like
        Predicted readmission classes (0: no readmission, 1: >30 days, 2: <30 days)
        
    Returns:
    --------
    dict
        Dictionary containing cost metrics
    """
    # Convert pandas Series to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
        
    n_samples = len(y_true)
    
    # Initialize counters
    no_prevention_cost = 0
    with_prevention_cost = 0
    true_positives = 0
    false_positives = 0
    
    # Detailed tracking by readmission type
    less30_caught = 0  # <30 days readmissions correctly predicted as any readmission
    less30_missed = 0  # <30 days readmissions incorrectly predicted as no readmission
    less30_total = np.sum(y_true == 2)
    
    more30_caught = 0  # >30 days readmissions correctly predicted as any readmission
    more30_missed = 0  # >30 days readmissions incorrectly predicted as no readmission
    more30_total = np.sum(y_true == 1)
    
    # Track specific prediction types for confusion analysis
    predictions = {
        'true_no': 0,       # Correctly predicted no readmission
        'true_more30': 0,   # Correctly predicted >30 days
        'true_less30': 0,   # Correctly predicted <30 days
        'false_no_more30': 0,  # Predicted no readmission but was >30 days
        'false_no_less30': 0,  # Predicted no readmission but was <30 days
        'false_more30_no': 0,  # Predicted >30 days but was no readmission
        'false_more30_less30': 0,  # Predicted >30 days but was <30 days
        'false_less30_no': 0,  # Predicted <30 days but was no readmission
        'false_less30_more30': 0,  # Predicted <30 days but was >30 days
    }
    
    for i in range(n_samples):
        true_label = y_true[i]
        pred_label = y_pred[i]
        
        # Calculate costs without prevention
        if true_label == 1:  # >30 days readmission
            no_prevention_cost += COST_READMIT_MORE_30
        elif true_label == 2:  # <30 days readmission
            no_prevention_cost += COST_READMIT_LESS_30
        
        # Update confusion tracking
        if true_label == 0 and pred_label == 0:
            predictions['true_no'] += 1
        elif true_label == 1 and pred_label == 1:
            predictions['true_more30'] += 1
        elif true_label == 2 and pred_label == 2:
            predictions['true_less30'] += 1
        elif true_label == 1 and pred_label == 0:
            predictions['false_no_more30'] += 1
        elif true_label == 2 and pred_label == 0:
            predictions['false_no_less30'] += 1
        elif true_label == 0 and pred_label == 1:
            predictions['false_more30_no'] += 1
        elif true_label == 2 and pred_label == 1:
            predictions['false_more30_less30'] += 1
        elif true_label == 0 and pred_label == 2:
            predictions['false_less30_no'] += 1
        elif true_label == 1 and pred_label == 2:
            predictions['false_less30_more30'] += 1
            
        # Calculate costs with prevention based on predictions
        if pred_label > 0:  # Model predicts readmission (either type)
            with_prevention_cost += COST_PREVENTION
            
            if true_label > 0:  # True readmission
                true_positives += 1
                
                # Track by readmission type
                if true_label == 2:  # <30 days
                    less30_caught += 1
                else:  # >30 days
                    more30_caught += 1
            else:  # False positive (would not have been readmitted)
                false_positives += 1
                # Unnecessary prevention cost
        else:
            # No prevention applied
            if true_label == 1:  # >30 days readmission occurred
                with_prevention_cost += COST_READMIT_MORE_30
                more30_missed += 1
            elif true_label == 2:  # <30 days readmission occurred
                with_prevention_cost += COST_READMIT_LESS_30
                less30_missed += 1
    
    # Calculate metrics
    total_readmissions = np.sum(y_true > 0)
    predicted_readmissions = np.sum(y_pred > 0)
    savings = no_prevention_cost - with_prevention_cost
    
    # Calculate cost breakdown by readmission type
    savings_less30 = less30_caught * (COST_READMIT_LESS_30 - COST_PREVENTION)
    savings_more30 = more30_caught * (COST_READMIT_MORE_30 - COST_PREVENTION)
    loss_false_positives = false_positives * COST_PREVENTION
    
    results = {
        'total_cost_no_prevention': no_prevention_cost,
        'total_cost_with_prevention': with_prevention_cost,
        'savings': savings,
        'avg_cost_per_patient_no_prevention': no_prevention_cost / n_samples,
        'avg_cost_per_patient_with_prevention': with_prevention_cost / n_samples,
        'avg_savings_per_patient': savings / n_samples,
        'prevention_count': predicted_readmissions,
        'prevention_rate': predicted_readmissions / n_samples,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'total_readmissions': total_readmissions,
        'readmission_prevention_rate': true_positives / total_readmissions if total_readmissions > 0 else 0,
        'precision': true_positives / predicted_readmissions if predicted_readmissions > 0 else 0,
        'recall': true_positives / total_readmissions if total_readmissions > 0 else 0,
        
        # Detailed metrics by readmission type
        'less30_caught': less30_caught,
        'less30_missed': less30_missed,
        'less30_total': less30_total,
        'less30_recall': less30_caught / less30_total if less30_total > 0 else 0,
        'more30_caught': more30_caught,
        'more30_missed': more30_missed,
        'more30_total': more30_total,
        'more30_recall': more30_caught / more30_total if more30_total > 0 else 0,
        
        # Cost breakdown
        'savings_less30': savings_less30,
        'savings_more30': savings_more30,
        'loss_false_positives': loss_false_positives,
        'net_benefit': savings_less30 + savings_more30 - loss_false_positives,
        
        # Detailed prediction tracking
        'predictions': predictions
    }
    
    return results

# %%
# Model Training and Evaluation
def train_and_evaluate(X, y, test_size=0.2, n_folds=10, optimize=True):
    """
    Train and evaluate a Random Forest model with k-fold cross-validation
    and advanced optimization techniques
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas Series
        Target variable
    test_size : float
        Proportion of data to use for testing
    n_folds : int
        Number of folds for cross-validation
    optimize : bool
        Whether to perform hyperparameter optimization
        
    Returns:
    --------
    tuple
        (trained_model, cost_results)
    """
    print("\n" + "="*80)
    print("Random Forest Model Training and Evaluation (Optimized)")
    print("="*80)
    
    # Split the Dataset into Training and Test Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=100, stratify=y
    )
    
    # Check class distribution in training and test sets
    print("\nTraining set target distribution:")
    print(pd.Series(y_train).value_counts())
    print(f"Class percentages: \n{pd.Series(y_train).value_counts(normalize=True) * 100}")
    
    print("\nTest set target distribution:")
    print(pd.Series(y_test).value_counts())
    print(f"Class percentages: \n{pd.Series(y_test).value_counts(normalize=True) * 100}")
    
    if optimize:
        # 1. Optimize SMOTE parameters
        print("\nOptimizing SMOTE parameters...")
        best_smote_method, X_train_balanced, y_train_balanced = optimize_sampling(X_train, y_train)
        print(f"Best SMOTE method: {best_smote_method}")
    else:
        # Standard SMOTE
        print("\nApplying standard SMOTE to balance training data...")
        smote = SMOTE(random_state=100)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Check balanced class distribution
    print("\nBalanced training set target distribution:")
    print(pd.Series(y_train_balanced).value_counts())
    print(f"Class percentages: \n{pd.Series(y_train_balanced).value_counts(normalize=True) * 100}")
    
    if optimize:
        # 2. Hyperparameter optimization with GridSearchCV
        print("\nPerforming hyperparameter optimization with GridSearchCV...")
        best_params, rf_model = optimize_hyperparameters(X_train_balanced, y_train_balanced, n_folds)
        print(f"Best parameters: {best_params}")
    else:
        # Create basic Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            min_samples_split=10,
            random_state=100,
            n_jobs=-1
        )
    
    # K-Fold Cross-Validation
    print("\nPerforming cross-validation...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=100)
    
    # Perform cross-validation with optimized model
    cv_accuracy = cross_val_score(rf_model, X_train_balanced, y_train_balanced, cv=cv, scoring='accuracy')
    cv_f1_macro = cross_val_score(rf_model, X_train_balanced, y_train_balanced, cv=cv, scoring='f1_macro')
    cv_recall_macro = cross_val_score(rf_model, X_train_balanced, y_train_balanced, cv=cv, scoring='recall_macro')
    
    # Print cross-validation results
    print(f"Cross-Validation Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
    print(f"Cross-Validation F1 Macro: {cv_f1_macro.mean():.4f} (+/- {cv_f1_macro.std():.4f})")
    print(f"Cross-Validation Recall Macro: {cv_recall_macro.mean():.4f} (+/- {cv_recall_macro.std():.4f})")
    
    # Train the model on the entire balanced training set
    print("\nTraining final model on balanced training data...")
    start_time = time.time()
    rf_model.fit(X_train_balanced, y_train_balanced)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Train and evaluate model on test data
    print("\nEvaluating model on test data...")
    y_pred = rf_model.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Readmission', '>30 Days', '<30 Days']))
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Readmission', '>30 Days', '<30 Days'],
                yticklabels=['No Readmission', '>30 Days', '<30 Days'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest - Confusion Matrix')
    plt.show()
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Features by Importance')
    plt.tight_layout()
    plt.show()
    
    # Cost analysis
    print("\nCost Analysis with 100% Effective Prevention:")
    cost_results = calculate_costs(y_test, y_pred)
    
    # Print cost analysis results
    print(f"Total cost without prevention: ${cost_results['total_cost_no_prevention']:,.2f}")
    print(f"Total cost with prevention: ${cost_results['total_cost_with_prevention']:,.2f}")
    print(f"Total savings: ${cost_results['savings']:,.2f}")
    print(f"Average cost per patient (no prevention): ${cost_results['avg_cost_per_patient_no_prevention']:,.2f}")
    print(f"Average cost per patient (with prevention): ${cost_results['avg_cost_per_patient_with_prevention']:,.2f}")
    print(f"Average savings per patient: ${cost_results['avg_savings_per_patient']:,.2f}")
    
    # Prevention effectiveness
    print("\nPrevention Effectiveness:")
    print(f"Patients flagged for prevention: {cost_results['prevention_count']} ({cost_results['prevention_rate']:.2%})")
    print(f"True positives (correct prevention): {cost_results['true_positives']} patients")
    print(f"False positives (unnecessary prevention): {cost_results['false_positives']} patients")
    print(f"Readmission prevention rate: {cost_results['readmission_prevention_rate']:.2%}")
    print(f"Precision: {cost_results['precision']:.2%}")
    print(f"Recall: {cost_results['recall']:.2%}")
    
    # Prevention by readmission type
    print("\nPrevention by Readmission Type:")
    print(f"<30 days - Caught: {cost_results['less30_caught']} out of {cost_results['less30_total']} ({cost_results['less30_recall']:.2%})")
    print(f">30 days - Caught: {cost_results['more30_caught']} out of {cost_results['more30_total']} ({cost_results['more30_recall']:.2%})")
    
    # Financial breakdown
    print("\nFinancial Breakdown:")
    print(f"Savings from <30 days prevention: ${cost_results['savings_less30']:,.2f}")
    print(f"Savings from >30 days prevention: ${cost_results['savings_more30']:,.2f}")
    print(f"Cost of false positives: ${cost_results['loss_false_positives']:,.2f}")
    print(f"Net benefit: ${cost_results['net_benefit']:,.2f}")
    
    # Detailed prediction analysis
    print("\nDetailed Prediction Analysis:")
    print(f"Correct predictions - No readmission: {cost_results['predictions']['true_no']}")
    print(f"Correct predictions - >30 days: {cost_results['predictions']['true_more30']}")
    print(f"Correct predictions - <30 days: {cost_results['predictions']['true_less30']}")
    print(f"Missed >30 days readmissions: {cost_results['predictions']['false_no_more30']}")
    print(f"Missed <30 days readmissions: {cost_results['predictions']['false_no_less30']}")
    
    # Calculate ROI
    prevention_investment = cost_results['prevention_count'] * COST_PREVENTION
    roi = cost_results['savings'] / prevention_investment if prevention_investment > 0 else 0
    print(f"\nROI of prevention program: {roi:.2f}")
    
    if roi > 0:
        print("The prevention program is cost-effective!")
    else:
        print("The prevention program is not cost-effective with the current model and cost structure.")
    
    return rf_model, cost_results

# %%
# Optimization Helper Functions
def optimize_sampling(X_train, y_train):
    """
    Try different sampling techniques to find the best one
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series
        Training target variable
        
    Returns:
    --------
    tuple
        (best_method_name, resampled_X, resampled_y)
    """
    # For memory-limited systems, just use BorderlineSMOTE directly
    # as it was previously found to be the best method
    try:
        print("Using BorderlineSMOTE for class balancing (pre-optimized choice)...")
        method = BorderlineSMOTE(random_state=100, kind='borderline-2')
        X_resampled, y_resampled = method.fit_resample(X_train, y_train)
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        return "BorderlineSMOTE", X_resampled, y_resampled
    
    except Exception as e:
        # If even BorderlineSMOTE fails, fall back to regular SMOTE
        print(f"Error with BorderlineSMOTE: {str(e)}")
        print("Falling back to standard SMOTE...")
        try:
            method = SMOTE(random_state=100)
            X_resampled, y_resampled = method.fit_resample(X_train, y_train)
            print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
            return "SMOTE", X_resampled, y_resampled
        except Exception as e2:
            # If all else fails, return the original unbalanced data
            print(f"Error with SMOTE: {str(e2)}")
            print("WARNING: Using original imbalanced data due to memory limitations")
            return "Original", X_train, y_train

def optimize_hyperparameters(X_train, y_train, n_folds=5):
    """
    Optimize Random Forest hyperparameters using GridSearchCV
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series
        Training target variable
    n_folds : int
        Number of folds for cross-validation
        
    Returns:
    --------
    tuple
        (best_params, best_model)
    """
    # For extremely memory-limited environments, use this tiny grid
    tiny_param_grid = {
        'n_estimators': [200],
        'max_depth': [None],
        'min_samples_split': [5],
        'class_weight': [{0: 1, 1: 3, 2: 5}]  # Custom class weight: higher penalty for missing <30 days
    }
    
    # Define base classifier
    rf = RandomForestClassifier(random_state=100, n_jobs=1)
    
    # Define custom scoring that focuses specifically on readmission recall
    # We care much more about catching readmissions than precision
    scoring = {
        'recall_macro': 'recall_macro',
        'recall_readmit': make_scorer(lambda y_true, y_pred: recall_score(y_true, y_pred, labels=[1, 2], average='macro'))
    }
    
    # Create GridSearchCV - refit on recall_readmit to focus on catching readmissions
    print("Starting simple parameter tuning with minimal resources...")
    start_time = time.time()
    
    # Instead of full GridSearch, just use the recommended parameters directly
    # This avoids the memory-intensive GridSearchCV process
    best_params = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 5,
        'class_weight': {0: 1, 1: 3, 2: 5},
        'n_jobs': 1  # Use single job to avoid memory issues
    }
    
    best_model = RandomForestClassifier(**best_params, random_state=100)
    
    # Train the model directly with the best params
    best_model.fit(X_train, y_train)
    
    execution_time = time.time() - start_time
    
    print(f"Parameter tuning completed in {execution_time:.2f} seconds")
    print(f"Using parameters: {best_params}")
    
    # For optional evaluation of the chosen parameters
    # Using only a single small cross-validation to check recall
    fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
    recall_macro = cross_val_score(best_model, X_train, y_train, cv=fold, scoring='recall_macro', n_jobs=1)
    
    print(f"Quick validation recall_macro: {recall_macro.mean():.4f}")
    
    return best_params, best_model

# Custom metrics for cost optimization
def readmission_cost_score(y_true, y_pred):
    """
    Custom scoring function that estimates the cost savings
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        Cost savings score (higher is better)
    """
    # Convert pandas Series to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    cost_results = calculate_costs(y_true, y_pred)
    return cost_results['savings']  # Return savings as the score

# %%
# Main Function
def main():
    """
    Main function to run the analysis
    """
    print("Hospital Readmission Prediction with Cost Analysis (Optimized)")
    print("=" * 75)
    
    # Load and explore data
    print("\nLoading and exploring data...")
    df = load_and_prepare_data()
    
    if df is None:
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_encoded = preprocess_data(df)
    
    # Visualize the cost matrix
    print("\nVisualizing cost matrix and optimal strategy...")
    visualize_cost_matrix()
    
    # Select features
    print("\nSelecting important features...")
    X_selected, y, selected_features = select_features(df_encoded)
    
    # Ask user about system capabilities
    memory_limited = True  # Default to memory-limited mode
    try:
        user_input = input("\nIs your system memory-limited? (y/n, default=y): ").strip().lower()
        if user_input == 'n':
            memory_limited = False
            print("Running with full optimization...")
        else:
            print("Running with memory-efficient mode...")
    except:
        print("Assuming memory-limited environment...")
    
    # Train and evaluate model with optimizations based on memory constraints
    optimization_flag = not memory_limited
    print(f"\nTraining and evaluating Random Forest model (optimization={optimization_flag})...")
    model, cost_results = train_and_evaluate(X_selected, y, optimize=optimization_flag)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"Model: Random Forest" + (" (Optimized)" if optimization_flag else " (Basic)"))
    print(f"Number of features used: {len(selected_features)}")
    print(f"Total cost without prevention: ${cost_results['total_cost_no_prevention']:,.2f}")
    print(f"Total cost with prevention: ${cost_results['total_cost_with_prevention']:,.2f}")
    print(f"Total savings: ${cost_results['savings']:,.2f}")
    
    # Add readmission recalls
    print(f"\nReadmission recalls:")
    print(f"<30 days readmission recall: {cost_results['less30_recall']:.2%}")
    print(f">30 days readmission recall: {cost_results['more30_recall']:.2%}")
    print(f"Overall readmission recall: {cost_results['recall']:.2%}")
    
    # Add financial breakdown
    print(f"\nFinancial metrics:")
    print(f"Savings from <30 days prevention: ${cost_results['savings_less30']:,.2f}")
    print(f"Savings from >30 days prevention: ${cost_results['savings_more30']:,.2f}")
    print(f"Cost of false positives: ${cost_results['loss_false_positives']:,.2f}")
    print(f"Net benefit: ${cost_results['net_benefit']:,.2f}")
    print(f"ROI: {cost_results['savings'] / (cost_results['prevention_count'] * COST_PREVENTION) if cost_results['prevention_count'] > 0 else 0:.2f}")
    
    # Try to run comparison only if we have enough memory
    if not memory_limited:
        try:
            # Compare with non-optimized model (optional)
            print("\nFor comparison, training basic model without optimization...")
            basic_model, basic_cost_results = train_and_evaluate(X_selected, y, optimize=False)
            
            # Print comparison
            print("\n" + "="*80)
            print("OPTIMIZATION IMPROVEMENT")
            print("="*80)
            print(f"Basic model savings: ${basic_cost_results['savings']:,.2f}")
            print(f"Optimized model savings: ${cost_results['savings']:,.2f}")
            savings_improvement = cost_results['savings'] - basic_cost_results['savings']
            percent_improvement = (savings_improvement / basic_cost_results['savings']) * 100 if basic_cost_results['savings'] > 0 else float('inf')
            print(f"Improvement: ${savings_improvement:,.2f} ({percent_improvement:.2f}%)")
            
            # Compare readmission recall improvement
            print(f"\nBasic model <30 days recall: {basic_cost_results['less30_recall']:.2%}")
            print(f"Optimized model <30 days recall: {cost_results['less30_recall']:.2%}")
            less30_improvement = (cost_results['less30_recall'] - basic_cost_results['less30_recall']) * 100
            print(f"<30 days recall improvement: {less30_improvement:.2f} percentage points")
            
            print(f"\nBasic model >30 days recall: {basic_cost_results['more30_recall']:.2%}")
            print(f"Optimized model >30 days recall: {cost_results['more30_recall']:.2%}")
            more30_improvement = (cost_results['more30_recall'] - basic_cost_results['more30_recall']) * 100
            print(f">30 days recall improvement: {more30_improvement:.2f} percentage points")
        except Exception as e:
            print("\nCouldn't run comparison due to resource limitations.")
            print(f"Error: {str(e)}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 