import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=== Model Performance Analysis with Corrected SMOTE Implementation ===")
print("\nThis analysis compares model performance when SMOTE is correctly applied")
print("only to the training data after the train-test split.\n")

# Load your dataset (assuming it's the same as in the notebook)
# You may need to adjust this path if needed
try:
    df = pd.read_csv('diabetic_readmission_data.csv')
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print("\nPreprocessing data...")

# First, let's view the data types
print("Data types:\n", df.dtypes)

# Handle missing values
print("\nHandling missing values...")
df = df.replace('?', np.nan)  # Replace '?' with NaN
df = df.fillna(0)  # Replace NaN with 0

# Convert all columns to string then to numeric where possible
print("\nConverting mixed-type columns...")
for col in df.columns:
    # Convert everything to string first
    df[col] = df[col].astype(str)
    
    # Try to convert to numeric if possible
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass  # Keep as string if conversion fails

# Now identify truly categorical columns (after conversions)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns after conversion: {categorical_cols}")

# Use label encoding for categorical variables
print("\nEncoding categorical variables...")
for col in categorical_cols:
    if col != 'readmitted':  # Don't encode target yet
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Encoded {col}")

# Prepare features and target
X = df.drop('readmitted', axis=1, errors='ignore')
y = df['readmitted']

# Encode the target variable if needed
if pd.api.types.is_object_dtype(y):
    print("\nEncoding target variable...")
    le = LabelEncoder()
    y = le.fit_transform(y)
    target_classes = le.classes_
    print(f"Target classes: {target_classes}")
    print(f"Encoded as: {np.unique(y)}")

print(f"\nTarget variable distribution: {Counter(y)}")

# Perform train-test split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print(f"\nOriginal training set distribution: {Counter(y_train)}")
print(f"Original test set distribution: {Counter(y_test)}")

# Apply SMOTE only to the training data
print("\nApplying SMOTE to the training data only...")
smt = SMOTE(random_state=20)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
print(f"Resampled training set distribution: {Counter(y_train_resampled)}")

# Train models
print("\n=== Training models with correct SMOTE implementation ===")

# Logistic Regression
print("\nTraining Logistic Regression...")
logit = LogisticRegression(random_state=0, max_iter=1000)
logit.fit(X_train_resampled, y_train_resampled)
logit_pred = logit.predict(X_test)

# Decision Tree
print("Training Decision Tree...")
dtree = DecisionTreeClassifier(criterion="entropy", min_samples_split=10, random_state=0)
dtree.fit(X_train_resampled, y_train_resampled)
dtree_pred = dtree.predict(X_test)

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, criterion="gini", min_samples_split=10, random_state=0)
rf.fit(X_train_resampled, y_train_resampled)
rf_pred = rf.predict(X_test)

# Evaluate models
print("\n=== Model Evaluation ===")

# Helper function for model evaluation
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# Evaluate each model
results = {}
results['Logistic Regression'] = evaluate_model(y_test, logit_pred, 'Logistic Regression')
results['Decision Tree'] = evaluate_model(y_test, dtree_pred, 'Decision Tree')
results['Random Forest'] = evaluate_model(y_test, rf_pred, 'Random Forest')

# Compare model performance
print("\n=== Model Comparison ===")
comparison = pd.DataFrame(results).T
print(comparison)

# Visualize results
plt.figure(figsize=(12, 6))
comparison.plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison with Corrected SMOTE')
plt.ylabel('Score')
plt.xlabel('Model')
plt.grid(True, alpha=0.3)
plt.savefig('model_performance_comparison.png')
print("\nPerformance chart saved as 'model_performance_comparison.png'")

print("\n=== Analysis Complete ===")
print("The models were trained on properly resampled training data and evaluated on")
print("the original distribution test data, which better reflects real-world performance.") 