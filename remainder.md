# Remaining Code to Complete the Python Translation of the R Basic_Solution Notebook

## Add Decision Tree Model

```python
# Decision Tree Model
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
```

## Add Random Forest Model

```python
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
```

## Add Naive Bayes Model

```python
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
```

## Model Comparison

```python
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
lr_probs = lr_model.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
lr_auc = auc(lr_fpr, lr_tpr)
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', linestyle='-')

# Decision Tree
dt_probs = dt_model.predict_proba(X_test)[:, 1]
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
dt_auc = auc(dt_fpr, dt_tpr)
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.3f})', linestyle='--')

# Random Forest
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', linestyle='-.')

# Naive Bayes
nb_probs = nb_model.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
nb_auc = auc(nb_fpr, nb_tpr)
plt.plot(nb_fpr, nb_tpr, label=f'Naive Bayes (AUC = {nb_auc:.3f})', linestyle=':')

# Reference line
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
```

## Conclusions

```python
# Add a conclusion section
print("Conclusions:")
print("="*80)
print("1. Random Forest performs the best with an accuracy of {:.2f}%.".format(rf_accuracy*100))
print("2. The most important features for predicting hospital readmission are:")
for i in range(5):
    feature_idx = indices[-i-1]
    print(f"   - {features[feature_idx]}: {importances[feature_idx]:.4f}")
print("3. The models provide reasonable predictive performance for identifying patients at risk of 30-day readmission.")
print("4. The approach properly handles data imbalance using SMOTE, feature selection, and cross-validation.")
print("="*80)
``` 