# Hospital Readmission Model Performance Analysis

## Confusion Matrix Results

From the model outputs, the following confusion matrices were extracted:

### Logistic Regression
```
TN = 11348  FP = 6258
FN = 1289   TP = 974
```

### Decision Tree
```
TN = 13596  FP = 4010
FN = 1648   TP = 615
```

### Random Forest
```
TN = 15456  FP = 2150
FN = 1867   TP = 396
```

### Naive Bayes
```
TN = 10174  FP = 7432
FN = 1103   TP = 1160
```

## KPI Calculations for Each Model

### Logistic Regression
- Accuracy: (11348 + 974) / 19869 = 0.6202 (62.02%)
- Precision: 974 / (974 + 6258) = 0.1347 (13.47%)
- Recall/Sensitivity: 974 / (974 + 1289) = 0.4303 (43.03%)
- Specificity: 11348 / (11348 + 6258) = 0.6446 (64.46%)
- F1 Score: 2 * (0.1347 * 0.4303) / (0.1347 + 0.4303) = 0.2049 (20.49%)

### Decision Tree
- Accuracy: (13596 + 615) / 19869 = 0.7152 (71.52%)
- Precision: 615 / (615 + 4010) = 0.1330 (13.30%)
- Recall/Sensitivity: 615 / (615 + 1648) = 0.2718 (27.18%)
- Specificity: 13596 / (13596 + 4010) = 0.7722 (77.22%)
- F1 Score: 2 * (0.1330 * 0.2718) / (0.1330 + 0.2718) = 0.1786 (17.86%)

### Random Forest
- Accuracy: (15456 + 396) / 19869 = 0.7978 (79.78%)
- Precision: 396 / (396 + 2150) = 0.1555 (15.55%)
- Recall/Sensitivity: 396 / (396 + 1867) = 0.1750 (17.50%)
- Specificity: 15456 / (15456 + 2150) = 0.8778 (87.78%)
- F1 Score: 2 * (0.1555 * 0.1750) / (0.1555 + 0.1750) = 0.1648 (16.48%)

### Naive Bayes
- Accuracy: (10174 + 1160) / 19869 = 0.5704 (57.04%)
- Precision: 1160 / (1160 + 7432) = 0.1350 (13.50%)
- Recall/Sensitivity: 1160 / (1160 + 1103) = 0.5126 (51.26%)
- Specificity: 10174 / (10174 + 7432) = 0.5779 (57.79%)
- F1 Score: 2 * (0.1350 * 0.5126) / (0.1350 + 0.5126) = 0.2138 (21.38%)

## Baseline Performance (Simple Guessing)

From the test set distribution:
- Class 0 (no early readmission): 17606 samples (88.61%)
- Class 1 (early readmission): 2263 samples (11.39%)

### Baseline Model 1: Always predict majority class (no early readmission)
- Accuracy: 17606 / 19869 = 0.8861 (88.61%)
- Precision: 0 / 0 = undefined (0%)
- Recall: 0 / 2263 = 0 (0%)
- Specificity: 17606 / 17606 = 1.0 (100%)
- F1 Score: 0 (since recall is 0)

### Baseline Model 2: Random guessing based on class distribution
- Accuracy: 0.8861² + 0.1139² = 0.7987 (79.87%)
- Precision: 0.1139 (11.39%)
- Recall: 0.1139 (11.39%)
- Specificity: 0.8861 (88.61%)
- F1 Score: 2 * (0.1139 * 0.1139) / (0.1139 + 0.1139) = 0.1139 (11.39%)

## Comparison Summary

| Model | Accuracy | Precision | Recall | Specificity | F1 Score |
|-------|----------|-----------|--------|-------------|----------|
| Logistic Regression | 62.02% | 13.47% | 43.03% | 64.46% | 20.49% |
| Decision Tree | 71.52% | 13.30% | 27.18% | 77.22% | 17.86% |
| Random Forest | 79.78% | 15.55% | 17.50% | 87.78% | 16.48% |
| Naive Bayes | 57.04% | 13.50% | 51.26% | 57.79% | 21.38% |
| Baseline (Always 0) | 88.61% | 0% | 0% | 100% | 0% |
| Baseline (Random) | 79.87% | 11.39% | 11.39% | 88.61% | 11.39% |

## Analysis:

1. **Accuracy**: Random Forest has the highest accuracy (79.78%), which is competitive with the random guessing baseline (79.87%) but lower than always predicting the majority class (88.61%). This demonstrates why accuracy alone is misleading for imbalanced datasets.

2. **Precision**: All models have similar precision around 13-15%, slightly better than random guessing (11.39%).

3. **Recall**: Naive Bayes has the highest recall (51.26%), followed by Logistic Regression (43.03%), both significantly better than the baselines (0% and 11.39%).

4. **F1 Score**: Naive Bayes has the highest F1 score (21.38%), followed by Logistic Regression (20.49%), both better than the baselines.

5. **Specificity**: Random Forest has the highest specificity (87.78%), making it best at correctly identifying non-readmissions, but at the cost of lower recall.

## Conclusion

While Random Forest has the highest accuracy, this is partly due to the class imbalance in the dataset. For identifying patients at risk of early readmission (the minority class), Naive Bayes offers the best balance of precision and recall as reflected in its F1 score. This is particularly important in healthcare contexts where failing to identify at-risk patients (false negatives) can have serious consequences.

The analysis shows that all models perform better than random guessing, but no model achieves both high precision and recall. This highlights the challenge of predicting hospital readmissions and suggests that further feature engineering or more sophisticated modeling approaches might be needed to improve performance. 