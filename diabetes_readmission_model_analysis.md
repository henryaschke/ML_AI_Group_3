# Diabetes Readmission Prediction Model Analysis

## Dataset Overview
- **Source**: 130 US hospitals over 10 years (1999-2008)
- **Size**: 99,343 hospital encounters (after preprocessing)
- **Target**: Patient readmission (binary classification)
  - **Class 0** (No readmission): 88.61% of cases
  - **Class 1** (Readmission): 11.39% of cases

## Baseline Performance
- **Majority Class Classifier**: 88.61% accuracy (always predicting no readmission)
- **Random Classifier**: 50.00% accuracy
- **Class-Weighted Random**: 79.57% accuracy (0.8861² + 0.1139² = 0.7957)

## Model Performance Comparison

| Model | Accuracy | Precision (No Readmit) | Recall (No Readmit) | F1 (No Readmit) | Precision (Readmit) | Recall (Readmit) | F1 (Readmit) |
|-------|----------|------------------------|---------------------|-----------------|---------------------|------------------|--------------|
| Random Forest | 79.8% | 0.89 | 0.88 | 0.88 | 0.16 | 0.17 | 0.16 |
| Decision Tree | 71.5% | 0.89 | 0.77 | 0.83 | 0.13 | 0.27 | 0.17 |
| Logistic Regression | 62.0% | 0.88 | 0.66 | 0.75 | 0.12 | 0.43 | 0.19 |
| Naive Bayes | 57.0% | 0.88 | 0.59 | 0.71 | 0.12 | 0.51 | 0.20 |

## Key Performance Indicators (KPIs)

### 1. Random Forest
- **Overall Accuracy**: 79.8%
- **Balanced Accuracy**: 52.5% ((0.88 + 0.17)/2)
- **Readmission Detection Rate**: 17%
- **ROC AUC**: 0.71
- **Cross-Validation Accuracy**: 77.2% (±1.5%)
- **Performance vs Baseline**: Matches class-weighted random baseline in accuracy, but with better discrimination

### 2. Decision Tree
- **Overall Accuracy**: 71.5% 
- **Balanced Accuracy**: 52.0% ((0.77 + 0.27)/2)
- **Readmission Detection Rate**: 27%
- **ROC AUC**: 0.64
- **Cross-Validation Accuracy**: 70.3% (±1.8%)
- **Performance vs Baseline**: Lower than majority class baseline in accuracy, but catches more readmissions

### 3. Logistic Regression
- **Overall Accuracy**: 62.0%
- **Balanced Accuracy**: 54.5% ((0.66 + 0.43)/2)
- **Readmission Detection Rate**: 43%
- **ROC AUC**: 0.61
- **Cross-Validation Accuracy**: 64.1% (±2.1%)
- **Performance vs Baseline**: Much lower accuracy than baseline, but much better at identifying readmissions

### 4. Naive Bayes
- **Overall Accuracy**: 57.0%
- **Balanced Accuracy**: 55.0% ((0.59 + 0.51)/2)
- **Readmission Detection Rate**: 51%
- **ROC AUC**: 0.58
- **Cross-Validation Accuracy**: 58.7% (±2.3%)
- **Performance vs Baseline**: Lowest accuracy but highest readmission detection rate

## Most Important Features
1. Number of lab procedures
2. Number of medications
3. Time in hospital
4. Age
5. Diagnosis categories
6. Number of inpatient visits

## Model Selection Guidance

### When to Use Each Model:

**Random Forest**
- Best for: Overall prediction accuracy
- Ideal when: False alarms about readmission would be costly
- Limitation: Misses most readmission cases

**Decision Tree**
- Best for: Interpretable predictions with moderate recall
- Ideal when: Understanding decision factors is important
- Limitation: Still misses many readmission cases

**Logistic Regression**
- Best for: Better identification of readmission risk with acceptable false positives
- Ideal when: Model interpretability and readmission detection are both important
- Limitation: Lower overall accuracy

**Naive Bayes**
- Best for: Maximum readmission detection
- Ideal when: Missing a readmission case is very costly
- Limitation: Lowest accuracy, many false positives

## Conclusion

The analysis reveals the inherent trade-off between overall accuracy and readmission detection in this highly imbalanced dataset. While the Random Forest model provides the best overall accuracy at 79.8%, it only identifies 17% of readmission cases. On the other end of the spectrum, the Naive Bayes model has the lowest accuracy at 57.0% but detects 51% of readmission cases.

For a healthcare organization, model selection should be based on the relative costs of missing a readmission case versus incorrectly flagging a non-readmission case. If the priority is to catch as many potential readmissions as possible for intervention, the Naive Bayes or Logistic Regression models would be recommended despite their lower overall accuracy.

For future iterations, we should consider:
1. Multi-class classification to distinguish between readmissions within 30 days and after 30 days
2. Cost-sensitive learning to place higher importance on detecting readmission cases
3. Ensemble methods combining the strengths of multiple models
4. Additional feature engineering focusing on clinical risk factors 