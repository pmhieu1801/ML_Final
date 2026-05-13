# Final Report Sections (Ready to Use)

## Experimental Results and Model Comparison
We evaluated four classical machine learning models on the development set: Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost. The comparison used a consistent protocol and the same feature engineering pipeline across all models.

### Quantitative Results on Development Set

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9200 | 1.0000 | 0.8000 | 0.8889 | 0.9983 | 0.9977 |
| SVM | 0.9200 | 1.0000 | 0.8000 | 0.8889 | 0.9958 | 0.9942 |
| Random Forest | 0.9000 | 1.0000 | 0.7500 | 0.8571 | 0.9771 | 0.9699 |
| XGBoost | 0.8900 | 1.0000 | 0.7250 | 0.8406 | 0.9925 | 0.9905 |

The top two models by F1-score are Logistic Regression and SVM. Logistic Regression achieved the highest ROC-AUC and PR-AUC, indicating stronger ranking quality and better performance for the positive class under precision-recall trade-offs.

### Confusion Matrix Discussion
All models produced zero false positives on the development set, but differed in false negatives:
- Logistic Regression: 8 false negatives
- SVM: 8 false negatives
- Random Forest: 10 false negatives
- XGBoost: 11 false negatives

This result suggests that recall is the main differentiator in this dataset, and Logistic Regression/SVM are preferable when reducing missed scholarship candidates is important.

## Error Analysis
To avoid superficial sample-level analysis, we conducted group-based error analysis on the selected model.

### Error Rate by Income Group
- Low income: 0.1290
- Medium income: 0.0333
- High income: 0.0769

The model makes more mistakes in the low-income segment than in the medium-income segment, indicating possible overlap or noisier boundaries in this group.

### Error Rate by GPA Band
- Low GPA: 0.0909
- Borderline GPA: 0.1304
- High GPA: 0.0303

The highest error appears in the borderline GPA region, consistent with classification difficulty near decision boundaries.

### Error Rate by Exam Score Band
- Low exam band: 0.0968
- Mid exam band: 0.0385
- High exam band: 0.0930

Errors are lower in the mid-range and increase at both extremes, suggesting non-linear interactions between exam score and other factors.

## Model Interpretation
We used model-specific interpretation methods:
- Logistic Regression: coefficient analysis
- Random Forest and XGBoost: feature importance

Across all interpretation methods, the same core features dominate:
- GPA
- GPA x Attendance interaction (gpa_attendance)
- Study hours x Exam score interaction (study_exam)
- Exam score

This consistency supports the validity of the feature engineering strategy and indicates that academic performance signals are primary predictors in this task.

## System Improvement Over Midterm
Compared with the midterm baseline, the final system introduces several meaningful improvements:
1. Expanded model set from two models to four models (LR, RF, SVM, XGBoost).
2. Added advanced ranking metrics (ROC-AUC and PR-AUC).
3. Added structured, group-based error analysis instead of only individual-case inspection.
4. Added model interpretation for both linear and tree/boosting families.
5. Added repeated-split stability validation.

These improvements are technically meaningful and directly supported by quantitative evidence.

## Stability and Reproducibility
We ran repeated stratified holdout evaluation (5 splits) to verify robustness.

| Model | Mean F1 | Std F1 | Mean Accuracy | Std Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9709 | 0.0391 | 0.9760 | 0.0329 |
| SVM | 0.9386 | 0.0233 | 0.9520 | 0.0179 |
| Random Forest | 0.9136 | 0.0469 | 0.9320 | 0.0390 |
| XGBoost | 0.9114 | 0.0636 | 0.9320 | 0.0482 |

Logistic Regression remains strongest in average F1 and overall accuracy, while SVM is competitive but lower in mean F1.

## Leakage Prevention Checklist
- Scaler fit on training data only: PASS
- Hyperparameter tuning without using test data: PASS
- Feature engineering without target leakage: PASS
- Same development protocol across models: PASS

## Final Recommendation
We recommend Logistic Regression as the final deployment candidate for this project.

Justification:
1. Best overall development-set performance (highest ROC-AUC and PR-AUC; top-tied F1).
2. Strong repeated-split stability (highest mean F1 and mean accuracy).
3. High interpretability through coefficients, suitable for academic reporting and policy-oriented explanation.
4. Lower operational complexity than more complex ensemble/boosting alternatives, with no performance advantage observed from those alternatives in this dataset.

## Limitations and Future Work
- The dataset size is relatively small, so future work should validate conclusions on a larger hold-out dataset.
- Threshold tuning and cost-sensitive optimization may further reduce false negatives.
- Additional fairness checks across socioeconomic groups should be added before real deployment.
