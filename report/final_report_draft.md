# Scholarship Prediction System Development

**Course:** Introduction to Machine Learning  
**Project Type:** Final Project  
**Team Size:** 2 students  
**Submission:** report.pdf, notebook.ipynb, predictions.csv

## Abstract
This project develops a scholarship prediction system using a supervised machine learning workflow. Starting from the midterm baseline, we rebuilt the pipeline on the v2 dataset and expanded it into a more rigorous final solution. The workflow includes exploratory data analysis, feature engineering, preprocessing, model training, systematic comparison, error analysis, model interpretation, and a stability check for reproducibility. We trained four classical models, namely Logistic Regression, Support Vector Machine, Random Forest, and XGBoost. On the development set, Logistic Regression and SVM achieved the strongest classification performance, while Logistic Regression provided the best overall balance of predictive quality, stability, and interpretability. The final recommendation is Logistic Regression because it combines the highest ROC-AUC and PR-AUC with strong F1-score and the most transparent decision logic.

## 1. Introduction
The goal of this project is to predict whether a student receives a scholarship based on academic and socio-economic attributes. This task is a binary classification problem and is representative of a practical educational decision-support setting. The objective of the final project is not only to obtain good predictive performance, but also to justify modeling choices, compare several models in a principled way, and explain model behavior in a way that is suitable for technical reporting.

Compared with the midterm version, the final project extends the baseline system in several meaningful ways. We use the updated v2 dataset, perform more systematic exploratory analysis, add interaction-based feature engineering, train and compare four models, report additional metrics such as ROC-AUC and PR-AUC, analyze errors by group, interpret model outputs, and evaluate stability across multiple splits.

## 2. Dataset Description
The dataset contains student records with the following features:
- `gpa`
- `attendance_rate`
- `study_hours_per_week`
- `exam_score`
- `household_income`
- `part_time_job_hours`
- `label` as the target variable

The dataset is split into a training set and a development set. In the training set, there are 250 samples and 8 columns including the target. The target distribution is moderately imbalanced: 60% class 0 and 40% class 1. No missing values were found in the v2 dataset.

### Problem Scope
The task is to predict scholarship eligibility from student attributes. Because the label is binary, the problem is best framed as supervised classification. The most important requirement is not only accuracy, but also the ability to identify scholarship recipients without missing too many positive cases.

## 3. Exploratory Data Analysis
We conducted EDA to understand dataset structure, feature ranges, and the target distribution. The analysis showed that the dataset is small, clean, and fully numeric. This affects model choice because it reduces the need for complex encoding and allows classical tabular models to work well.

### 3.1 Dataset Structure and Descriptive Statistics
The data consists of one target variable and several continuous or discrete predictors. Summary statistics showed that:
- GPA values range from 2.00 to 3.99
- Attendance rates range from 0.60 to 1.00
- Exam scores range from 45 to 99
- Household income has a broad range and much larger magnitude than the other features

This scale difference supports the use of standardization for linear and distance-based models.

### 3.2 Target Distribution
The target distribution is 150 negative samples and 100 positive samples in the training set. The class imbalance is not extreme, but it is large enough to justify using F1-score, PR-AUC, and recall in addition to accuracy.

### 3.3 Feature-Level Observations
The EDA highlighted several useful patterns:
- Higher GPA and higher exam scores tend to be associated with scholarship approval.
- Attendance appears to be positively related to the target.
- Household income appears to interact with academic variables and may act as a strong contextual signal.
- Part-time job hours may negatively relate to scholarship probability because it can reflect limited time for study.

These findings motivated feature engineering and metric selection in the next stage.

### 3.4 Data Quality
No missing values were found, and the dataset appears internally consistent. Because the data is already numerical, we did not need categorical encoding. We still added frequency tables and grouped views for income and part-time-job features to support interpretation.

## 4. Data Preprocessing
Preprocessing was designed to avoid leakage and to match model requirements.

### 4.1 Feature Engineering
We introduced two interaction features:
- `gpa_attendance = gpa * attendance_rate`
- `study_exam = study_hours_per_week * exam_score`

These features encode combined academic effects that may be more informative than raw variables alone. For example, a student with both high GPA and high attendance should be more likely to receive a scholarship than a student who is strong in only one dimension.

### 4.2 Scaling
We applied `StandardScaler` to the train split and transformed the development split separately. This is important because Logistic Regression and SVM are scale-sensitive. Tree-based models do not require scaling, but using the same preprocessing pipeline for the scaled models ensures a fair comparison.

### 4.3 Leakage Prevention
We explicitly avoided leakage by:
- fitting the scaler only on training data,
- tuning hyperparameters without using the test set,
- engineering features without using the target label,
- comparing all models on the same development protocol.

## 5. Models and Baseline
We reconstructed the midterm baseline and then extended it to four models.

### 5.1 Logistic Regression
Logistic Regression serves as the linear baseline. It is interpretable, efficient, and appropriate for tabular binary classification. Hyperparameter tuning was performed over `C`.

### 5.2 Random Forest
Random Forest is the tree-based baseline. It can model non-linear relationships and feature interactions without scaling. We tuned the number of trees and max depth.

### 5.3 SVM
SVM was added as the additional classical model required by the final project. Because the dataset is relatively small and well-structured, SVM is a strong candidate for high-margin classification.

### 5.4 XGBoost
XGBoost was added as a stronger boosting-based model. It typically performs well on tabular data and also provides feature importance for interpretation.

## 6. System Improvement
The final system improves the midterm baseline in several meaningful ways:
1. Feature engineering with interaction terms.
2. Expansion from two models to four models.
3. More complete metric evaluation using ROC-AUC and PR-AUC.
4. Group-based error analysis instead of only sample-based inspection.
5. Stability evaluation using repeated stratified holdout splits.
6. Model interpretation using coefficients and feature importance.

These are substantive improvements rather than trivial tuning changes.

## 7. Experimental Results
We evaluated all models on the development set using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.

### 7.1 Development Set Comparison
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9200 | 1.0000 | 0.8000 | 0.8889 | 0.9983 | 0.9977 |
| SVM | 0.9200 | 1.0000 | 0.8000 | 0.8889 | 0.9958 | 0.9942 |
| Random Forest | 0.9000 | 1.0000 | 0.7500 | 0.8571 | 0.9771 | 0.9699 |
| XGBoost | 0.8900 | 1.0000 | 0.7250 | 0.8406 | 0.9925 | 0.9905 |

### 7.2 Interpretation of Results
Logistic Regression and SVM are tied on F1-score and accuracy. However, Logistic Regression has the best ROC-AUC and PR-AUC, which suggests better ranking quality and better positive-class discrimination. Random Forest and XGBoost are competitive, but they fall behind on recall and F1-score.

### 7.3 Confusion Matrix Discussion
The models produced no false positives on the development set, but they differed in false negatives. Logistic Regression and SVM each produced 8 false negatives, Random Forest produced 10, and XGBoost produced 11. Because scholarship prediction is a positive-class-sensitive task, recall is especially important. Missing true scholarship candidates is more costly than making a small number of false alarms.

## 8. Error Analysis
Instead of limiting ourselves to a few individual samples, we analyzed error patterns by groups.

### 8.1 Error by Income Group
- Low income: 0.1290
- Medium income: 0.0333
- High income: 0.0769

The low-income group has the highest error rate. This may reflect boundary ambiguity or overlap in student profiles where socioeconomic and academic signals conflict.

### 8.2 Error by GPA Band
- Low GPA: 0.0909
- Borderline GPA: 0.1304
- High GPA: 0.0303

The borderline GPA region is the hardest for the model. This is expected because samples near the decision boundary are more difficult to classify reliably.

### 8.3 Error by Exam Score Band
- Low exam band: 0.0968
- Mid exam band: 0.0385
- High exam band: 0.0930

Errors are lower in the mid-range and higher at the extremes. This suggests the final decision rule is not purely linear and may depend on interactions between exam score and other variables.

### 8.4 Main Error Modes
The main error type is false negatives. This indicates that the model tends to be conservative when predicting scholarship approval. In a practical system, this could be mitigated by threshold tuning or cost-sensitive optimization if the institution wants to reduce missed positive cases.

## 9. Model Interpretation
Interpretability is important because scholarship prediction has policy-like consequences.

### 9.1 Logistic Regression Coefficients
The most influential Logistic Regression coefficients are associated with:
- `gpa`
- `gpa_attendance`
- `exam_score`
- `study_exam`
- `study_hours_per_week`

This is reasonable because academic performance should strongly influence scholarship decisions. The interaction terms also behave as expected, suggesting that the feature engineering strategy is valid.

### 9.2 Tree-Based Feature Importance
Random Forest and XGBoost both show similar important features:
- `gpa`
- `gpa_attendance`
- `study_exam`
- `exam_score`
- `study_hours_per_week`

The agreement across models increases confidence that the model is learning meaningful structure rather than noise.

### 9.3 Interpretation Summary
The interpretation results are aligned with domain intuition: academic performance matters most, and combined academic signals are stronger than single isolated features. Household income and part-time-job hours are useful but less dominant than core academic indicators.

## 10. Stability and Reproducibility
We performed repeated stratified holdout evaluation with 5 splits to test whether the model ranking is stable.

| Model | Mean F1 | Std F1 | Mean Accuracy | Std Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9709 | 0.0391 | 0.9760 | 0.0329 |
| SVM | 0.9386 | 0.0233 | 0.9520 | 0.0179 |
| Random Forest | 0.9136 | 0.0469 | 0.9320 | 0.0390 |
| XGBoost | 0.9114 | 0.0636 | 0.9320 | 0.0482 |

Logistic Regression has the best average F1 and accuracy, and its performance is stable across splits. SVM is also strong, but its mean F1 is lower. XGBoost shows the largest variability and therefore is less attractive as a final deployment choice.

## 11. Final Recommendation
We recommend Logistic Regression as the final model.

### Why Logistic Regression?
1. It achieves the best overall balance of F1, ROC-AUC, and PR-AUC.
2. It is stable across repeated splits.
3. It is easy to explain through coefficients.
4. It is simpler to maintain than the more complex alternatives.

Although SVM has similar F1, Logistic Regression is preferred because it is more interpretable and slightly stronger on ranking metrics. This makes it the most suitable final choice for a scholarship decision-support setting.

## 12. Limitations and Future Work
This project has several limitations:
- The dataset is relatively small.
- There is no fairness audit across demographic or socioeconomic groups.
- Threshold tuning was not optimized for cost-sensitive decision making.
- The final evaluation is based on one development split and a repeated-holdout stability check, not a full nested validation framework.

Future work could include:
- threshold optimization to reduce false negatives,
- fairness and bias analysis,
- calibration analysis,
- more robust cross-validation,
- and deployment-oriented monitoring for data drift.

## 13. Conclusion
This final project transformed the midterm baseline into a more complete and justified machine learning system. The pipeline now includes structured EDA, justified preprocessing, four models, meaningful improvements, rich evaluation, grouped error analysis, model interpretation, and reproducibility checks. Based on the evidence, Logistic Regression is the best final model for this scholarship prediction task.

## Team Contribution
- Student 1: data analysis, preprocessing, baseline modeling, and notebook implementation.
- Student 2: evaluation, interpretation, error analysis, and final report writing.

## Appendix: Reproducibility Notes
The notebook uses the following packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

The workflow is designed to run from top to bottom with fixed random seeds where applicable.
