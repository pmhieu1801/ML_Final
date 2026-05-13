# Scholarship Prediction Final Project

This repository contains the final version of the scholarship prediction system for the course *Introduction to Machine Learning*.

## Contents
- `notebook.ipynb`: main end-to-end machine learning workflow
- `data/`: train and dev CSV files
- `report/`: report draft and submission checklist
- `submission_format/`: reserved for required submission formatting

## How to Run
1. Activate the environment.
2. Open `notebook.ipynb`.
3. Run the notebook from top to bottom.
4. Ensure `predictions.csv` is generated in the project root.

## Main Models
- Logistic Regression
- Random Forest
- SVM
- XGBoost

## Main Improvements Over Midterm
- Feature engineering with interaction terms
- Additional model comparison
- ROC-AUC and PR-AUC evaluation
- Group-based error analysis
- Model interpretation
- Stability check across multiple splits

## Output Files
- `report.pdf`
- `notebook.ipynb`
- `predictions.csv`

## Notes
- The best final model on the current development split is Logistic Regression.
- The notebook avoids leakage by fitting preprocessing only on training data.
# ML_Final
