# Employee Attrition Prediction

## Project Overview

This project uses Machine Learning to predict whether an employee is likely to leave an organization. The goal is to help HR teams identify potential attrition risks and understand patterns associated with employee turnover.

The project covers data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, feature importance analysis, and model evaluation.

## Problem Statement

Employee attrition can increase recruitment costs, disrupt teams, and lead to the loss of experienced employees. This project builds a classification model to identify employees who may leave an organization based on historical HR data.

## Dataset

- **Dataset:** IBM HR Employee Attrition dataset
- **Records:** 1,470 employees
- **Original features:** 35 columns
- **Target variable:** `Attrition`
- **Target classes:** `Yes` and `No`

Four irrelevant columns were removed during preprocessing.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis
4. Feature engineering
5. Stratified train-test split
6. Feature scaling and one-hot encoding
7. Training classification models
8. Hyperparameter tuning using GridSearchCV
9. Feature importance analysis
10. Model saving and evaluation

## Machine Learning Model

The final model is a **Random Forest Classifier** inside a Scikit-learn pipeline.

- Class imbalance handled using `class_weight="balanced"`
- Preprocessing handled using `ColumnTransformer`
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- Hyperparameters optimized using 5-fold cross-validation and F1-score

### Best Hyperparameters

- `n_estimators`: 200
- `max_depth`: 5
- `min_samples_split`: 5
- `min_samples_leaf`: 1

## Final Model Evaluation

Evaluation on the held-out test dataset of 294 employees:

| Metric | Score |
|---|---:|
| Accuracy | 79.25% |
| Precision | 39.71% |
| Recall | 57.45% |
| F1-score | 46.96% |
| ROC-AUC | 0.7888 |

### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 206 | 41 |
| Actual Yes | 20 | 27 |

The model identified 27 of the 47 employees in the actual attrition class. The precision score indicates that some predicted high-risk employees did not belong to the actual attrition class.

## Important Features

The Random Forest model assigned high feature importance to:

1. Monthly Income
2. Overtime
3. Age
4. Years at Company
5. Total Working Years
6. Years With Current Manager
7. Stock Option Level
8. Job Level

Feature importance reflects the model's use of these features; it does not establish causation.

## Model Saving

The trained pipeline is saved using Joblib:

`notebooks/models/employee_attrition_rf.pkl`

The saved pipeline includes the preprocessing steps and trained classifier.

## Future Improvements

- Improve recall and F1-score through threshold tuning.
- Evaluate additional classification algorithms.
- Use explainability techniques such as SHAP.
- Build a REST API using FastAPI.
- Add a simple interface for HR users.
- Monitor model performance and potential bias.

## Disclaimer

This model is an educational machine learning project. Predictions should support further investigation, not serve as the sole basis for employment decisions.
