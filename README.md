# AI Internship Day 04 - Feature Engineering & Classical Machine Learning Pipeline (Titanic Dataset)

---

## 📌 Project Overview

This repository contains the solution and documentation for my AI Internship Day 4 project. In this project, I built an end-to-end machine learning pipeline using the Titanic dataset to predict passenger survival outcomes based on feature engineering, missing value imputation, encoding, scaling, and classical machine learning models.

---

## 🚀 Learning Outcomes

- Applied robust missing-value imputation techniques for numerical and categorical features.
- Implemented categorical variable encoding via One-Hot Encoding.
- Standardized continuous numerical variables using standard scaling.
- Built reproducible scikit-learn Pipelines and ColumnTransformers.
- Trained and evaluated baseline, linear, and tree-based classification models with 5-fold cross-validation.

---

## 🗂️ Project Architecture & Folder Structure

To ensure a clean, modular, and maintainable project architecture, the following folder structure was implemented:

```text
ai-day4-feature-engineering-ml-jahanzaib/
│
├── data/
│   ├── processed/
│   └── raw/
│       └── titanic.csv
│
├── docs/
│   ├── business_recommendation.md
│   └── feature_engineering_report.md
│
├── notebooks/
│   └── day4_ml_pipeline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   └── train_models.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```

## ⚠️ Important Rule

Files inside the data/raw/ folder are treated as an immutable source of truth and must never be edited, modified, or overwritten directly.

## 💻 Environment Setup & Installation

Open your project folder in VS Code.

Ensure that Python is installed on your system.

Open the terminal and run the following command to install dependencies:

Bash
python -m pip install -r requirements.txt

## 🏃‍♂️ Running the Project

To run the machine learning pipeline notebook, open notebooks/day4_ml_pipeline.ipynb in VS Code or Jupyter Notebook.

## 📊 Key Tasks Performed

Dataset Loading & Inspection: Loaded the Titanic dataset (titanic.csv) and inspected feature types and structures.

Target Variable Selection: Selected and justified the Survived target variable for passenger survival prediction.

Missing Value Imputation: Handled missing values using median imputation for numerical features and most-frequent imputation for categorical features.

Encoding: Converted categorical features into numerical format using One-Hot Encoding.

Scaling: Scaled numerical variables using standard scaling.

Feature Selection: Selected relevant predictors to optimize model efficiency.

Data Splitting: Partitioned data into 80% training and 20% testing sets using stratification (random_state=42).

Pipeline Construction: Built an integrated scikit-learn preprocessing pipeline using ColumnTransformer.

Baseline Model: Trained a Dummy baseline classifier to establish lower-bound performance metrics.

Logistic Regression: Trained a linear classification model.

Random Forest: Trained an ensemble tree-based classification model.

Cross-Validation: Performed 5-fold cross-validation across all models.

Model Comparison: Compared Accuracy, Precision, Recall, F1-Score, and CV scores.

Model Interpretation: Analyzed results and recommended the optimal production model.

Workflow Documentation: Documented the complete pipeline architecture and business insights.

## 📈 Model Performance & Comparison Results

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **80.44%** | 0.79 | 0.67 | 0.72 |
| **Random Forest Classifier** | **81.56%** | 0.80 | 0.70 | 0.74 |

## 📋 Project Deliverables Checklist

[1] Python Jupyter Notebook (notebooks/day4_ml_pipeline.ipynb)

[2] Reusable Python Scripts (src/)

[3] Feature Engineering Report (docs/feature_engineering_report.md)

[4] Business Recommendation Report (docs/business_recommendation.md)

[5] Model Comparison & Evaluation Table (README.md)

[6] Immutable Raw Dataset (data/raw/titanic.csv)

**Author**  
**Student Name: Muhammad Jahanzaib Azhar**  
**Internship Program: AI Internship**  
**Date: July 27, 2026**
