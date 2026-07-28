# AI Lab 99 Internship Program - Day 5 Activity Task

## Model Evaluation, Explainability, and Error Analysis using Machine Learning

---

## 📌 Project Overview

This repository contains the solution and documentation for my AI Internship Day 5 project. Building upon the classification models trained previously, this project focuses on rigorous model evaluation from a technical and business perspective, threshold tuning, cost-sensitive analysis, feature importance ranking, SHAP-based model explainability, error analysis, and actionable business recommendations using the Titanic dataset.

---

## 🚀 Learning Outcomes

- Evaluated classification models using confusion matrices, accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Performed classification threshold tuning to optimize performance for business requirements.
- Assessed model calibration using reliability diagrams.
- Conducted cost-sensitive evaluation to minimize business financial risks arising from prediction errors.
- Generated feature importance rankings for tree-based models.
- Interpreted model predictions using SHAP (SHapley Additive exPlanations) summary, bar, and waterfall plots.
- Executed manual error analysis and built a structured error taxonomy.
- Documented model limitations and recommended future improvements.

---

## 🗂️ Project Architecture & Folder Structure

To ensure a clean, modular, and maintainable project architecture, the following folder structure was implemented:

```text
ai-day5-model-evaluation-jahanzaib/
│
├── data/
│   ├── processed/
│   │   ├── test_processed.csv
│   │   └── train_processed.csv
│   └── raw/
│       └── titanic.csv
│
├── docs/
│   └── final_evaluation_report.md
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── day5_evaluation.ipynb
│
├── outputs/
│   └── figures/
│       ├── calibration_curve.png
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       ├── precision_recall_curve.png
│       ├── roc_curve.png
│       ├── shap_summary.png
│       └── shap_waterfall.png
│
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   └── explain.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```

## ⚠️ Important Rule

Files inside the `data/raw/` folder are treated as an immutable source of truth and must never be edited, modified, or overwritten directly.

---

## 💻 Environment Setup & Installation

- Open your project folder in VS Code.
- Ensure that Python is installed on your system.
- Open the terminal and run the following command to install dependencies:

bash

py -m pip install -r requirements.txt

## 🏃‍♂️ Running the Project

To run the interactive model evaluation pipeline, open notebooks/day5_evaluation.ipynb in VS Code or Jupyter Notebook, or execute the python source scripts located in the src/ directory:

Bash

python src/evaluate.py

python src/explain.py

## 📊 Key Tasks Performed

1. **Model Loading:** Loaded the best-performing model (`best_model.pkl`) and processed test datasets.
2. **Performance Metrics:** Calculated Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
3. **Confusion Matrix Analysis:** Generated confusion matrix plots and analyzed True/False Positives and Negatives.
4. **Threshold Tuning:** Evaluated alternative classification thresholds (0.30 to 0.70) to optimize performance.
5. **ROC & PR Curves:** Generated and interpreted ROC and Precision-Recall curves.
6. **Model Calibration:** Plotted and reviewed probability calibration curves.
7. **Cost-Sensitive Evaluation:** Computed total business error costs based on assigned False Positive ($50) and False Negative ($200) costs.
8. **Feature Importance:** Extracted top feature rankings from tree-based models.
9. **SHAP Explainability:** Generated SHAP summary, bar, and waterfall plots for deep model interpretability.
10. **Error Analysis:** Categorized misclassified records into an error taxonomy and inspected failure reasons.
11. **Limitations & Recommendations:** Documented dataset constraints and proposed 5 future experimentation strategies.

## 📈 Model Evaluation Summary

| Metric / Evaluation Component | Value / Outcome | Business Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC Score** | High performance | Strong capability in distinguishing between survival classes. |
| **PR-AUC Score** | Balanced metric | Effective performance handling class distributions. |
| **Total Error Cost** | **$5,000** | Calculated based on FP ($50) and FN ($200) error penalties. |
| **Primary Error Risk** | False Negatives ($4,200) | High business penalty for missed positive predictions. |

## 📋 Project Deliverables Checklist

[1] Python Jupyter Notebook (notebooks/day5_evaluation.ipynb)  
[2] Reusable Python Scripts (src/evaluate.py, src/explain.py)  
[3] Final Evaluation Report (docs/final_evaluation_report.md)  
[4] Saved Model Visualizations (outputs/figures/)  
[5] Serialized Best Model (models/best_model.pkl)  
[6] Immutable Raw & Processed Datasets (data/)  

**Author**  
**Student Name: Muhammad Jahanzaib Azhar**  
**Internship Program: AI Lab 99 Internship Program**  
**Date: July 28, 2026**
