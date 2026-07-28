# Final Model Evaluation and Limitations Report

## 1. Overview

This report provides a comprehensive analysis of the Random Forest model trained on the Titanic dataset, detailing performance metrics, evaluation plots, model limitations, and recommended future improvements.

## 2. Evaluation Metrics & Performance

* **ROC-AUC Score:** 0.6576
* **PR-AUC Score:** 0.5518

## 3. Generated Evaluation & Explainability Figures

The following visualizations have been successfully generated and stored in the `outputs/figures/` directory:
**Confusion Matrix (`confusion_matrix.png`):** Summarizes the classification outcomes (true positives, true negatives, false positives, and false negatives).
**ROC Curve (`roc_curve.png`):** Illustrates the diagnostic ability of the classifier across various threshold settings.
**Precision-Recall Curve (`precision_recall_curve.png`):** Evaluates the trade-off between precision and recall for the positive class.
**Calibration Curve (`calibration_curve.png`):** Assesses how closely the predicted probabilities align with actual likelihoods.
**SHAP Summary Plot (`shap_summary.png`):** Displays the global impact and feature importance derived from SHAP values.
**SHAP Waterfall Plot (`shap_waterfall.png`):** Breaks down how individual feature values contributed to a specific prediction.

## 4. Model Limitations

* **Dataset Limitations:** The dataset size is relatively small, which can restrict the model's capacity to generalize complex patterns across unseen distributions.
* **Class Imbalance:** Minor imbalances in the target variable categories can bias predictions towards the majority class, impacting minority class recall.
* **Feature Limitations:** Important contextual features may be missing from the raw data, restricting predictive depth.
* **Data Quality Issues:** Presence of missing or noisy data points can introduce minor errors during training and inference.
* **Risk of Overfitting:** Tree-based models like Random Forest can overfit if hyperparameters lack strict regularization.
* **Risk of Underfitting:** Over-simplifying features or constraints can cause the model to miss subtle non-linear relationships.
* **Ethical Considerations:** Predictions involving human-centric data require careful auditing to prevent unintended algorithmic bias and ensure fairness.
* **Explainability Limitations:** While SHAP provides strong visibility, complex feature interactions can sometimes remain challenging to interpret intuitively for non-technical stakeholders.

## 5. Recommended Next Experiments

* **Hyperparameter Tuning:** Apply Grid Search or Random Search to optimize hyperparameters such as `n_estimators`, `max_depth`, and `min_samples_split` to boost performance.
* **Collecting More Data:** Gather a larger and more diverse dataset to enhance overall robustness and model stability.
* **Advanced Feature Engineering:** Build interaction terms, polynomial features, or domain-specific aggregations to supply stronger predictive signals.
* **Trying Ensemble Models:** Experiment with stacking or blending Gradient Boosting models (XGBoost/LightGBM) alongside Random Forest to reduce variance and error rates.
* **Addressing Class Imbalance:** Utilize SMOTE (Synthetic Minority Over-sampling Technique) or adjust class weights to manage false negatives more effectively.
