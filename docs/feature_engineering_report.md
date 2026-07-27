# Feature Engineering and Preprocessing Report (Titanic Dataset)

## Overview

This report documents the end-to-end feature engineering and preprocessing steps applied to the Titanic dataset prior to model training to ensure data quality and optimal pipeline performance.

## Preprocessing Steps Implemented

1. **Missing Value Imputation:** Numerical features (such as Age) were imputed using the median strategy to reduce outlier sensitivity, while categorical features used the most frequent category.
2. **Categorical Encoding:** Text-based categorical columns (such as Sex and Embarked) were transformed into numerical format using One-Hot Encoding with `handle_unknown='ignore'`.
3. **Feature Scaling:** Continuous numerical attributes were standardized using `StandardScaler` to achieve a zero mean and unit variance, ensuring stable convergence for gradient-based models.
4. **Data Partitioning:** The dataset was split into an 80% training set and a 20% testing set with stratification (`random_state=42`) to maintain class balance across splits.
