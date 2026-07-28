import os
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

def main():
    print("Starting Model Explainability Pipeline...")

    # 1. Load Data and Model
    X_test = pd.read_csv('./data/processed/test_processed.csv')
    if 'Survived' in X_test.columns:
        X_test = X_test.drop(columns=['Survived'])

    
    X_test = X_test.select_dtypes(include=['number'])

    with open('./models/best_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    if hasattr(clf, "feature_names_in_"):
        X_test = X_test[[col for col in clf.feature_names_in_ if col in X_test.columns]]

    # 2. Output Directory
    os.makedirs('./outputs/figures', exist_ok=True)

    # 3. SHAP Explainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_test)

    # 4. SHAP Summary Plot
    plt.figure(figsize=(8, 6))
    # Check if shap_values has 3 dimensions (multiclass/binary probability) or 2 dimensions
    if len(shap_values.values.shape) == 3:
        shap_vals_to_plot = shap_values[:, :, 1]
        waterfall_val = shap_values[0, :, 1]
    else:
        shap_vals_to_plot = shap_values
        waterfall_val = shap_values[0]

    shap.summary_plot(shap_vals_to_plot, X_test, show=False)
    plt.savefig('./outputs/figures/shap_summary.png', bbox_inches='tight')
    plt.close()

    # 5. SHAP Waterfall Plot for the first customer
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(waterfall_val, show=False)
    plt.savefig('./outputs/figures/shap_waterfall.png', bbox_inches='tight')
    plt.close()

    print("SHAP explainability plots generated and saved successfully!")

if __name__ == '__main__':
    main()