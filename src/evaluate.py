import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

def main():
    print("Starting Model Evaluation Pipeline...")


    X_test = pd.read_csv('./data/processed/test_processed.csv')
    y_test = X_test['Survived']
    X_test = X_test.drop(columns=['Survived'])

  
    X_test = X_test.select_dtypes(include=['number'])

    with open('./models/best_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    
    if hasattr(clf, "feature_names_in_"):
        X_test = X_test[[col for col in clf.feature_names_in_ if col in X_test.columns]]

    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

   
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

  
    os.makedirs('./outputs/figures', exist_ok=True)

  
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix')
    plt.savefig('./outputs/figures/confusion_matrix.png', bbox_inches='tight')
    plt.close()

  
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('./outputs/figures/roc_curve.png', bbox_inches='tight')
    plt.close()

    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='green', label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig('./outputs/figures/precision_recall_curve.png', bbox_inches='tight')
    plt.close()

    
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', color='purple', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    plt.savefig('./outputs/figures/calibration_curve.png', bbox_inches='tight')
    plt.close()

    print("Evaluation script executed successfully and figures saved!")

if __name__ == '__main__':
    main()