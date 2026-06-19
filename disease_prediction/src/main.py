"""
Trains and compares Logistic Regression, SVM, Random Forest, and XGBoost
on the heart disease, diabetes, and breast cancer datasets.

Usage:
    python main.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from data_loaders import load_heart_disease_data, load_diabetes_data, load_breast_cancer_data
from utils import run_all_models, print_report, plot_confusion_matrices, plot_roc_curves, plot_feature_importance

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pipeline(dataset_name, loader_fn):
    print(f"\nLoading {dataset_name} dataset...")
    X, y, feature_names = loader_fn()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    results_df, fitted_models, preds, probas = run_all_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    best_model_name = print_report(dataset_name, results_df, y_test, preds)

    dataset_slug = dataset_name.lower().replace(" ", "_")
    results_df.round(4).to_csv(os.path.join(OUTPUT_DIR, f"{dataset_slug}_metrics.csv"))

    plot_confusion_matrices(
        dataset_name, y_test, preds,
        os.path.join(OUTPUT_DIR, f"{dataset_slug}_confusion_matrices.png")
    )
    plot_roc_curves(
        dataset_name, y_test, probas,
        os.path.join(OUTPUT_DIR, f"{dataset_slug}_roc_curves.png")
    )

    rf_model = fitted_models.get("Random Forest")
    if rf_model is not None:
        plot_feature_importance(
            dataset_name, rf_model, feature_names,
            os.path.join(OUTPUT_DIR, f"{dataset_slug}_feature_importance.png")
        )

    return {
        "dataset": dataset_name,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "best_model": best_model_name,
        "best_roc_auc": float(results_df.loc[best_model_name, "ROC-AUC"]),
        "metrics": results_df.round(4).to_dict(orient="index"),
    }


def main():
    summary = [
        run_pipeline("Heart Disease", load_heart_disease_data),
        run_pipeline("Diabetes", load_diabetes_data),
        run_pipeline("Breast Cancer", load_breast_cancer_data),
    ]

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for s in summary:
        print(f"{s['dataset']:<18} | n={s['n_samples']:<5} | features={s['n_features']:<3} "
              f"| best model: {s['best_model']:<35} | ROC-AUC={s['best_roc_auc']:.4f}")

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
