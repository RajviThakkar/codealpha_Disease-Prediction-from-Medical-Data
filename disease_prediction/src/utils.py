"""Model training, evaluation, and plotting helpers shared by main.py and predict.py."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_models(random_state=42):
    """Return the 4 classifiers used in this project, unfitted."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, eval_metric="logloss", random_state=random_state
        )
    else:
        # xgboost isn't installed; HistGradientBoosting is a close substitute
        # (also gradient-boosted trees) so the comparison still has 4 models.
        print("xgboost not installed -- using sklearn's HistGradientBoostingClassifier instead.")
        models["Gradient Boosting (xgboost unavailable)"] = HistGradientBoostingClassifier(
            max_iter=200, random_state=random_state
        )

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=5):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") \
        else model.decision_function(X_test)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "CV Accuracy (mean)": cv_scores.mean(),
        "CV Accuracy (std)": cv_scores.std(),
    }
    return metrics, y_pred, y_proba


def run_all_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all models; return a results table plus fitted models/predictions."""
    models = get_models()
    results, fitted_models, preds, probas = {}, {}, {}, {}

    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        fitted_models[name] = model
        preds[name] = y_pred
        probas[name] = y_proba

    results_df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
    return results_df, fitted_models, preds, probas


def print_report(dataset_name, results_df, y_test, preds):
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {dataset_name}")
    print(f"{'=' * 70}")
    print(results_df.round(4).to_string())

    best_model = results_df["ROC-AUC"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_model}")
    print(f"\nClassification report for {best_model}:")
    print(classification_report(y_test, preds[best_model]))
    return best_model


def plot_confusion_matrices(dataset_name, y_test, preds, out_path):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"Confusion Matrices - {dataset_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(dataset_name, y_test, probas, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {dataset_name}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(dataset_name, model, feature_names, out_path, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return False

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    importances.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Top Feature Importances - {dataset_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
