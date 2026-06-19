"""
Interactive command-line predictor.

Trains all 4 models on a chosen dataset, then asks for a patient's
values field by field and prints each model's prediction.

Usage:
    python predict.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from data_loaders import load_heart_disease_data, load_diabetes_data, load_breast_cancer_data
from input_specs import DATASET_SPECS
from utils import get_models

LOADERS = {
    "heart": load_heart_disease_data,
    "diabetes": load_diabetes_data,
    "breast_cancer": load_breast_cancer_data,
}


def choose_dataset():
    print("\nWhich disease would you like to predict?")
    print("  1. Heart Disease")
    print("  2. Diabetes")
    print("  3. Breast Cancer")
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "1":
            return "heart"
        if choice == "2":
            return "diabetes"
        if choice == "3":
            return "breast_cancer"
        print("Please type 1, 2, or 3.")


def train_models(dataset_key):
    # Trained on the full dataset since this is for making a single
    # prediction, not for evaluation (that happens in main.py).
    print(f"\nLoading data and training models for {DATASET_SPECS[dataset_key]['label']}...")
    X, y, feature_names = LOADERS[dataset_key]()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    models = get_models()
    for model in models.values():
        model.fit(X_scaled, y)

    feature_medians = X.median()
    print(f"Done. Trained on {len(X)} samples, {len(feature_names)} features.\n")
    return models, scaler, feature_names, feature_medians


def prompt_for_value(label, kind, extra):
    if kind == "choice":
        print(f"\n{label}:")
        for i, (_, choice_label) in enumerate(extra, start=1):
            print(f"  {i}. {choice_label}")
        while True:
            raw = input(f"Enter 1-{len(extra)}: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(extra):
                return extra[int(raw) - 1][0]
            print(f"Please enter a number between 1 and {len(extra)}.")

    lo, hi = extra
    cast = int if kind == "int" else float
    while True:
        raw = input(f"{label} [{lo}-{hi}]: ").strip()
        try:
            val = cast(raw)
            if lo <= val <= hi:
                return val
            print(f"Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("Please enter a valid number.")


def collect_patient_input(dataset_key, feature_names, feature_medians):
    spec = DATASET_SPECS[dataset_key]
    print("Please enter the patient's details below.\n")

    values = {}
    for field_name, label, kind, extra in spec["fields"]:
        values[field_name] = prompt_for_value(label, kind, extra)

    # Breast cancer only prompts for a subset of its 30 features;
    # fill in the rest with dataset medians.
    if not spec["full_features"]:
        for fname in feature_names:
            if fname not in values:
                values[fname] = feature_medians[fname]

    return pd.DataFrame([values])[feature_names]  # enforce column order


def predict_and_report(models, scaler, row, dataset_key):
    spec = DATASET_SPECS[dataset_key]
    target_names = spec["target_names"]
    row_scaled = pd.DataFrame(scaler.transform(row), columns=row.columns)

    print(f"\n{'=' * 60}")
    print(f"  PREDICTION RESULTS - {spec['label']}")
    print(f"{'=' * 60}")

    votes = []
    for name, model in models.items():
        pred = model.predict(row_scaled)[0]
        proba = model.predict_proba(row_scaled)[0][pred] if hasattr(model, "predict_proba") else None

        label = target_names.get(pred, str(pred))
        votes.append(pred)
        proba_str = f"  (confidence: {proba * 100:.1f}%)" if proba is not None else ""
        print(f"  {name:<42} -> {label}{proba_str}")

    consensus = round(np.mean(votes))
    n_agree = sum(1 for v in votes if v == consensus)
    print(f"\n  CONSENSUS ({n_agree}/{len(votes)} models agree): {target_names.get(consensus)}")
    print(f"{'=' * 60}")
    print("\nThis is a machine learning estimate for demo purposes only,")
    print("not a medical diagnosis. Consult a healthcare professional")
    print("for any real medical decision.")


def main():
    print("=" * 60)
    print("  INTERACTIVE DISEASE PREDICTOR")
    print("=" * 60)

    dataset_key = choose_dataset()
    models, scaler, feature_names, feature_medians = train_models(dataset_key)

    while True:
        row = collect_patient_input(dataset_key, feature_names, feature_medians)
        predict_and_report(models, scaler, row, dataset_key)

        again = input("\nPredict another patient? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nokay!")


if __name__ == "__main__":
    main()
