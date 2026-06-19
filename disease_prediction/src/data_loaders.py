"""
Loads the three datasets used in this project.

Breast cancer is bundled with scikit-learn. Heart disease and diabetes
are downloaded from public sources on first run and cached locally in
the data/ folder, so later runs work offline.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
HEART_PATH = os.path.join(DATA_DIR, "heart_disease.csv")
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DIABETES_PATH = os.path.join(DATA_DIR, "diabetes.csv")
DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]


def _ensure_downloaded(path, url):
    if os.path.exists(path):
        return
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise RuntimeError(
            f"Could not download dataset from {url} ({e}).\n"
            f"Download it manually and save it as {path}, then rerun."
        )


def load_breast_cancer_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign
    return X, y, list(X.columns)


def load_heart_disease_data():
    _ensure_downloaded(HEART_PATH, HEART_URL)
    df = pd.read_csv(HEART_PATH, names=HEART_COLUMNS, na_values="?")
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    df["target"] = (df["target"] > 0).astype(int)  # 0-4 severity -> binary

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, list(X.columns)


def load_diabetes_data():
    _ensure_downloaded(DIABETES_PATH, DIABETES_URL)
    df = pd.read_csv(DIABETES_PATH, names=DIABETES_COLUMNS)

    # 0 is used as a missing-value placeholder in these columns
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y, list(X.columns)
