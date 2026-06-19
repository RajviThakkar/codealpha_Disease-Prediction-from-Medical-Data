# Disease Prediction from Medical Data

A machine learning project that predicts the presence of three diseases
(heart disease, diabetes, breast cancer) from structured patient data,
using four classifiers: Logistic Regression, SVM, Random Forest, and
XGBoost.

Built as part of a Machine Learning Internship, demonstrating disease prediction using supervised learning techniques and real-world medical datasets.

## Features

- Predicts Heart Disease, Diabetes, and Breast Cancer
- Uses Logistic Regression, SVM, Random Forest, and XGBoost
- Compares models using Accuracy, Precision, Recall, F1, and ROC-AUC
- Performs 5-fold cross-validation
- Generates confusion matrices and ROC curves
- Interactive patient prediction through command-line interface

## Datasets

| Dataset | Source | Samples | Features |
|---|---|---|---|
| Heart Disease | UCI Cleveland | 303 | 13 |
| Diabetes | Pima Indians Diabetes | 768 | 8 |
| Breast Cancer | UCI Wisconsin (Diagnostic) | 569 | 30 |

Breast cancer loads directly from scikit-learn. Heart disease and
diabetes are downloaded from public sources the first time `main.py`
or `predict.py` is run, and cached in `data/` for later runs.

If you're offline, download the CSVs manually and place them at
`data/heart_disease.csv` and `data/diabetes.csv` (no headers — see
column order in `src/data_loaders.py`).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train and evaluate all 4 models on all 3 datasets:

```bash
cd src
python main.py
```

This prints accuracy/precision/recall/F1/ROC-AUC for each model and
saves metrics + plots (confusion matrices, ROC curves, feature
importance) to `outputs/`.

Predict a single patient interactively:

```bash
cd src
python predict.py
```

Pick a dataset, enter the patient's values when prompted, and the
script prints each model's prediction with a confidence score and an
overall consensus. Breast cancer only asks for 12 of its 30 features
(the rest are filled with dataset medians) since typing all 30 by
hand isn't practical.

## Project structure

```
disease_prediction/
├── requirements.txt
├── README.md
├── data/                  # downloaded datasets are cached here
├── outputs/                # metrics and plots from main.py
└── src/
    ├── data_loaders.py     # loads/downloads the 3 datasets
    ├── input_specs.py      # prompts and valid ranges for predict.py
    ├── utils.py            # model training, evaluation, plotting
    ├── main.py              # trains + evaluates + compares all models
    └── predict.py            # interactive single-patient prediction
```

## Approach

- 80/20 stratified train-test split, features scaled with StandardScaler
- Each model evaluated with accuracy, precision, recall, F1, ROC-AUC,
  and 5-fold cross-validation accuracy
- Random Forest feature importance plotted to see which inputs matter most
- Accuracy alone isn't enough here since missing a real disease case
  (false negative) is worse than a false alarm, so recall and ROC-AUC
  are reported alongside it

## Notes / limitations

- If `xgboost` fails to install on your machine, the code falls back
  to scikit-learn's `HistGradientBoostingClassifier`, which is also a
  gradient-boosted tree model. A message is printed when this happens.
- The diabetes dataset uses `0` as a placeholder for missing values in
  several columns; these are replaced with the column median before
  training.
- This is for learning purposes, not medical use. Predictions should
  not be treated as a diagnosis.
