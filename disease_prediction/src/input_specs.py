"""
Field definitions used by predict.py to prompt for patient values.

Each field is (column_name, prompt_label, kind, extra):
  kind = "float" | "int" | "choice"
  extra = (min, max) for float/int, or a list of (value, label) for choice
"""

HEART_FIELDS = [
    ("age", "Age (years)", "int", (1, 120)),
    ("sex", "Sex", "choice", [(1, "Male"), (0, "Female")]),
    ("cp", "Chest pain type", "choice", [
        (0, "Typical angina"), (1, "Atypical angina"),
        (2, "Non-anginal pain"), (3, "Asymptomatic")
    ]),
    ("trestbps", "Resting blood pressure (mm Hg)", "float", (60, 250)),
    ("chol", "Serum cholesterol (mg/dl)", "float", (80, 700)),
    ("fbs", "Fasting blood sugar > 120 mg/dl?", "choice", [(1, "Yes"), (0, "No")]),
    ("restecg", "Resting ECG result", "choice", [
        (0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")
    ]),
    ("thalach", "Maximum heart rate achieved", "float", (50, 250)),
    ("exang", "Exercise-induced angina?", "choice", [(1, "Yes"), (0, "No")]),
    ("oldpeak", "ST depression induced by exercise", "float", (0, 10)),
    ("slope", "Slope of peak exercise ST segment", "choice", [
        (0, "Upsloping"), (1, "Flat"), (2, "Downsloping")
    ]),
    ("ca", "Number of major vessels colored by fluoroscopy (0-3)", "int", (0, 3)),
    ("thal", "Thalassemia result", "choice", [
        (3, "Normal"), (6, "Fixed defect"), (7, "Reversible defect")
    ]),
]

DIABETES_FIELDS = [
    ("Pregnancies", "Number of pregnancies", "int", (0, 20)),
    ("Glucose", "Plasma glucose concentration (mg/dl)", "float", (40, 300)),
    ("BloodPressure", "Diastolic blood pressure (mm Hg)", "float", (20, 180)),
    ("SkinThickness", "Triceps skin fold thickness (mm)", "float", (0, 100)),
    ("Insulin", "2-hour serum insulin (mu U/ml)", "float", (0, 900)),
    ("BMI", "Body mass index (kg/m^2)", "float", (10, 80)),
    ("DiabetesPedigreeFunction", "Diabetes pedigree function (genetic risk score, ~0.05-2.5)", "float", (0.0, 3.0)),
    ("Age", "Age (years)", "int", (1, 120)),
]

# Breast cancer has 30 numeric features, too many to type by hand here,
# so only the 12 most informative ones are prompted; the rest default
# to dataset medians (see predict.py).
BREAST_CANCER_KEY_FIELDS = [
    ("mean radius", "Mean radius of cell nuclei", "float", (5, 30)),
    ("mean texture", "Mean texture (gray-scale value std dev)", "float", (5, 40)),
    ("mean perimeter", "Mean perimeter of cell nuclei", "float", (40, 200)),
    ("mean area", "Mean area of cell nuclei", "float", (100, 2600)),
    ("mean smoothness", "Mean smoothness (local variation in radius)", "float", (0.02, 0.2)),
    ("mean compactness", "Mean compactness", "float", (0.0, 0.4)),
    ("mean concavity", "Mean concavity", "float", (0.0, 0.5)),
    ("mean concave points", "Mean concave points", "float", (0.0, 0.3)),
    ("mean symmetry", "Mean symmetry", "float", (0.1, 0.4)),
    ("worst radius", "Worst (largest) radius observed", "float", (5, 40)),
    ("worst perimeter", "Worst (largest) perimeter observed", "float", (40, 260)),
    ("worst area", "Worst (largest) area observed", "float", (100, 4300)),
]

DATASET_SPECS = {
    "heart": {
        "label": "Heart Disease",
        "fields": HEART_FIELDS,
        "full_features": True,
        "target_names": {0: "No heart disease detected", 1: "Heart disease detected"},
    },
    "diabetes": {
        "label": "Diabetes",
        "fields": DIABETES_FIELDS,
        "full_features": True,
        "target_names": {0: "Not diabetic", 1: "Diabetic"},
    },
    "breast_cancer": {
        "label": "Breast Cancer",
        "fields": BREAST_CANCER_KEY_FIELDS,
        "full_features": False,  # only a subset is prompted; rest filled with medians
        "target_names": {0: "Malignant", 1: "Benign"},
    },
}
