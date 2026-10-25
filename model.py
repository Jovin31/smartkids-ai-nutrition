
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "malnutrition_balanced_synthetic_dataset.csv"
MODEL_FILE = BASE_DIR / "malnutrition_model_final.pkl"
FEATURES_FILE = BASE_DIR / "model_features.pkl"

FEATURES = [
    "Age_months",
    "Weight_kg",
    "Height_cm",
    "Gender",
]

TARGET = "Nutrition_Status"

VALID_GENDERS = {"Male", "Female"}
VALID_STATUSES = {"Normal", "Underweight", "Overweight"}


# =========================================================
# EXTRACT & VALIDATE DATA
# =========================================================

def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, sep=";")

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
        )

    df = df[required_columns].copy()

    df["Gender"] = df["Gender"].astype("string").str.strip()
    df[TARGET] = df[TARGET].astype("string").str.strip()

    for col in ["Age_months", "Weight_kg", "Height_cm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Gender"] = df["Gender"].map({
        "Male": 0,
        "Female": 1,
    })

    df[TARGET] = df[TARGET].astype("string")

    valid_rows = (
        df[["Age_months", "Weight_kg", "Height_cm", "Gender"]]
        .notna()
        .all(axis=1)
        & df["Age_months"].between(12, 60)
        & (df["Weight_kg"] > 0)
        & (df["Height_cm"] > 0)
        & df[TARGET].isin(VALID_STATUSES)
    )

    rejected_rows = int((~valid_rows).sum())
    df = df.loc[valid_rows].copy()

    if df.empty:
        raise ValueError("Tidak ada data valid untuk training.")

    if set(df[TARGET].unique()) != VALID_STATUSES:
        raise ValueError(
            "Dataset harus memiliki ketiga label: "
            "Normal, Underweight, dan Overweight."
        )

    print(f"Dataset berhasil dimuat: {len(df)} baris")
    print(f"Baris yang ditolak: {rejected_rows}")

    return df


# =========================================================
# TRAINING & HYPERPARAMETER TUNING
# =========================================================

def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
    }

    

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=1,
        refit=True,
    )

    search.fit(X_train, y_train)


    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    print("\n=== MODEL EVALUATION ===")
    print(f"Best parameters: {search.best_params_}")
    print(
        "Best CV Macro F1:",
        round(search.best_score_, 4),
    )
    print(
        "Test Accuracy:",
        round(accuracy_score(y_test, predictions), 4),
    )
    print(
        "Test Macro F1:",
        round(
            f1_score(
                y_test,
                predictions,
                average="macro",
            ),
            4,
        ),
    )

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, predictions))

    print("=== CONFUSION MATRIX ===")
    print(
        pd.DataFrame(
            confusion_matrix(
                y_test,
                predictions,
                labels=best_model.classes_,
            ),
            index=best_model.classes_,
            columns=best_model.classes_,
        )
    )

    return best_model


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(FEATURES, FEATURES_FILE)

    print("\n=== SAVED FILES ===")
    print(f"Model: {MODEL_FILE}")
    print(f"Features: {FEATURES_FILE}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 55)
    print("SMARTKIDS - MODEL TRAINING")
    print("=" * 55)

    dataset = load_dataset()
    model = train_model(dataset)
    save_model(model)

    print("\nTraining selesai.")


if __name__ == "__main__":
    main()
