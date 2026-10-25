
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
df = pd.read_csv("malnutrition_balanced_synthetic_dataset.csv", sep=";")

# 2. Siapkan fitur dan target
features = ["Age_months", "Weight_kg", "Height_cm", "Gender"]
X = df[features].copy()
y = df["Nutrition_Status"].copy()

# Validasi kategori gender
X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})

if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Ada nilai kosong atau kategori Gender tidak dikenal.")

# 3. Split sebelum preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Baseline: tebak kelas terbanyak
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("=== BASELINE ===")
print("Accuracy:", round(accuracy_score(y_test, baseline_pred), 4))
print("Macro F1:", round(f1_score(y_test, baseline_pred, average="macro"), 4))

# 5. Random Forest dengan preprocessing di dalam pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n=== RANDOM FOREST ===")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro F1:", round(f1_score(y_test, pred, average="macro"), 4))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, pred, zero_division=0))

print("=== CONFUSION MATRIX ===")
print(pd.DataFrame(
    confusion_matrix(y_test, pred, labels=model.classes_),
    index=model.classes_,
    columns=model.classes_
))

print("\n=== CROSS-VALIDATION (TRAINING DATA) ===")
cv = cross_validate(
    model,
    X_train,
    y_train,
    cv=5,
    scoring=["accuracy", "f1_macro"]
)
print("Mean Accuracy:", round(cv["test_accuracy"].mean(), 4))
print("Mean Macro F1:", round(cv["test_f1_macro"].mean(), 4))
print("Std Macro F1:", round(cv["test_f1_macro"].std(), 4))


# 6. Analisis feature importance
print("\n=== FEATURE IMPORTANCE ===")

rf = model.named_steps["classifier"]

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))
