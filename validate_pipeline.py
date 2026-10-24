
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "malnutrition_balanced_synthetic_dataset.csv"
CLEAN_FILE = BASE_DIR / "output" / "malnutrition_cleaned.csv"

raw = pd.read_csv(RAW_FILE)
clean = pd.read_csv(CLEAN_FILE)

print("=== PIPELINE VALIDATION ===")

# 1. Jumlah baris
print(f"Raw rows: {len(raw)}")
print(f"Clean rows: {len(clean)}")
assert len(raw) == len(clean), "Jumlah baris berbeda!"

# 2. Nilai kosong
missing = clean.isna().sum()
print("\nMissing values:")
print(missing.to_string())
assert clean.isna().sum().sum() == 0, "Ada nilai kosong!"

# 3. Kategori label
print("\nNutrition status:")
print(clean["Nutrition_Status"].value_counts().to_string())

expected_statuses = {"Normal", "Underweight", "Overweight"}
assert set(clean["Nutrition_Status"].unique()) == expected_statuses

# 4. Validasi BMI
expected_bmi = (
    clean["Weight_kg"] / (clean["Height_cm"] / 100) ** 2
).round(2)

bmi_match = (clean["BMI"] == expected_bmi).all()
print(f"\nBMI calculation correct: {bmi_match}")
assert bmi_match, "Perhitungan BMI tidak konsisten!"

# 5. Rentang nilai
print("\nNumeric ranges:")
print(clean[["Age_months", "Weight_kg", "Height_cm", "BMI"]].describe().round(2))

print("\nAll pipeline validation checks passed!")
