
import pandas as pd

df = pd.read_csv("malnutrition_balanced_synthetic_dataset.csv")

# Ringkasan statistik per label
print("=== RINGKASAN PER LABEL ===")
print(
    df.groupby("Nutrition_Status")[
        ["Age_months", "Weight_kg", "Height_cm"]
    ].agg(["mean", "min", "max"]).round(2).to_string()
)

# BMI untuk analisis pola saja, bukan diagnosis
df["BMI"] = df["Weight_kg"] / (df["Height_cm"] / 100) ** 2

print("\n=== BMI PER LABEL ===")
print(
    df.groupby("Nutrition_Status")["BMI"]
    .agg(["mean", "std", "min", "median", "max"])
    .round(2)
    .to_string()
)

# Korelasi fitur numerik
print("\n=== KORELASI FITUR NUMERIK ===")
print(
    df[["Age_months", "Weight_kg", "Height_cm", "BMI"]]
    .corr()
    .round(2)
    .to_string()
)

# Contoh data dengan BMI paling tinggi dan paling rendah
print("\n=== 10 BMI TERENDAH ===")
print(
    df.nsmallest(10, "BMI")[
        ["Age_months", "Gender", "Weight_kg",
         "Height_cm", "Nutrition_Status", "BMI"]
    ].round(2).to_string(index=False)
)

print("\n=== 10 BMI TERTINGGI ===")
print(
    df.nlargest(10, "BMI")[
        ["Age_months", "Gender", "Weight_kg",
         "Height_cm", "Nutrition_Status", "BMI"]
    ].round(2).to_string(index=False)
)
