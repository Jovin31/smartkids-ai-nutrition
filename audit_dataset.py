
import pandas as pd

FILE = "malnutrition_balanced_synthetic_dataset.csv"

df = pd.read_csv(FILE)

print("=" * 50)
print("1. UKURAN DATASET")
print("=" * 50)
print(f"Jumlah baris : {len(df)}")
print(f"Jumlah kolom : {len(df.columns)}")
print(f"Nama kolom   : {df.columns.tolist()}")

print("\n" + "=" * 50)
print("2. CEK NULL / DATA KOSONG")
print("=" * 50)
null_count = df.isnull().sum()
null_result = pd.DataFrame({
    "Jumlah_Null": null_count,
    "Persentase_Null": (null_count / len(df) * 100).round(2)
})
print(null_result.to_string())

print(f"\nTotal sel kosong: {df.isnull().sum().sum()}")

print("\n" + "=" * 50)
print("3. CEK DUPLIKAT")
print("=" * 50)
print(f"Jumlah baris duplikat: {df.duplicated().sum()}")

print("\n" + "=" * 50)
print("4. DISTRIBUSI LABEL")
print("=" * 50)
if "Nutrition_Status" in df.columns:
    print(df["Nutrition_Status"].value_counts(dropna=False).to_string())

print("\n" + "=" * 50)
print("5. RINGKASAN KOLOM NUMERIK")
print("=" * 50)
print(df.describe().round(2).to_string())

print("\n" + "=" * 50)
print("6. CEK NILAI KATEGORI")
print("=" * 50)
for col in ["Gender", "Nutrition_Status"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_string())

print("\nAudit selesai.")
