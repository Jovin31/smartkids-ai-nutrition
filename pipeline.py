
from pathlib import Path
import pandas as pd

# =========================
# CONFIGURATION
# =========================
BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "malnutrition_balanced_synthetic_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "malnutrition_cleaned.csv"

REQUIRED_COLUMNS = [
    "Age_months",
    "Gender",
    "Weight_kg",
    "Height_cm",
    "Nutrition_Status",
]

VALID_GENDERS = {"Male", "Female"}
VALID_STATUSES = {"Normal", "Underweight", "Overweight"}


def extract_data():
    """Extract data from the source CSV."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    df = pd.read_csv(SOURCE_FILE, sep=";")
    print(f"[EXTRACT] Rows loaded: {len(df)}")
    return df


def transform_data(df):
    """Validate, clean, and derive BMI."""
    original_count = len(df)

    # Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only the required columns and standardize text
    df = df[REQUIRED_COLUMNS].copy()
    df["Gender"] = df["Gender"].astype("string").str.strip()
    df["Nutrition_Status"] = (
        df["Nutrition_Status"].astype("string").str.strip()
    )

    # Convert numeric columns safely
    numeric_columns = ["Age_months", "Weight_kg", "Height_cm"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mark invalid rows
    valid = (
        df[numeric_columns].notna().all(axis=1)
        & df["Age_months"].between(12, 60)
        & (df["Weight_kg"] > 0)
        & (df["Height_cm"] > 0)
        & df["Gender"].isin(VALID_GENDERS)
        & df["Nutrition_Status"].isin(VALID_STATUSES)
    )

    rejected_count = int((~valid).sum())
    clean = df.loc[valid].copy()

    # Remove exact duplicate rows
    before_dedup = len(clean)
    clean = clean.drop_duplicates()
    duplicates_removed = before_dedup - len(clean)

    # Derived feature for analysis only; not a clinical diagnosis
    clean["BMI"] = (
        clean["Weight_kg"] / (clean["Height_cm"] / 100) ** 2
    ).round(2)

    # Stable column order
    clean = clean[REQUIRED_COLUMNS + ["BMI"]]

    report = {
        "input_rows": original_count,
        "rejected_rows": rejected_count,
        "duplicates_removed": duplicates_removed,
        "output_rows": len(clean),
    }

    return clean, report


def load_data(df):
    """Load the transformed dataset into an output CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[LOAD] Cleaned dataset saved to: {OUTPUT_FILE}")


def main():
    print("=" * 50)
    print("SMARTKIDS DATA PIPELINE")
    print("=" * 50)

    raw_data = extract_data()
    clean_data, report = transform_data(raw_data)
    load_data(clean_data)

    print("\n=== PIPELINE REPORT ===")
    for metric, value in report.items():
        print(f"{metric}: {value}")

    print("\n=== OUTPUT PREVIEW ===")
    print(clean_data.head(5).to_string(index=False))
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()

