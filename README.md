
# SmartKids AI Nutrition

Educational Machine Learning project for classifying nutrition labels in a synthetic children's dataset using Random Forest.

> **Disclaimer:** This project is for educational purposes only. It is not a medical diagnostic tool. The dataset's original source and label-generation method have not been independently verified.

## Project Overview

SmartKids AI Nutrition demonstrates a basic machine learning workflow, from data preparation and validation to model training, evaluation, and prediction through a Streamlit web application.

## Features

- Data quality audit and validation
- ETL pipeline for dataset cleaning
- Random Forest classification
- Model evaluation using accuracy, macro F1-score, and confusion matrix
- Interactive Streamlit interface
- BMI calculation for informational display only

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit

## Dataset

The project uses `malnutrition_balanced_synthetic_dataset.csv`.

- 3,000 rows
- 5 original columns
- 3 labels: `Normal`, `Underweight`, and `Overweight`
- 1,000 rows per label

The dataset contains age, gender, weight, height, and nutrition labels. Its source and labeling methodology are not verified, so model results should not be interpreted as evidence of clinical validity.

## Data Pipeline

1. Load the raw CSV dataset.
2. Validate age, gender, weight, height, and target labels.
3. Remove duplicate records if present.
4. Calculate BMI as an informational feature in the cleaned output.
5. Save the cleaned dataset to `output/malnutrition_cleaned.csv`.
6. Validate the cleaned dataset.

The latest pipeline run processed 3,000 rows, rejected 0 rows, and removed 0 duplicates. All pipeline validation checks passed.

## Model Training

The model uses four input features:

- `Age_months`
- `Weight_kg`
- `Height_cm`
- `Gender`

Gender is numerically encoded before prediction. The training script uses a scikit-learn pipeline with median imputation and a Random Forest classifier.

The data is split into training and test sets using an 80:20 ratio with stratification. Grid search with 5-fold cross-validation selects model hyperparameters using macro F1-score.

### Latest Evaluation Results

| Metric | Result |
|---|---:|
| Test accuracy | 88.33% |
| Test macro F1-score | 88.37% |
| Best cross-validation macro F1-score | 87.13% |

These results measure agreement with the dataset labels only. They do not establish medical accuracy or suitability for real-world health decisions.

## Project Structure
```text
.
├── app_gizi.py
├── model.py
├── model_features.pkl
├── malnutrition_model_final.pkl
├── malnutrition_balanced_synthetic_dataset.csv
├── pipeline.py
├── validate_pipeline.py
├── audit_dataset.py
├── audit_patterns.py
├── audit_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Use a compatible Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app_gizi.py
```

## Retrain the Model

```bash
python model.py
```

Run the training script before launching the application if the model artifacts need to be regenerated.

## Limitations

- The dataset is synthetic and its labeling method has not been independently verified.
- Evaluation metrics reflect agreement with the provided labels.
- BMI alone is not sufficient to assess children's nutritional status.
- Predictions must not be used for diagnosis, treatment, or personalized dietary decisions.

## Project Purpose

This project demonstrates data cleaning, validation, machine learning model development, evaluation, and deployment of an educational prediction interface.
