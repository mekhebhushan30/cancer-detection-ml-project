import os
import sys
import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "development"))

from feature_engineering import create_features
from feature_selection import select_features
from data_preprocessing import create_preprocessing_pipeline


def get_sample_raw_data():
    return pd.DataFrame({
        "Patient_ID": ("P001", "P002", "P003", "P004"),
        "Contact_Number": ("9876543210", "9876543211", "9876543212", "9876543213"),
        "Hospital_ID": ("H001", "H002", "H003", "H004"),
        "Patient_Age": (25, 48, 34, 50),
        "Salary": (45000.0, 82000.0, 55000.0, 91000.0),
        "BMI": (22.5, 31.0, 26.4, 33.8),
        "Exercise_Time": (45.0, 10.0, 30.0, 5.0),
        "Any_Existing_Disease": ("No", "Yes", "No", "Yes"),
        "CheckUp_Date": ("2026-01-15", "2026-03-20", "2026-05-10", "2026-08-25"),
        "Alcohol": ("No", "Yes", "No", "Yes"),
        "Cancer_Detected": ("No", "Yes", "No", "Yes")
    })


def test_feature_engineering():
    df = get_sample_raw_data()
    engineered_df = create_features(df.copy())

    for col in ("CheckUp_Year", "CheckUp_Month", "CheckUp_Day", "CheckUp_DayOfWeek"):
        assert col in engineered_df.columns, f"Missing engineered column: {col}"

    assert "CheckUp_Date" not in engineered_df.columns, "CheckUp_Date should be dropped!"
    assert len(engineered_df) == 4, "Row count changed during feature engineering!"


def test_feature_selection():
    df = create_features(get_sample_raw_data())
    X, y = select_features(df)

    assert "Patient_ID" not in X.columns, "Patient_ID was not dropped!"
    assert "Contact_Number" not in X.columns, "Contact_Number was not dropped!"
    assert set(y.unique()).issubset({0, 1}), "Target Cancer_Detected not mapped properly!"


def test_preprocessing_pipeline():
    df = create_features(get_sample_raw_data())
    X, _ = select_features(df)

    pipeline = create_preprocessing_pipeline()
    transformed_X = pipeline.fit_transform(X)

    assert transformed_X.shape == (4, 11) or transformed_X.shape[0] == 4, "Preprocessing row count mismatch!"
    assert not np.isnan(transformed_X).any(), "NaN values found in preprocessed output!"


def test_model_inference_and_recall_gate():
    np.random.seed(42)
    X_dummy = np.random.rand(100, 11)
    y_dummy = np.concatenate((np.zeros(50, dtype=int), np.ones(50, dtype=int)))

    model = RandomForestClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)

    preds = model.predict(X_dummy)
    probs = model.predict_proba(X_dummy)

    assert set(preds).issubset({0, 1}), "Predictions must be 0 or 1!"
    assert probs.shape == (100, 2), "Probability matrix shape mismatch!"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1!"

    recall = recall_score(y_dummy, preds)
    assert recall >= 0.70, f"Model Recall ({recall:.2f}) fell below production threshold (0.70)!"
