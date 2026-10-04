import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler


def create_preprocessing_pipeline():

    numerical_columns = [
        "Patient_Age",
        "Salary",
        "BMI",
        "Exercise_Time",
        "CheckUp_Year",
        "CheckUp_Month",
        "CheckUp_Day",
        "CheckUp_DayOfWeek"
    ]

    OneHot_columns = [
        "Hospital_ID",
        "Any_Existing_Disease"
    ]

    Ordinal_Columns = ["Alcohol"]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    OneHot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    Ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])

    preprocessing_pipeline = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("onehot", OneHot_pipeline, OneHot_columns),
        ("ordinal", Ordinal_pipeline, Ordinal_Columns)
    ])

    return preprocessing_pipeline