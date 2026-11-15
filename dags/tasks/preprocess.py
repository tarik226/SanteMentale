
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (StandardScaler,OneHotEncoder,OrdinalEncoder)


BASE_DIR = Path("/opt/airflow")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

INPUT_PATH = DATA_DIR / "prepared.csv"


NUMERICAL_VARIABLES = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate"
]


ONEHOT_VARIABLES = [
    "Gender",
    "Part_Time_Job",
    "Previous_Mental_Health_History"
]


ORDINAL_VARIABLES = [
    "Diet_Quality",
    "Physical_Activity"
]


def create_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer" , SimpleImputer(strategy="mean")),
        ("scaler" , StandardScaler())
    ])

    onehot_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("ordinal",OrdinalEncoder(
                categories=[
                    ["Poor", "Average", "Good"],["Low", "Moderate", "High"]]
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("numerical",numerical_pipeline,NUMERICAL_VARIABLES),
        ("onehot",onehot_pipeline,ONEHOT_VARIABLES),
        ("ordinal",ordinal_pipeline,ORDINAL_VARIABLES)
    ])

    return preprocessor


def run():

    print("Starting preprocessing...")

    df = pd.read_csv(INPUT_PATH)

    X = df[NUMERICAL_VARIABLES + ONEHOT_VARIABLES+ ORDINAL_VARIABLES]

    y = df["Mental_Health_Risk"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create and fit the preprocessor
    preprocessor = create_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)

    X_test_transformed = preprocessor.transform(X_test)

    # Retrieve transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    X_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns=feature_names
    )

    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save transformed datasets
    X_train_transformed.to_csv(DATA_DIR / "X_train.csv",index=False)

    X_test_transformed.to_csv(DATA_DIR / "X_test.csv",index=False)

    y_train.to_csv(DATA_DIR / "y_train.csv",index=False)

    y_test.to_csv(DATA_DIR / "y_test.csv",index=False)

    # Save fitted preprocessor
    joblib.dump(preprocessor,MODEL_DIR / "preprocessor.joblib")

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    run()
