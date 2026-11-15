
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/opt/airflow")
DATA_DIR = BASE_DIR / "data"
DATA_PATH = BASE_DIR / "data" / "student_mental_health.csv"

INPUT_PATH = BASE_DIR / "student_mental_health.csv"
OUTPUT_PATH = DATA_DIR / "prepared.csv"


if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

def fill_null(df):
    df = df.copy()

    numerical_columns = ["Sleep_Hours","GPA","Social_Media_Hours","Financial_Stress","Family_Support",
                "Attendance_Rate"]

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].mean())

    df["Diet_Quality"] = df["Diet_Quality"].fillna(
        df["Diet_Quality"].mode().iloc[0]
    )

    return df


def correct_ranges(df):
    """Remove records containing invalid values."""

    df = df[df["Age"].between(18, 30)]

    df = df[df["Gender"].isin(["Male", "Female", "Non Binary"])]

    df = df[df["Year_of_Study"].between(1, 5)]

    df = df[df["Sleep_Hours"].between(3.5, 9.5)]

    df = df[df["Study_Hours_Per_Week"].between(5, 55)]

    df = df[df["GPA"].between(1.5, 4.0)]

    df = df[df["Social_Media_Hours"].between(0.5, 10.0)]

    df = df[df["Screen_Time_Hours"].between(1.0, 14.0)]

    df = df[df["Physical_Activity"].isin(["Low", "Moderate", "High"])]

    df = df[df["Financial_Stress"].between(1.0, 10.0)]

    df = df[df["Family_Support"].between(1.0, 10.0)]

    df = df[df["Attendance_Rate"].between(30, 100)]

    df = df[df["Part_Time_Job"].isin(["Yes", "No"])]

    df = df[df["Diet_Quality"].isin(["Poor", "Average", "Good"])]

    df = df[df["Previous_Mental_Health_History"].isin(["Yes", "No"])]

    return df.copy()


def run():

    print("Starting data preparation...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # df = fill_null(df)

    df = correct_ranges(df)

    print("Prepared dataset shape:", df.shape)

    print("\nTarget distribution:")
    print(df["Mental_Health_Risk"].value_counts())

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("Prepared dataset saved:", OUTPUT_PATH)


if __name__ == "__main__":
    run()
