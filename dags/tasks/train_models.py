
from pathlib import Path

import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


BASE_DIR = Path("/opt/airflow")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


def run():

    print("Starting model training...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load transformed datasets
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]

    # 1. Logistic Regression - Baseline
    base_model = LogisticRegression(max_iter=100)

    base_model.fit(X_train, y_train)

    joblib.dump(base_model,MODEL_DIR / "logistic_regression_base.joblib")

    print("Baseline Logistic Regression trained.")

    # 2. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100,random_state=42)

    rf_model.fit(X_train, y_train)

    joblib.dump(rf_model,MODEL_DIR / "random_forest.joblib")

    print("Random Forest trained.")

    # 3. XGBoost
    label_encoder = LabelEncoder()

    y_train_xgb = label_encoder.fit_transform(y_train)

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective="multi:softmax",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss"
    )

    xgb_model.fit(X_train, y_train_xgb)

    joblib.dump(xgb_model,MODEL_DIR / "xgboost_model.joblib"
    )

    joblib.dump(label_encoder,MODEL_DIR / "label_encoder.joblib")

    print("XGBoost trained.")

    # 4. Logistic Regression - GridSearchCV

    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"]
    }

    grid_search = GridSearchCV(
        estimator=LogisticRegression(
            random_state=42,
            max_iter=2000
        ),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    joblib.dump(best_model,MODEL_DIR / "logistic_regression_best.joblib")

    print("Optimized Logistic Regression trained.")

    print("Best CV score:", grid_search.best_score_)
    print("Best parameters:", grid_search.best_params_)

    print("All models saved successfully.")


if __name__ == "__main__":
    run()
