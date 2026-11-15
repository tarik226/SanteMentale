from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
                             confusion_matrix)


BASE_DIR = Path("/opt/airflow")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"


def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),

        "precision_macro": float(precision_score(y_test,y_pred,average="macro",zero_division=0)),

        "recall_macro": float(recall_score(y_test,y_pred,average="macro",zero_division=0)),

        "f1_macro": float(f1_score(y_test,y_pred,average="macro",zero_division=0)),

        "confusion_matrix": confusion_matrix(y_test,y_pred).tolist(),
    }

    print(f"\n{model_name}")
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision_macro"])
    print("Recall:", results["recall_macro"])
    print("F1-score:", results["f1_macro"])
    print("Confusion Matrix:", results["f1_macro"])

    return results


def run():

    print("Starting model evaluation...")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    base_model = joblib.load(MODEL_DIR / "logistic_regression_base.joblib")

    rf_model = joblib.load(MODEL_DIR / "random_forest.joblib")

    xgb_model = joblib.load(MODEL_DIR / "xgboost_model.joblib")

    best_model = joblib.load(MODEL_DIR / "logistic_regression_best.joblib")

    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

    # Evaluate the models
    results = {}

    results["logistic_regression_base"] = evaluate_model(base_model,X_test,y_test,
        "Logistic Regression - Baseline"
    )

    results["random_forest"] = evaluate_model(rf_model,X_test,y_test,
        "Random Forest"
    )

    # XGBoost predictions need to be decoded
    xgb_predictions = label_encoder.inverse_transform(
        xgb_model.predict(X_test).astype(int)
    )

    results["xgboost"] = evaluate_model(
        type("XGBoostWrapper", (), {
            "predict": lambda self, X: xgb_predictions
        })(),
        X_test,
        y_test,
        "XGBoost"
    )

    results["logistic_regression_best"] = evaluate_model(best_model,X_test,y_test,
        "Logistic Regression - Optimized"
    )

    # Save evaluation results
    output_path = REPORT_DIR / "evaluation.json"

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\nEvaluation completed successfully.")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    run()
