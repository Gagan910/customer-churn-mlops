from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import CHURN_THRESHOLD


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


# --------------------------------------------------
# Model promotion
# --------------------------------------------------

def promote_model_to_production(model_name, model_version):
    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        model_name,
        "production",
        str(model_version),
    )

    print(
        f"Model {model_name} version {model_version} "
        "promoted to production."
    )


def passes_production_quality_gate(roc_auc, minimum_roc_auc=0.80):
    return roc_auc >= minimum_roc_auc


# --------------------------------------------------
# Training pipeline
# --------------------------------------------------

def train_model():

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # Target
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Remove unnecessary / EDA-only columns
    drop_columns = ["customerID", "TenureGroup"]

    for column in drop_columns:
        if column in df.columns:
            df = df.drop(columns=column)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # --------------------------------------------------
    # Train-test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    numerical_cols = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    categorical_cols = [
        column for column in X.columns
        if column not in numerical_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # --------------------------------------------------
    # MLflow experiment
    # --------------------------------------------------

    mlflow.set_experiment("customer-churn-prediction")

    with mlflow.start_run(run_name="tuned_xgboost"):

        # --------------------------------------------------
        # XGBoost + Hyperparameter Tuning
        # --------------------------------------------------

        model = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
        )

        grid_search.fit(X_train_processed, y_train)

        best_model = grid_search.best_estimator_

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        y_probability = best_model.predict_proba(
            X_test_processed
        )[:, 1]

        y_pred = (
            y_probability >= CHURN_THRESHOLD
        ).astype(int)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_probability)

        # --------------------------------------------------
        # Production quality gate
        # --------------------------------------------------

        if not passes_production_quality_gate(roc_auc):
            raise ValueError(
                f"Model failed production quality gate: "
                f"ROC-AUC={roc_auc:.4f}, required>=0.80"
            )

        # --------------------------------------------------
        # Log parameters
        # --------------------------------------------------

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param(
            "churn_threshold",
            CHURN_THRESHOLD,
        )

        # --------------------------------------------------
        # Log metrics
        # --------------------------------------------------

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # --------------------------------------------------
        # Save model and preprocessor
        # --------------------------------------------------

        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

        # --------------------------------------------------
        # Log artifacts
        # --------------------------------------------------

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(PREPROCESSOR_PATH),
            artifact_path="model",
        )

        # --------------------------------------------------
        # Register model
        # --------------------------------------------------

        model_info = mlflow.xgboost.log_model(
            best_model,
            name="xgboost-model",
            registered_model_name="customer-churn-model",
        )

        registered_version = model_info.registered_model_version

        # --------------------------------------------------
        # Promote to production
        # --------------------------------------------------

        promote_model_to_production(
            "customer-churn-model",
            registered_version,
        )

        print("Training completed successfully.")
        print("Best parameters:", grid_search.best_params_)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")


# --------------------------------------------------
# Explicit script entry point
# --------------------------------------------------

if __name__ == "__main__":
    train_model()