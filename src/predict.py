import joblib
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.config import MODEL_SOURCE, CHURN_THRESHOLD

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

def load_model():
    try:
        if MODEL_SOURCE == "mlflow":
            import mlflow

            loaded_model = mlflow.xgboost.load_model(
                "models:/customer-churn-model@production"
            )

            production_model = (
                mlflow.MlflowClient().get_model_version_by_alias(
                    "customer-churn-model",
                    "production",
                )
            )

            preprocessor_path = mlflow.artifacts.download_artifacts(
                run_id=production_model.run_id,
                artifact_path="model/preprocessor.pkl",
            )

            loaded_preprocessor = joblib.load(preprocessor_path)

        else:
            loaded_model = joblib.load(MODEL_PATH)
            loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)

        logger.info(
            "Model and preprocessor loaded successfully | model_source=%s",
            MODEL_SOURCE,
        )

        return loaded_model, loaded_preprocessor

    except Exception:
        logger.exception(
            "Failed to load model and preprocessor | model_source=%s",
            MODEL_SOURCE,
        )

        return None, None


model, preprocessor = load_model()

def reload_model():
    global model, preprocessor

    model, preprocessor = load_model()

    return model is not None and preprocessor is not None


def predict_churn(customer_data, request_id=None):
    if model is None or preprocessor is None:
        logger.error(
            "Prediction unavailable because model or preprocessor is not loaded | request_id=%s",
            request_id,
        )
        raise RuntimeError("Model is not available.")

    data = pd.DataFrame([customer_data])
    
    logger.info(
        "Churn prediction request received | request_id=%s",
        request_id,
    )

    processed_data = preprocessor.transform(data)

    probability = model.predict_proba(processed_data)[0][1]
    prediction = int(probability >= CHURN_THRESHOLD)
    
    logger.info(
        "Churn prediction completed | request_id=%s | model_source=%s | probability=%.4f | threshold=%.2f | prediction=%d",
        request_id,
        MODEL_SOURCE,
        probability,
        CHURN_THRESHOLD,
        prediction,
    )

    log_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
        "timestamp",
        "churn_probability",
        "prediction",
    ]

    log_data = {
        column: customer_data.get(column)
        for column in log_columns
    }

    log_data["timestamp"] = datetime.now().isoformat()
    log_data["churn_probability"] = float(probability)
    log_data["prediction"] = prediction

    log_path = BASE_DIR / "data" / "processed" / "prediction_logs.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([log_data], columns=log_columns).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )

    return {
        "churn_probability": float(probability),
        "prediction": prediction
    }