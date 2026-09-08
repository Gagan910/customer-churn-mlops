import joblib
import logging
import random
import pandas as pd

from datetime import datetime
from pathlib import Path

from src.config import (
    MODEL_SOURCE,
    CHURN_THRESHOLD,
    CANARY_ENABLED,
    CANARY_TRAFFIC_PERCENT,
    CANARY_MODEL_VERSION,
    MLFLOW_TRACKING_URI,
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


model_version = None

canary_model = None
canary_preprocessor = None
canary_model_version = None


def _configure_mlflow():
    import mlflow

    if hasattr(mlflow, "set_tracking_uri"):
        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        logger.info(
            "MLflow configured | tracking_uri=%s",
            MLFLOW_TRACKING_URI,
        )

    return mlflow


def load_model():
    try:
        loaded_model_version = "local"

        if MODEL_SOURCE == "mlflow":
            mlflow = _configure_mlflow()

            client = mlflow.MlflowClient()

            production_model = (
                client.get_model_version_by_alias(
                    "customer-churn-model",
                    "production",
                )
            )

            loaded_model_version = str(
                production_model.version
            )

            model_uri = (
                f"models:/customer-churn-model/"
                f"{loaded_model_version}"
            )

            loaded_model = mlflow.xgboost.load_model(
                model_uri
            )

            preprocessor_path = (
                mlflow.artifacts.download_artifacts(
                    run_id=production_model.run_id,
                    artifact_path="model/preprocessor.pkl",
                )
            )

            loaded_preprocessor = joblib.load(
                preprocessor_path
            )

        else:
            loaded_model = joblib.load(
                MODEL_PATH
            )

            loaded_preprocessor = joblib.load(
                PREPROCESSOR_PATH
            )

        logger.info(
            "Model and preprocessor loaded successfully | "
            "model_source=%s | model_version=%s",
            MODEL_SOURCE,
            loaded_model_version,
        )

        return (
            loaded_model,
            loaded_preprocessor,
            loaded_model_version,
        )

    except Exception:
        logger.exception(
            "Failed to load model and preprocessor | "
            "model_source=%s",
            MODEL_SOURCE,
        )

        return None, None, None


def load_canary_model():
    if not CANARY_ENABLED:
        logger.info(
            "Canary model loading skipped | canary_enabled=false"
        )
        return None, None, None

    if CANARY_MODEL_VERSION is None:
        logger.error(
            "Canary model loading failed | "
            "CANARY_MODEL_VERSION is not configured"
        )
        return None, None, None

    try:
        if MODEL_SOURCE != "mlflow":
            raise ValueError(
                "Canary model requires MODEL_SOURCE=mlflow."
            )

        mlflow = _configure_mlflow()

        model_uri = (
            f"models:/customer-churn-model/"
            f"{CANARY_MODEL_VERSION}"
        )

        loaded_canary_model = mlflow.xgboost.load_model(
            model_uri
        )

        client = mlflow.MlflowClient()

        canary_model_info = client.get_model_version(
            "customer-churn-model",
            str(CANARY_MODEL_VERSION),
        )

        preprocessor_path = (
            mlflow.artifacts.download_artifacts(
                run_id=canary_model_info.run_id,
                artifact_path="model/preprocessor.pkl",
            )
        )

        loaded_canary_preprocessor = joblib.load(
            preprocessor_path
        )

        loaded_canary_version = str(
            canary_model_info.version
        )

        logger.info(
            "Canary model loaded successfully | "
            "model_source=%s | model_version=%s",
            MODEL_SOURCE,
            loaded_canary_version,
        )

        return (
            loaded_canary_model,
            loaded_canary_preprocessor,
            loaded_canary_version,
        )

    except Exception as exc:
        logger.exception(
            "Failed to load canary model | "
            "model_source=%s | model_version=%s | "
            "error_type=%s | error=%s",
            MODEL_SOURCE,
            CANARY_MODEL_VERSION,
            type(exc).__name__,
            str(exc),
        )

        return None, None, None


model, preprocessor, model_version = load_model()

(
    canary_model,
    canary_preprocessor,
    canary_model_version,
) = load_canary_model()


def reload_model():
    global model
    global preprocessor
    global model_version
    global canary_model
    global canary_preprocessor
    global canary_model_version

    model, preprocessor, model_version = load_model()

    (
        canary_model,
        canary_preprocessor,
        canary_model_version,
    ) = load_canary_model()

    return (
        model is not None
        and preprocessor is not None
    )


def _select_prediction_model():
    if (
        CANARY_ENABLED
        and CANARY_TRAFFIC_PERCENT > 0
        and canary_model is not None
        and canary_preprocessor is not None
        and canary_model_version is not None
    ):
        canary_roll = random.uniform(
            0,
            100,
        )

        if canary_roll < CANARY_TRAFFIC_PERCENT:
            logger.info(
                "Canary model selected | "
                "traffic_percent=%.2f | roll=%.4f | "
                "model_version=%s",
                CANARY_TRAFFIC_PERCENT,
                canary_roll,
                canary_model_version,
            )

            return (
                canary_model,
                canary_preprocessor,
                canary_model_version,
                "canary",
            )

        logger.info(
            "Production model selected | "
            "canary_traffic_percent=%.2f | roll=%.4f | "
            "model_version=%s",
            CANARY_TRAFFIC_PERCENT,
            canary_roll,
            model_version,
        )

    elif (
        CANARY_ENABLED
        and CANARY_TRAFFIC_PERCENT > 0
    ):
        logger.warning(
            "Canary traffic requested but canary model "
            "is unavailable | falling back to production | "
            "canary_model_version=%s",
            CANARY_MODEL_VERSION,
        )

    return (
        model,
        preprocessor,
        model_version,
        "production",
    )


def predict_churn(
    customer_data,
    request_id=None,
):
    (
        selected_model,
        selected_preprocessor,
        selected_model_version,
        selected_model_role,
    ) = _select_prediction_model()

    if (
        selected_model is None
        or selected_preprocessor is None
    ):
        logger.error(
            "Prediction unavailable because model or "
            "preprocessor is not loaded | "
            "request_id=%s | model_role=%s",
            request_id,
            selected_model_role,
        )
        raise RuntimeError(
            "Model is not available."
        )

    data = pd.DataFrame(
        [customer_data]
    )

    logger.info(
        "Churn prediction request received | "
        "request_id=%s | model_role=%s | "
        "model_version=%s",
        request_id,
        selected_model_role,
        selected_model_version,
    )

    processed_data = (
        selected_preprocessor.transform(data)
    )

    probability = (
        selected_model.predict_proba(
            processed_data
        )[0][1]
    )

    prediction = int(
        probability >= CHURN_THRESHOLD
    )

    logger.info(
        "Churn prediction completed | "
        "request_id=%s | model_source=%s | "
        "model_role=%s | model_version=%s | "
        "probability=%.4f | threshold=%.2f | "
        "prediction=%d",
        request_id,
        MODEL_SOURCE,
        selected_model_role,
        selected_model_version,
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
        "model_version",
    ]

    log_data = {
        column: customer_data.get(column)
        for column in log_columns
    }

    log_data["timestamp"] = (
        datetime.now().isoformat()
    )

    log_data["churn_probability"] = float(
        probability
    )

    log_data["prediction"] = prediction

    log_data["model_version"] = (
        selected_model_version
    )

    log_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "prediction_logs.csv"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [log_data],
        columns=log_columns,
    ).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )

    return {
        "churn_probability": float(
            probability
        ),
        "prediction": prediction,
    }