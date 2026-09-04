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

if MODEL_SOURCE == "mlflow":
    import mlflow
    model = mlflow.xgboost.load_model(
        "models:/customer-churn-model@production"
    )
else:
    model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict_churn(customer_data):
    data = pd.DataFrame([customer_data])
    logger.info("Churn prediction request received")

    processed_data = preprocessor.transform(data)

    probability = model.predict_proba(processed_data)[0][1]
    prediction = int(probability >= CHURN_THRESHOLD)
    
    logger.info(
        "Churn prediction completed | model_source=%s | probability=%.4f | threshold=%.2f | prediction=%d",
        MODEL_SOURCE,
        probability,
        CHURN_THRESHOLD,
        prediction,
    )

    log_data = customer_data.copy()
    log_data["timestamp"] = datetime.now().isoformat()
    log_data["churn_probability"] = float(probability)
    log_data["prediction"] = prediction

    log_path = BASE_DIR / "data" / "processed" / "prediction_logs.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([log_data]).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False
    )

    return {
        "churn_probability": float(probability),
        "prediction": prediction
    }