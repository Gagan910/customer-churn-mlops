import joblib
import logging
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")

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

CHURN_THRESHOLD = 0.35


def predict_churn(customer_data):
    data = pd.DataFrame([customer_data])
    logger.info("Churn prediction request received")

    processed_data = preprocessor.transform(data)

    probability = model.predict_proba(processed_data)[0][1]

    prediction = int(probability >= CHURN_THRESHOLD)

    log_data = customer_data.copy()
    log_data["timestamp"] = datetime.now().isoformat()
    log_data["churn_probability"] = float(probability)
    log_data["prediction"] = prediction

    log_path = BASE_DIR / "data" / "processed" / "prediction_logs.csv"

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