import os
from dotenv import load_dotenv

load_dotenv()

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.35"))

def validate_config():
    if MODEL_SOURCE not in {"local", "mlflow"}:
        raise ValueError(
            f"Invalid MODEL_SOURCE: '{MODEL_SOURCE}'. "
            "Must be 'local' or 'mlflow'."
        )

    if not 0 <= CHURN_THRESHOLD <= 1:
        raise ValueError(
            f"Invalid CHURN_THRESHOLD: {CHURN_THRESHOLD}. "
            "Must be between 0 and 1."
        )
        
validate_config()