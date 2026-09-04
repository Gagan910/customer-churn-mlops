import os

from dotenv import load_dotenv


load_dotenv()


MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.35"))

API_KEY = os.getenv("API_KEY")
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
API_VERSION = os.getenv("API_VERSION", "1.0.0")


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

    if not RATE_LIMIT:
        raise ValueError("RATE_LIMIT must not be empty.")

    if not API_VERSION:
        raise ValueError("API_VERSION must not be empty.")


validate_config()