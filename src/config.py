import os

from dotenv import load_dotenv


load_dotenv()


MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
CHURN_THRESHOLD_RAW = os.getenv("CHURN_THRESHOLD", "0.35")

API_KEY = os.getenv("API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
API_VERSION = os.getenv("API_VERSION", "1.0.0")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///C:/MLProjects/customer-churn-mlops/mlflow.db",
)

CANARY_ENABLED_RAW = os.getenv(
    "CANARY_ENABLED",
    "false",
)

CANARY_TRAFFIC_PERCENT_RAW = os.getenv(
    "CANARY_TRAFFIC_PERCENT",
    "0",
)

CANARY_MODEL_VERSION_RAW = os.getenv(
    "CANARY_MODEL_VERSION",
    "",
)


def validate_config():
    if MODEL_SOURCE not in {"local", "mlflow"}:
        raise ValueError(
            f"Invalid MODEL_SOURCE: '{MODEL_SOURCE}'. "
            "Must be 'local' or 'mlflow'."
        )

    try:
        churn_threshold = float(CHURN_THRESHOLD_RAW)
    except ValueError:
        raise ValueError(
            f"Invalid CHURN_THRESHOLD: '{CHURN_THRESHOLD_RAW}'. "
            "Must be a number between 0 and 1."
        )

    if not 0 <= churn_threshold <= 1:
        raise ValueError(
            f"Invalid CHURN_THRESHOLD: {churn_threshold}. "
            "Must be between 0 and 1."
        )

    if not API_KEY:
        raise ValueError("API_KEY must not be empty.")

    if not ADMIN_API_KEY:
        raise ValueError("ADMIN_API_KEY must not be empty.")

    if not RATE_LIMIT:
        raise ValueError("RATE_LIMIT must not be empty.")

    if not API_VERSION:
        raise ValueError("API_VERSION must not be empty.")

    if CANARY_ENABLED_RAW.lower() not in {"true", "false"}:
        raise ValueError(
            f"Invalid CANARY_ENABLED: '{CANARY_ENABLED_RAW}'. "
            "Must be 'true' or 'false'."
        )

    try:
        canary_traffic_percent = float(
            CANARY_TRAFFIC_PERCENT_RAW
        )
    except ValueError:
        raise ValueError(
            f"Invalid CANARY_TRAFFIC_PERCENT: "
            f"'{CANARY_TRAFFIC_PERCENT_RAW}'. "
            "Must be a number between 0 and 100."
        )

    if not 0 <= canary_traffic_percent <= 100:
        raise ValueError(
            f"Invalid CANARY_TRAFFIC_PERCENT: "
            f"{canary_traffic_percent}. "
            "Must be between 0 and 100."
        )

    if CANARY_ENABLED_RAW.lower() == "true":
        if not CANARY_MODEL_VERSION_RAW.strip():
            raise ValueError(
                "CANARY_MODEL_VERSION must not be empty "
                "when CANARY_ENABLED is true."
            )

        try:
            canary_model_version = int(
                CANARY_MODEL_VERSION_RAW
            )
        except ValueError:
            raise ValueError(
                f"Invalid CANARY_MODEL_VERSION: "
                f"'{CANARY_MODEL_VERSION_RAW}'. "
                "Must be a positive integer."
            )

        if canary_model_version <= 0:
            raise ValueError(
                f"Invalid CANARY_MODEL_VERSION: "
                f"{canary_model_version}. "
                "Must be a positive integer."
            )


validate_config()

CHURN_THRESHOLD = float(CHURN_THRESHOLD_RAW)

CANARY_ENABLED = CANARY_ENABLED_RAW.lower() == "true"

CANARY_TRAFFIC_PERCENT = float(
    CANARY_TRAFFIC_PERCENT_RAW
)

CANARY_MODEL_VERSION = (
    int(CANARY_MODEL_VERSION_RAW)
    if CANARY_MODEL_VERSION_RAW.strip()
    else None
)