import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

CHURN_THRESHOLD = 0.35


def predict_churn(customer_data):
    data = pd.DataFrame([customer_data])

    processed_data = preprocessor.transform(data)

    probability = model.predict_proba(processed_data)[0][1]

    prediction = int(probability >= CHURN_THRESHOLD)

    return {
        "churn_probability": float(probability),
        "prediction": prediction
    }