import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_churn


INPUT_DATA_PATH = "data/processed/current_data.csv"
REQUIRED_PREDICTIONS = 500


current_data = pd.read_csv(INPUT_DATA_PATH)


if len(current_data) < REQUIRED_PREDICTIONS:
    raise ValueError(
        f"Not enough current data for monitoring. "
        f"Required {REQUIRED_PREDICTIONS} rows, found {len(current_data)}."
    )


sample_data = current_data.sample(
    n=REQUIRED_PREDICTIONS,
    random_state=42,
)


for _, row in sample_data.iterrows():
    customer_data = row.to_dict()

    # Remove columns that are not model inputs
    customer_data.pop("customerID", None)

    predict_churn(customer_data)


print(
    f"{REQUIRED_PREDICTIONS} production predictions generated successfully."
)