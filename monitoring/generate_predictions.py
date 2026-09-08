import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_churn, _select_prediction_model


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


evaluation_mode = os.getenv(
    "CANARY_EVALUATION_MODE",
    "false",
).lower() == "true"


for _, row in sample_data.iterrows():
    customer_data = row.to_dict()

    # Remove columns that are not model inputs
    customer_data.pop("customerID", None)

    if evaluation_mode:
        selected_model = _select_prediction_model()

        if selected_model[3] != "canary":
            raise RuntimeError(
                "Canary evaluation mode expected canary routing, "
                f"but selected model role was '{selected_model[3]}'."
            )

    predict_churn(customer_data)


if evaluation_mode:
    print(
        f"{REQUIRED_PREDICTIONS} canary evaluation predictions generated successfully."
    )
else:
    print(
        f"{REQUIRED_PREDICTIONS} production predictions generated successfully."
    )