import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_churn


INPUT_DATA_PATH = "data/processed/current_data.csv"


current_data = pd.read_csv(INPUT_DATA_PATH)


for _, row in current_data.sample(n=500).iterrows():
    customer_data = row.to_dict()

    # Remove columns that are not model inputs
    customer_data.pop("customerID", None)

    predict_churn(customer_data)


print("500 production predictions generated successfully.")