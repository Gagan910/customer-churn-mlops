import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_DATA_PATH = "data/processed/reference_data.csv"
CURRENT_DATA_PATH = "data/processed/prediction_logs.csv"


reference_data = pd.read_csv(REFERENCE_DATA_PATH)
current_data = pd.read_csv(CURRENT_DATA_PATH)


current_data = current_data.tail(500)

if len(current_data) < 500:
    raise ValueError(
        f"Not enough prediction data for monitoring. "
        f"Expected 500 rows, found {len(current_data)}."
    )


# Prediction monitoring summary
print(f"Monitoring window: {len(current_data)} predictions")
print(f"Predicted churn rate: {current_data['prediction'].mean():.2%}")
print(
    f"Average churn probability: "
    f"{current_data['churn_probability'].mean():.2%}"
)


# Remove prediction-specific columns from drift monitoring
current_data = current_data.drop(
    columns=["timestamp", "churn_probability", "prediction"],
    errors="ignore"
)


# Remove identifier and target columns from drift monitoring
columns_to_exclude = ["customerID", "Churn"]

reference_data = reference_data.drop(
    columns=columns_to_exclude,
    errors="ignore"
)

current_data = current_data.drop(
    columns=columns_to_exclude,
    errors="ignore"
)


# Generate data drift report
report = Report(
    [
        DataDriftPreset()
    ]
)


result = report.run(
    reference_data=reference_data,
    current_data=current_data
)


result.save_html("monitoring/drift_report.html")

print("Data drift report generated successfully.")
print("Open monitoring/drift_report.html to view the drift results.")