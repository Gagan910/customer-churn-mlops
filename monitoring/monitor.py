from alerts import send_alert
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

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

actual = current_data["Churn"].map({"No": 0, "Yes": 1})
predicted = current_data["prediction"]

print("\nModel Performance:")
print(f"Accuracy:  {accuracy_score(actual, predicted):.2%}")
print(f"Precision: {precision_score(actual, predicted):.2%}")
print(f"Recall:    {recall_score(actual, predicted):.2%}")
print(f"F1 Score:  {f1_score(actual, predicted):.2%}")
f1 = f1_score(actual, predicted)

F1_THRESHOLD = 0.50

if f1 < F1_THRESHOLD:
    send_alert(
        "Customer Churn Model Performance Alert",
        f"F1 score has dropped to {f1:.2%}, "
        f"below the threshold of {F1_THRESHOLD:.2%}."
    )

    raise ValueError(
        f"Model performance alert: F1 score {f1:.2%} "
        f"is below the threshold of {F1_THRESHOLD:.2%}"
    )

print(f"F1 threshold check passed: {f1:.2%} >= {F1_THRESHOLD:.2%}")


roc_auc = roc_auc_score(
    actual,
    current_data["churn_probability"]
)

ROC_AUC_THRESHOLD = 0.75

print(f"ROC-AUC:   {roc_auc:.2%}")

if roc_auc < ROC_AUC_THRESHOLD:
    send_alert(
        "Customer Churn Model Performance Alert",
        f"ROC-AUC has dropped to {roc_auc:.2%}, "
        f"below the threshold of {ROC_AUC_THRESHOLD:.2%}."
    )
    raise ValueError(
        f"Model performance alert: ROC-AUC {roc_auc:.2%} "
        f"is below the threshold of {ROC_AUC_THRESHOLD:.2%}"
    )

print(
    f"ROC-AUC threshold check passed: "
    f"{roc_auc:.2%} >= {ROC_AUC_THRESHOLD:.2%}"
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


# Check for data drift
result_data = result.dict()

drift_metrics = [
    metric
    for metric in result_data["metrics"]
    if metric["metric_name"].startswith("DriftedColumnsCount")
]

if drift_metrics:
    drift_summary = drift_metrics[0]["value"]

    drifted_columns = int(drift_summary["count"])
    drift_share = float(drift_summary["share"])

    DRIFT_SHARE_THRESHOLD = 0.50

    print(
        f"Drifted columns: {drifted_columns}"
    )
    print(
        f"Drift share: {drift_share:.2%}"
    )

    if drift_share >= DRIFT_SHARE_THRESHOLD:
        send_alert(
            "Customer Churn Data Drift Alert",
            f"Data drift detected across "
            f"{drifted_columns} columns "
            f"({drift_share:.2%} of monitored columns)."
        )

        raise ValueError(
            f"Data drift threshold exceeded: "
            f"{drift_share:.2%} >= "
            f"{DRIFT_SHARE_THRESHOLD:.2%}"
        )

    print(
        f"Data drift threshold check passed: "
        f"{drift_share:.2%} < "
        f"{DRIFT_SHARE_THRESHOLD:.2%}"
    )


result.save_html("monitoring/drift_report.html")

print("Data drift report generated successfully.")
print("Open monitoring/drift_report.html to view the drift results.")

