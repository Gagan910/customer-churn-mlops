import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from monitoring.alerts import send_alert
from configs.monitoring_config import (
    F1_THRESHOLD,
    ROC_AUC_THRESHOLD,
    DRIFT_SHARE_THRESHOLD,
)

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
MONITORING_WINDOW = 500


def load_monitoring_data(
    reference_data_path=REFERENCE_DATA_PATH,
    current_data_path=CURRENT_DATA_PATH,
):
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)

    current_data = current_data.tail(MONITORING_WINDOW)

    if len(current_data) < MONITORING_WINDOW:
        raise ValueError(
            f"Not enough prediction data for monitoring. "
            f"Expected {MONITORING_WINDOW} rows, found {len(current_data)}."
        )

    return reference_data, current_data


def calculate_model_metrics(current_data):
    actual = current_data["Churn"].map({"No": 0, "Yes": 1})
    predicted = current_data["prediction"]

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    roc_auc = roc_auc_score(
        actual,
        current_data["churn_probability"],
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def check_model_performance(metrics):
    f1 = metrics["f1"]
    roc_auc = metrics["roc_auc"]

    if f1 < F1_THRESHOLD:
        send_alert(
            "Customer Churn Model Performance Alert",
            f"F1 score has dropped to {f1:.2%}, "
            f"below the threshold of {F1_THRESHOLD:.2%}."
        )

        print(
            f"Model performance alert: F1 score {f1:.2%} "
            f"is below the threshold of {F1_THRESHOLD:.2%}"
        )

    if roc_auc < ROC_AUC_THRESHOLD:
        send_alert(
            "Customer Churn Model Performance Alert",
            f"ROC-AUC has dropped to {roc_auc:.2%}, "
            f"below the threshold of {ROC_AUC_THRESHOLD:.2%}."
        )

        print(
            f"Model performance alert: ROC-AUC {roc_auc:.2%} "
            f"is below the threshold of {ROC_AUC_THRESHOLD:.2%}"
        )


def prepare_drift_data(reference_data, current_data):
    current_data = current_data.drop(
        columns=["timestamp", "churn_probability", "prediction"],
        errors="ignore",
    )

    columns_to_exclude = ["customerID", "Churn"]

    reference_data = reference_data.drop(
        columns=columns_to_exclude,
        errors="ignore",
    )

    current_data = current_data.drop(
        columns=columns_to_exclude,
        errors="ignore",
    )

    return reference_data, current_data


def calculate_drift(reference_data, current_data):
    report = Report(
        [
            DataDriftPreset()
        ]
    )

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    result_data = result.dict()

    drift_metrics = [
        metric
        for metric in result_data["metrics"]
        if metric["metric_name"].startswith("DriftedColumnsCount")
    ]

    if not drift_metrics:
        return {
            "drifted_columns": 0,
            "drift_share": 0.0,
            "result": result,
        }

    drift_summary = drift_metrics[0]["value"]

    return {
        "drifted_columns": int(drift_summary["count"]),
        "drift_share": float(drift_summary["share"]),
        "result": result,
    }


def check_data_drift(drift_summary):
    drifted_columns = drift_summary["drifted_columns"]
    drift_share = drift_summary["drift_share"]

    if drift_share >= DRIFT_SHARE_THRESHOLD:
        send_alert(
            "Customer Churn Data Drift Alert",
            f"Data drift detected across "
            f"{drifted_columns} columns "
            f"({drift_share:.2%} of monitored columns)."
        )

        print(
            f"Data drift threshold exceeded: "
            f"{drift_share:.2%} >= "
            f"{DRIFT_SHARE_THRESHOLD:.2%}"
        )


def should_retrain(metrics, drift_summary):
    return (
        metrics["f1"] < F1_THRESHOLD
        or metrics["roc_auc"] < ROC_AUC_THRESHOLD
        or drift_summary["drift_share"] >= DRIFT_SHARE_THRESHOLD
    )


def write_retraining_output(retrain_required):
    github_output = os.getenv("GITHUB_OUTPUT")

    if not github_output:
        return

    with open(
        github_output,
        "a",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            f"retrain_required="
            f"{'true' if retrain_required else 'false'}\n"
        )


def main():
    reference_data, current_data = load_monitoring_data()

    metrics = calculate_model_metrics(current_data)

    print("\nModel Performance:")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1']:.2%}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.2%}")

    check_model_performance(metrics)

    print(
        f"F1 threshold check: "
        f"{metrics['f1']:.2%} >= {F1_THRESHOLD:.2%}"
    )

    print(
        f"ROC-AUC threshold check: "
        f"{metrics['roc_auc']:.2%} >= {ROC_AUC_THRESHOLD:.2%}"
    )

    print(f"Monitoring window: {len(current_data)} predictions")
    print(
        f"Predicted churn rate: "
        f"{current_data['prediction'].mean():.2%}"
    )
    print(
        f"Average churn probability: "
        f"{current_data['churn_probability'].mean():.2%}"
    )

    reference_drift_data, current_drift_data = prepare_drift_data(
        reference_data,
        current_data,
    )

    drift_summary = calculate_drift(
        reference_drift_data,
        current_drift_data,
    )

    print(
        f"Drifted columns: "
        f"{drift_summary['drifted_columns']}"
    )
    print(
        f"Drift share: "
        f"{drift_summary['drift_share']:.2%}"
    )

    check_data_drift(drift_summary)

    print(
        f"Data drift threshold check: "
        f"{drift_summary['drift_share']:.2%} < "
        f"{DRIFT_SHARE_THRESHOLD:.2%}"
    )

    retrain_required = should_retrain(
        metrics,
        drift_summary,
    )

    if retrain_required:
        print("RETRAIN_REQUIRED")
    else:
        print("RETRAIN_NOT_REQUIRED")

    write_retraining_output(retrain_required)

    drift_summary["result"].save_html(
        "monitoring/drift_report.html"
    )

    print("Data drift report generated successfully.")
    print("Open monitoring/drift_report.html to view the drift results.")


if __name__ == "__main__":
    main()
