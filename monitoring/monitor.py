import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from monitoring.alerts import send_alert
from configs.monitoring_config import (
    F1_THRESHOLD,
    ROC_AUC_THRESHOLD,
    DRIFT_SHARE_THRESHOLD,
    CANARY_MINIMUM_SAMPLES,
)

from src.config import CANARY_MODEL_VERSION

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

def load_canary_evaluation_data(
    current_data_path=CURRENT_DATA_PATH,
):
    current_data = pd.read_csv(current_data_path)

    labeled_data = current_data.dropna(
        subset=["Churn"]
    )

    if len(labeled_data) < CANARY_MINIMUM_SAMPLES:
        raise ValueError(
            f"Not enough labeled prediction data for canary evaluation. "
            f"Required {CANARY_MINIMUM_SAMPLES} rows, "
            f"found {len(labeled_data)}."
        )

    return labeled_data

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


def calculate_model_metrics_by_version(current_data):
    metrics_by_version = {}

    labeled_data = current_data.dropna(
        subset=["Churn"]
    )

    for model_version, model_data in labeled_data.groupby(
        "model_version",
        dropna=False,
    ):
        if model_data["Churn"].nunique() < 2:
            continue

        metrics_by_version[str(model_version)] = (
            calculate_model_metrics(model_data)
        )

    return metrics_by_version


def get_production_model_version():
    try:
        import mlflow

        production_model = (
            mlflow.MlflowClient().get_model_version_by_alias(
                "customer-churn-model",
                "production",
            )
        )

        return str(production_model.version)

    except Exception as exc:
        raise RuntimeError(
            "Unable to resolve the production model version "
            "from the MLflow production alias."
        ) from exc


def evaluate_canary(
    current_data,
    metrics_by_version,
):
    production_version = get_production_model_version()

    if production_version not in metrics_by_version:
        return {
            "ready": False,
            "reason": (
                f"Production model version {production_version} "
                "has no predictions in the monitoring window."
            ),
            "production_version": production_version,
        }

    if CANARY_MODEL_VERSION is None:
        return {
            "ready": False,
            "reason": "Canary model version is not configured.",
            "production_version": production_version,
        }

    canary_version = str(CANARY_MODEL_VERSION)

    if canary_version == production_version:
        return {
            "ready": False,
            "reason": (
                f"Configured canary version {canary_version} "
                "is already the production version."
            ),
            "model_version": canary_version,
            "production_version": production_version,
        }

    if canary_version not in metrics_by_version:
        return {
            "ready": False,
            "reason": (
                f"Configured canary model version {canary_version} "
                "has no labeled predictions."
            ),
            "model_version": canary_version,
            "production_version": production_version,
        }

    canary_data = current_data[
        current_data["model_version"].astype(str)
        == canary_version
    ]

    sample_count = len(canary_data)

    if sample_count < CANARY_MINIMUM_SAMPLES:
        return {
            "ready": False,
            "reason": (
                f"Canary has only {sample_count} labeled predictions; "
                f"{CANARY_MINIMUM_SAMPLES} are required."
            ),
            "model_version": canary_version,
            "sample_count": sample_count,
            "production_version": production_version,
        }

    canary_metrics = metrics_by_version[canary_version]
    production_metrics = metrics_by_version[production_version]

    f1_passes = canary_metrics["f1"] >= F1_THRESHOLD

    roc_auc_passes = (
        canary_metrics["roc_auc"] >= ROC_AUC_THRESHOLD
    )

    f1_comparison_passes = (
        canary_metrics["f1"] >= production_metrics["f1"]
    )

    roc_auc_comparison_passes = (
        canary_metrics["roc_auc"] >= production_metrics["roc_auc"]
    )

    if (
        not f1_passes
        or not roc_auc_passes
        or not f1_comparison_passes
        or not roc_auc_comparison_passes
    ):
        return {
            "ready": False,
            "reason": "Canary performance checks failed.",
            "model_version": canary_version,
            "sample_count": sample_count,
            "metrics": canary_metrics,
            "production_version": production_version,
            "production_metrics": production_metrics,
            "f1_passes": f1_passes,
            "roc_auc_passes": roc_auc_passes,
            "f1_comparison_passes": f1_comparison_passes,
            "roc_auc_comparison_passes": roc_auc_comparison_passes,
        }

    return {
        "ready": True,
        "reason": (
            "Canary passed minimum sample, absolute performance, "
            "and production comparison checks."
        ),
        "model_version": canary_version,
        "sample_count": sample_count,
        "metrics": canary_metrics,
        "production_version": production_version,
        "production_metrics": production_metrics,
        "f1_passes": f1_passes,
        "roc_auc_passes": roc_auc_passes,
        "f1_comparison_passes": f1_comparison_passes,
        "roc_auc_comparison_passes": roc_auc_comparison_passes,
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
        columns=[
            "timestamp",
            "churn_probability",
            "prediction",
            "model_version",
        ],
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

def write_canary_output(canary_ready):
    github_output = os.getenv("GITHUB_OUTPUT")

    if not github_output:
        return

    with open(
        github_output,
        "a",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            f"canary_ready="
            f"{'true' if canary_ready else 'false'}\n"
        )

def main():
    reference_data, current_data = load_monitoring_data()
    canary_data = load_canary_evaluation_data()

    labeled_current_data = current_data.dropna(
        subset=["Churn"]
    )

    if labeled_current_data.empty:
        raise ValueError(
            "No labeled prediction data available for model monitoring."
        )

    metrics = calculate_model_metrics(labeled_current_data)

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

    print("\nModel Performance by Version:")

    metrics_by_version = calculate_model_metrics_by_version(
        canary_data
    )

    for model_version, version_metrics in metrics_by_version.items():
        print(f"\nModel version: {model_version}")
        print(f"  Accuracy:  {version_metrics['accuracy']:.2%}")
        print(f"  Precision: {version_metrics['precision']:.2%}")
        print(f"  Recall:    {version_metrics['recall']:.2%}")
        print(f"  F1 Score:  {version_metrics['f1']:.2%}")
        print(f"  ROC-AUC:   {version_metrics['roc_auc']:.2%}")

    canary_evaluation = evaluate_canary(
        canary_data,
        metrics_by_version,
    )

    print("\nCanary Evaluation:")

    if canary_evaluation["ready"]:
        print(
            f"Canary v{canary_evaluation['model_version']} "
            f"is ready for deployment decision."
        )
        print(
            f"Canary samples: "
            f"{canary_evaluation['sample_count']}"
        )
        print(
            f"Canary F1: "
            f"{canary_evaluation['metrics']['f1']:.2%}"
        )
        print(
            f"Canary ROC-AUC: "
            f"{canary_evaluation['metrics']['roc_auc']:.2%}"
        )
        print(
            f"Production version: "
            f"{canary_evaluation['production_version']}"
        )
    else:
        print(
            f"Canary evaluation not ready: "
            f"{canary_evaluation['reason']}"
        )

    print(f"\nMonitoring window: {len(current_data)} predictions")
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
    write_canary_output(
        canary_evaluation["ready"]
    )

    drift_summary["result"].save_html(
        "monitoring/drift_report.html"
    )

    print("Data drift report generated successfully.")
    print("Open monitoring/drift_report.html to view the drift results.")


if __name__ == "__main__":
    main()