import pytest
import pandas as pd

from monitoring.monitor import (
    calculate_model_metrics,
    check_model_performance,
    check_data_drift,
)


def test_calculate_model_metrics():
    current_data = pd.DataFrame(
        {
            "Churn": ["No", "Yes", "Yes", "No"],
            "prediction": [0, 1, 0, 0],
            "churn_probability": [0.10, 0.90, 0.40, 0.20],
        }
    )

    metrics = calculate_model_metrics(current_data)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(0.6666666667)


def test_model_performance_passes():
    metrics = {
        "f1": 0.60,
        "roc_auc": 0.85,
    }

    check_model_performance(metrics)


def test_model_performance_f1_fails():
    metrics = {
        "f1": 0.40,
        "roc_auc": 0.85,
    }

    with pytest.raises(
        ValueError,
        match="Model performance alert",
    ):
        check_model_performance(metrics)


def test_model_performance_roc_auc_fails():
    metrics = {
        "f1": 0.60,
        "roc_auc": 0.70,
    }

    with pytest.raises(
        ValueError,
        match="Model performance alert",
    ):
        check_model_performance(metrics)


def test_data_drift_passes():
    drift_summary = {
        "drifted_columns": 1,
        "drift_share": 0.05,
    }

    check_data_drift(drift_summary)


def test_data_drift_fails():
    drift_summary = {
        "drifted_columns": 10,
        "drift_share": 0.60,
    }

    with pytest.raises(
        ValueError,
        match="Data drift threshold exceeded",
    ):
        check_data_drift(drift_summary)