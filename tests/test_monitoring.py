import pytest
import pandas as pd

from monitoring.monitor import (
    calculate_model_metrics,
    calculate_model_metrics_by_version,
    check_model_performance,
    check_data_drift,
    evaluate_canary,
    should_retrain,
    write_retraining_output,
    prepare_drift_data,
)


def test_should_retrain_for_low_f1():
    metrics = {
        "f1": 0.40,
        "roc_auc": 0.85,
    }

    drift_summary = {
        "drift_share": 0.05,
    }

    assert should_retrain(metrics, drift_summary) is True


def test_should_retrain_for_low_roc_auc():
    metrics = {
        "f1": 0.60,
        "roc_auc": 0.70,
    }

    drift_summary = {
        "drift_share": 0.05,
    }

    assert should_retrain(metrics, drift_summary) is True


def test_should_retrain_for_high_drift():
    metrics = {
        "f1": 0.60,
        "roc_auc": 0.85,
    }

    drift_summary = {
        "drift_share": 0.60,
    }

    assert should_retrain(metrics, drift_summary) is True


def test_should_not_retrain_when_all_checks_pass():
    metrics = {
        "f1": 0.60,
        "roc_auc": 0.85,
    }

    drift_summary = {
        "drift_share": 0.05,
    }

    assert should_retrain(metrics, drift_summary) is False


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


def test_calculate_model_metrics_by_version():
    current_data = pd.DataFrame(
        {
            "Churn": [
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
            ],
            "prediction": [
                0,
                1,
                0,
                1,
                0,
                0,
            ],
            "churn_probability": [
                0.10,
                0.90,
                0.20,
                0.80,
                0.30,
                0.40,
            ],
            "model_version": [
                "8",
                "8",
                "8",
                "9",
                "9",
                "9",
            ],
        }
    )

    metrics_by_version = calculate_model_metrics_by_version(
        current_data
    )

    assert set(metrics_by_version.keys()) == {"8", "9"}

    assert "f1" in metrics_by_version["8"]
    assert "roc_auc" in metrics_by_version["8"]

    assert "f1" in metrics_by_version["9"]
    assert "roc_auc" in metrics_by_version["9"]


def test_evaluate_canary_not_ready_below_minimum_samples(
    monkeypatch,
):
    monkeypatch.setattr(
        "monitoring.monitor.get_production_model_version",
        lambda: "9",
    )
    
    monkeypatch.setattr(
        "monitoring.monitor.CANARY_MODEL_VERSION",
        8,
    )

    current_data = pd.DataFrame(
        {
            "Churn": ["No", "Yes"] * 25,
            "prediction": [0, 1] * 25,
            "churn_probability": [0.10, 0.90] * 25,
            "model_version": ["8"] * 50,
        }
    )

    metrics_by_version = {
        "8": {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
        },
        "9": {
            "accuracy": 0.80,
            "precision": 0.80,
            "recall": 0.80,
            "f1": 0.80,
            "roc_auc": 0.85,
        },
    }

    result = evaluate_canary(
        current_data,
        metrics_by_version,
    )

    assert result["ready"] is False
    assert result["model_version"] == "8"
    assert result["sample_count"] == 50
    assert "100" in result["reason"]


def test_evaluate_canary_ready_when_thresholds_pass(
    monkeypatch,
):
    monkeypatch.setattr(
        "monitoring.monitor.get_production_model_version",
        lambda: "9",
    )
    
    monkeypatch.setattr(
        "monitoring.monitor.CANARY_MODEL_VERSION",
        8,
    )

    current_data = pd.DataFrame(
        {
            "Churn": ["No", "Yes"] * 50,
            "prediction": [0, 1] * 50,
            "churn_probability": [0.10, 0.90] * 50,
            "model_version": ["8"] * 100,
        }
    )

    metrics_by_version = {
        "8": {
            "accuracy": 0.90,
            "precision": 0.85,
            "recall": 0.85,
            "f1": 0.85,
            "roc_auc": 0.90,
        },
        "9": {
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.75,
            "f1": 0.75,
            "roc_auc": 0.80,
        },
    }

    result = evaluate_canary(
        current_data,
        metrics_by_version,
    )

    assert result["ready"] is True
    assert result["model_version"] == "8"
    assert result["sample_count"] == 100
    assert result["production_version"] == "9"
    assert result["f1_passes"] is True
    assert result["roc_auc_passes"] is True


def test_evaluate_canary_not_ready_when_threshold_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        "monitoring.monitor.get_production_model_version",
        lambda: "9",
    )

    monkeypatch.setattr(
        "monitoring.monitor.CANARY_MODEL_VERSION",
        8,
    )
    current_data = pd.DataFrame(
        {
            "Churn": ["No", "Yes"] * 50,
            "prediction": [0, 1] * 50,
            "churn_probability": [0.10, 0.90] * 50,
            "model_version": ["8"] * 100,
        }
    )

    metrics_by_version = {
        "8": {
            "accuracy": 0.70,
            "precision": 0.45,
            "recall": 0.45,
            "f1": 0.45,
            "roc_auc": 0.70,
        },
        "9": {
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.75,
            "f1": 0.75,
            "roc_auc": 0.80,
        },
    }

    result = evaluate_canary(
        current_data,
        metrics_by_version,
    )

    assert result["ready"] is False
    assert result["model_version"] == "8"
    assert result["sample_count"] == 100
    assert result["f1_passes"] is False
    assert result["roc_auc_passes"] is False
    assert "performance checks failed" in result["reason"]


def test_prepare_drift_data_excludes_metadata():
    reference_data = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "gender": ["Male", "Female"],
            "tenure": [10, 20],
            "Churn": ["No", "Yes"],
        }
    )

    current_data = pd.DataFrame(
        {
            "customerID": ["3", "4"],
            "gender": ["Male", "Female"],
            "tenure": [15, 25],
            "Churn": ["No", "Yes"],
            "timestamp": ["2026-09-06", "2026-09-06"],
            "churn_probability": [0.20, 0.80],
            "prediction": [0, 1],
            "model_version": ["local", "local"],
        }
    )

    reference_drift_data, current_drift_data = prepare_drift_data(
        reference_data,
        current_data,
    )

    assert "customerID" not in reference_drift_data.columns
    assert "Churn" not in reference_drift_data.columns

    assert "customerID" not in current_drift_data.columns
    assert "Churn" not in current_drift_data.columns
    assert "timestamp" not in current_drift_data.columns
    assert "churn_probability" not in current_drift_data.columns
    assert "prediction" not in current_drift_data.columns
    assert "model_version" not in current_drift_data.columns


def test_model_performance_passes():
    metrics = {
        "f1": 0.60,
        "roc_auc": 0.85,
    }

    check_model_performance(metrics)


def test_model_performance_f1_fails(capsys):
    metrics = {
        "f1": 0.40,
        "roc_auc": 0.85,
    }

    check_model_performance(metrics)

    captured = capsys.readouterr()

    assert "Model performance alert: F1 score" in captured.out


def test_model_performance_roc_auc_fails(capsys):
    metrics = {
        "f1": 0.60,
        "roc_auc": 0.70,
    }

    check_model_performance(metrics)

    captured = capsys.readouterr()

    assert "Model performance alert: ROC-AUC" in captured.out


def test_data_drift_passes():
    drift_summary = {
        "drifted_columns": 1,
        "drift_share": 0.05,
    }

    check_data_drift(drift_summary)


def test_data_drift_fails(capsys):
    drift_summary = {
        "drifted_columns": 10,
        "drift_share": 0.60,
    }

    check_data_drift(drift_summary)

    captured = capsys.readouterr()

    assert "Data drift threshold exceeded" in captured.out


def test_write_retraining_output_true(monkeypatch, tmp_path):
    output_file = tmp_path / "github_output.txt"

    monkeypatch.setenv(
        "GITHUB_OUTPUT",
        str(output_file),
    )

    write_retraining_output(True)

    assert output_file.read_text(
        encoding="utf-8"
    ) == "retrain_required=true\n"


def test_write_retraining_output_false(monkeypatch, tmp_path):
    output_file = tmp_path / "github_output.txt"

    monkeypatch.setenv(
        "GITHUB_OUTPUT",
        str(output_file),
    )

    write_retraining_output(False)

    assert output_file.read_text(
        encoding="utf-8"
    ) == "retrain_required=false\n"


def test_write_retraining_output_without_github(monkeypatch, tmp_path):
    monkeypatch.delenv(
        "GITHUB_OUTPUT",
        raising=False,
    )

    output_file = tmp_path / "github_output.txt"

    write_retraining_output(True)

    assert not output_file.exists()