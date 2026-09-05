import pytest
import pandas as pd

from monitoring.monitor import (
    calculate_model_metrics,
    check_model_performance,
    check_data_drift,
    should_retrain,
    write_retraining_output,
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