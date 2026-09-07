import importlib

import pytest

import src.config


def test_invalid_churn_threshold(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "abc")
    monkeypatch.setenv("CANARY_ENABLED", "false")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "0")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "")

    with pytest.raises(
        ValueError,
        match="Must be a number between 0 and 1",
    ):
        importlib.reload(src.config)


def test_churn_threshold_out_of_range(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "1.5")
    monkeypatch.setenv("CANARY_ENABLED", "false")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "0")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "")

    with pytest.raises(
        ValueError,
        match="Must be between 0 and 1",
    ):
        importlib.reload(src.config)


def test_valid_churn_threshold(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.45")
    monkeypatch.setenv("CANARY_ENABLED", "false")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "0")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "")

    config = importlib.reload(src.config)

    assert config.CHURN_THRESHOLD == 0.45


def test_canary_defaults(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.35")
    monkeypatch.setenv("CANARY_ENABLED", "false")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "0")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "")

    config = importlib.reload(src.config)

    assert config.CANARY_ENABLED is False
    assert config.CANARY_TRAFFIC_PERCENT == 0
    assert config.CANARY_MODEL_VERSION is None


def test_canary_configuration(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.35")
    monkeypatch.setenv("CANARY_ENABLED", "true")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "10")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "10")

    config = importlib.reload(src.config)

    assert config.CANARY_ENABLED is True
    assert config.CANARY_TRAFFIC_PERCENT == 10
    assert config.CANARY_MODEL_VERSION == 10


def test_invalid_canary_enabled(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.35")
    monkeypatch.setenv("CANARY_ENABLED", "yes")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "0")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "")

    with pytest.raises(
        ValueError,
        match="Must be 'true' or 'false'",
    ):
        importlib.reload(src.config)


def test_canary_traffic_percent_out_of_range(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.35")
    monkeypatch.setenv("CANARY_ENABLED", "false")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "101")
    

    with pytest.raises(
        ValueError,
        match="Must be between 0 and 100",
    ):
        importlib.reload(src.config)


def test_canary_model_version_required_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.35")
    monkeypatch.setenv("CANARY_ENABLED", "true")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "10")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "")

    with pytest.raises(
        ValueError,
        match="CANARY_MODEL_VERSION must not be empty",
    ):
        importlib.reload(src.config)


def test_invalid_canary_model_version(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.35")
    monkeypatch.setenv("CANARY_ENABLED", "true")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "10")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "abc")

    with pytest.raises(
        ValueError,
        match="Must be a positive integer",
    ):
        importlib.reload(src.config)


def test_canary_model_version_must_be_positive(
    monkeypatch,
):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.35")
    monkeypatch.setenv("CANARY_ENABLED", "true")
    monkeypatch.setenv("CANARY_TRAFFIC_PERCENT", "10")
    monkeypatch.setenv("CANARY_MODEL_VERSION", "0")

    with pytest.raises(
        ValueError,
        match="Must be a positive integer",
    ):
        importlib.reload(src.config)