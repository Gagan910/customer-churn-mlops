import importlib

import pytest


def test_invalid_churn_threshold(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "abc")

    import src.config

    with pytest.raises(ValueError, match="Must be a number between 0 and 1"):
        importlib.reload(src.config)


def test_churn_threshold_out_of_range(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "1.5")

    import src.config

    with pytest.raises(ValueError, match="Must be between 0 and 1"):
        importlib.reload(src.config)


def test_valid_churn_threshold(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.45")

    import src.config

    config = importlib.reload(src.config)

    assert config.CHURN_THRESHOLD == 0.45