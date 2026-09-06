from src.train import (
    get_production_roc_auc,
    passes_production_comparison_gate,
    passes_production_quality_gate,
)


def test_quality_gate_passes_for_good_model():
    assert passes_production_quality_gate(0.85) is True


def test_quality_gate_fails_for_weak_model():
    assert passes_production_quality_gate(0.79) is False


def test_production_roc_auc_returns_value(monkeypatch):
    class FakeRun:
        class Data:
            metrics = {"roc_auc": 0.8461}

        data = Data()

    class FakeModelVersion:
        run_id = "test-run-id"

    class FakeClient:
        def get_model_version_by_alias(self, model_name, alias):
            assert model_name == "customer-churn-model"
            assert alias == "production"
            return FakeModelVersion()

        def get_run(self, run_id):
            assert run_id == "test-run-id"
            return FakeRun()

    monkeypatch.setattr(
        "src.train.mlflow.MlflowClient",
        FakeClient,
    )

    assert get_production_roc_auc("customer-churn-model") == 0.8461


def test_production_roc_auc_returns_none_when_no_production_model(
    monkeypatch,
):
    class FakeClient:
        def get_model_version_by_alias(self, model_name, alias):
            raise Exception("Production model not found")

    monkeypatch.setattr(
        "src.train.mlflow.MlflowClient",
        FakeClient,
    )

    assert get_production_roc_auc("customer-churn-model") is None
    
def test_production_comparison_gate_passes_when_candidate_is_better():
    assert (
        passes_production_comparison_gate(0.85, 0.84)
        is True
    )


def test_production_comparison_gate_fails_when_candidate_is_worse():
    assert (
        passes_production_comparison_gate(0.83, 0.84)
        is False
    )