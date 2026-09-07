from src.train import (
    calculate_file_hash,
    get_production_roc_auc,
    get_training_environment,
    passes_production_comparison_gate,
    passes_production_quality_gate,
    promote_model_to_production,
    rollback_model,
)


def test_quality_gate_passes_for_good_model():
    assert passes_production_quality_gate(0.85) is True


def test_quality_gate_fails_for_weak_model():
    assert passes_production_quality_gate(0.79) is False


def test_calculate_file_hash(tmp_path):
    test_file = tmp_path / "test_data.csv"

    test_file.write_bytes(
        b"customerID,Churn\n"
        b"001,Yes\n"
        b"002,No\n"
    )

    expected_hash = (
        "5eb7aae6de6374174fdb91a3f07addc4"
        "54da3586b48289732b9ba6d39c6b400e"
    )

    actual_hash = calculate_file_hash(test_file)

    assert actual_hash == expected_hash


def test_training_environment_contains_required_versions():
    environment = get_training_environment()

    expected_keys = {
        "python",
        "pandas",
        "numpy",
        "scikit_learn",
        "xgboost",
        "joblib",
        "mlflow",
    }

    assert set(environment.keys()) == expected_keys

    for value in environment.values():
        assert isinstance(value, str)
        assert value


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


def test_promotion_records_previous_production_version(
    monkeypatch,
):
    recorded_tags = []
    promoted_alias = []

    class FakeProductionModel:
        version = "7"

    class FakeClient:
        def get_model_version_by_alias(
            self,
            model_name,
            alias,
        ):
            assert model_name == "customer-churn-model"
            assert alias == "production"

            return FakeProductionModel()

        def set_model_version_tag(
            self,
            model_name,
            model_version,
            key,
            value,
        ):
            recorded_tags.append(
                (
                    model_name,
                    model_version,
                    key,
                    value,
                )
            )

        def set_registered_model_alias(
            self,
            model_name,
            alias,
            model_version,
        ):
            promoted_alias.append(
                (
                    model_name,
                    alias,
                    model_version,
                )
            )

    monkeypatch.setattr(
        "src.train.mlflow.MlflowClient",
        FakeClient,
    )

    promote_model_to_production(
        "customer-churn-model",
        9,
    )

    assert recorded_tags == [
        (
            "customer-churn-model",
            "9",
            "previous_production_version",
            "7",
        )
    ]

    assert promoted_alias == [
        (
            "customer-churn-model",
            "production",
            "9",
        )
    ]


def test_promotion_without_existing_production_does_not_create_tag(
    monkeypatch,
):
    recorded_tags = []
    promoted_alias = []

    class FakeClient:
        def get_model_version_by_alias(
            self,
            model_name,
            alias,
        ):
            raise Exception("Production model not found")

        def set_model_version_tag(
            self,
            model_name,
            model_version,
            key,
            value,
        ):
            recorded_tags.append(
                (
                    model_name,
                    model_version,
                    key,
                    value,
                )
            )

        def set_registered_model_alias(
            self,
            model_name,
            alias,
            model_version,
        ):
            promoted_alias.append(
                (
                    model_name,
                    alias,
                    model_version,
                )
            )

    monkeypatch.setattr(
        "src.train.mlflow.MlflowClient",
        FakeClient,
    )

    promote_model_to_production(
        "customer-churn-model",
        1,
    )

    assert recorded_tags == []

    assert promoted_alias == [
        (
            "customer-churn-model",
            "production",
            "1",
        )
    ]


def test_rollback_model_restores_previous_production_version(
    monkeypatch,
):
    promoted_alias = []

    class FakeProductionModel:
        version = "9"
        tags = {
            "previous_production_version": "7"
        }

    class FakeClient:
        def get_model_version_by_alias(
            self,
            model_name,
            alias,
        ):
            assert model_name == "customer-churn-model"
            assert alias == "production"

            return FakeProductionModel()

        def set_registered_model_alias(
            self,
            model_name,
            alias,
            model_version,
        ):
            promoted_alias.append(
                (
                    model_name,
                    alias,
                    model_version,
                )
            )

    monkeypatch.setattr(
        "src.train.mlflow.MlflowClient",
        FakeClient,
    )

    previous_version = rollback_model(
        "customer-churn-model"
    )

    assert previous_version == "7"

    assert promoted_alias == [
        (
            "customer-churn-model",
            "production",
            "7",
        )
    ]


def test_rollback_model_fails_without_previous_version(
    monkeypatch,
):
    promoted_alias = []

    class FakeProductionModel:
        version = "9"
        tags = {}

    class FakeClient:
        def get_model_version_by_alias(
            self,
            model_name,
            alias,
        ):
            return FakeProductionModel()

        def set_registered_model_alias(
            self,
            model_name,
            alias,
            model_version,
        ):
            promoted_alias.append(
                (
                    model_name,
                    alias,
                    model_version,
                )
            )

    monkeypatch.setattr(
        "src.train.mlflow.MlflowClient",
        FakeClient,
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="previous production version is not recorded",
    ):
        rollback_model(
            "customer-churn-model"
        )

    assert promoted_alias == []