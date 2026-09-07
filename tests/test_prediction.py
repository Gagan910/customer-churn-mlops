from fastapi.testclient import TestClient
from src.predict import predict_churn
from src.explain import explain_prediction
from api.main import app

client = TestClient(app)

def test_prediction():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    result = predict_churn(customer)

    assert "churn_probability" in result
    assert "prediction" in result
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in [0, 1]

def test_prediction_log_contains_model_version():
    import pandas as pd

    log_path = "data/processed/prediction_logs.csv"

    logs = pd.read_csv(log_path)

    assert "model_version" in logs.columns

def test_explanation():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    result = explain_prediction(customer)

    assert "prediction" in result
    assert len(result["prediction"]) > 0

    for explanation in result["prediction"]:
        assert "feature" in explanation
        assert "value" in explanation
        assert "shap_value" in explanation

def test_predict_without_api_key():
    response = client.post("/predict", json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"

def test_predict_with_api_key(monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "test-api-key")

    response = client.post(
        "/predict",
        json={},
        headers={"x-api-key": "test-api-key"}
    )

    assert response.status_code != 401
    
def test_predict_api_uses_canary_model(monkeypatch):
    monkeypatch.setattr(
        "api.main.API_KEY",
        "canary-api-test-key",
    )

    monkeypatch.setattr(
        "api.main.prediction_module.CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "api.main.prediction_module.CANARY_TRAFFIC_PERCENT",
        100,
    )
    monkeypatch.setattr(
        "api.main.prediction_module.model_version",
        "9",
    )
    monkeypatch.setattr(
        "api.main.prediction_module.canary_model_version",
        "10",
    )

    class FakePreprocessor:
        def transform(self, data):
            return data

    class FakeModel:
        def predict_proba(self, data):
            return [[0.2, 0.8]]

    monkeypatch.setattr(
        "api.main.prediction_module.canary_model",
        FakeModel(),
    )
    monkeypatch.setattr(
        "api.main.prediction_module.canary_preprocessor",
        FakePreprocessor(),
    )

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0,
    }

    response = client.post(
        "/predict",
        json=customer,
        headers={"x-api-key": "canary-api-test-key"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["churn_probability"] == 0.8
    assert data["prediction"] == 1

def test_explain_without_api_key():
    response = client.post("/explain", json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"

def test_health_is_public():
    response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0
    assert response.json()["status"] == "healthy"
    
def test_health_reports_model_version(monkeypatch):
    monkeypatch.setattr("api.main.model_version", "7")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_version"] == "7"
    
def test_health_reports_canary_status(monkeypatch):
    monkeypatch.setattr(
        "api.main.CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "api.main.CANARY_TRAFFIC_PERCENT",
        10.0,
    )
    monkeypatch.setattr(
        "api.main.CANARY_MODEL_VERSION",
        10,
    )
    monkeypatch.setattr(
        "api.main.prediction_module.canary_model",
        object(),
    )
    monkeypatch.setattr(
        "api.main.prediction_module.canary_preprocessor",
        object(),
    )
    monkeypatch.setattr(
        "api.main.prediction_module.canary_model_version",
        "10",
    )

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["canary_enabled"] is True
    assert data["canary_traffic_percent"] == 10.0
    assert data["canary_model_version"] == 10
    assert data["canary_model_loaded"] is True
    assert data["canary_loaded_version"] == "10"

def test_predict_returns_request_id(monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "request-id-test-key")

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    response = client.post(
        "/predict",
        json=customer,
        headers={"x-api-key": "request-id-test-key"}
    )

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0

def test_rate_limit(monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "test-api-key")

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    responses = []

    for _ in range(11):
        response = client.post(
            "/predict",
            json=customer,
            headers={"x-api-key": "test-api-key"}
        )
        responses.append(response.status_code)

    assert 429 in responses


def test_explain_has_impact(monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "test-api-key")

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    response = client.post(
        "/explain",
        json=customer,
        headers={"x-api-key": "test-api-key"}
    )

    assert response.status_code == 200

    explanations = response.json()["prediction"]

    assert len(explanations) > 0
    assert "impact" in explanations[0]
    assert explanations[0]["impact"] in [
        "increases_churn",
        "decreases_churn"
    ]
    assert "explanation" in explanations[0]
    assert "churn risk" in explanations[0]["explanation"]

def test_v1_predict_with_api_key(monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "test-api-key")

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    response = client.post(
        "/v1/predict",
        json=customer,
        headers={"x-api-key": "test-api-key"}
    )

    assert response.status_code == 200
    assert "churn_probability" in response.json()
    assert "prediction" in response.json()

def test_v1_explain_with_api_key(monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "test-api-key")

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    response = client.post(
        "/v1/explain",
        json=customer,
        headers={"x-api-key": "test-api-key"}
    )

    assert response.status_code == 200

    explanations = response.json()["prediction"]

    assert len(explanations) > 0
    assert "feature" in explanations[0]
    assert "impact" in explanations[0]
    assert "explanation" in explanations[0]

def test_predict_invalid_input(monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "test-api-key")

    response = client.post(
        "/predict",
        json={
            "gender": "InvalidGender"
        },
        headers={"x-api-key": "test-api-key"}
    )

    assert response.status_code == 422


def test_predict_internal_error(monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "error-test-api-key")
    monkeypatch.setattr(
        "api.main.predict_churn",
        lambda customer: (_ for _ in ()).throw(RuntimeError("test failure"))
    )

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    response = client.post(
        "/predict",
        json=customer,
        headers={"x-api-key": "error-test-api-key"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed. Please try again later."

def test_readiness_when_model_unavailable(monkeypatch):
    monkeypatch.setattr("api.main.model", None)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service not ready"


def test_mlflow_production_model_loads(tmp_path):
    import mlflow
    from xgboost import XGBClassifier

    tracking_db = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")

    model_name = "test-customer-churn-model"

    mlflow.set_experiment("test-customer-churn")

    with mlflow.start_run() as run:
        model = XGBClassifier(
            n_estimators=2,
            max_depth=2,
            random_state=42,
            eval_metric="logloss",
        )

        model.fit(
            [[0], [1], [2], [3]],
            [0, 0, 1, 1],
        )

        model_info = mlflow.xgboost.log_model(
            model,
            name="xgboost-model",
            registered_model_name=model_name,
        )

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        model_name,
        "production",
        str(model_info.registered_model_version),
    )

    loaded_model = mlflow.xgboost.load_model(
        f"models:/{model_name}@production"
    )

    assert loaded_model is not None
    assert loaded_model.__class__.__name__ == "XGBClassifier"

def test_admin_reload_model_with_valid_key(monkeypatch):
    monkeypatch.setattr("api.main.ADMIN_API_KEY", "test-admin-api-key")
    monkeypatch.setattr("api.main.reload_model", lambda: True)

    response = client.post(
        "/admin/reload-model",
        headers={"x-admin-api-key": "test-admin-api-key"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["message"] == "Model reloaded successfully"
    
def test_admin_reload_model_without_key():
    response = client.post("/admin/reload-model")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing admin API key"
    
def test_admin_reload_model_failure(monkeypatch):
    monkeypatch.setattr("api.main.ADMIN_API_KEY", "test-admin-api-key")
    monkeypatch.setattr("api.main.reload_model", lambda: False)

    response = client.post(
        "/admin/reload-model",
        headers={"x-admin-api-key": "test-admin-api-key"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model reload failed"
    
def test_admin_rollback_model_with_valid_key(monkeypatch):
    monkeypatch.setattr(
        "api.main.ADMIN_API_KEY",
        "test-admin-api-key",
    )
    monkeypatch.setattr(
        "api.main.prediction_module.model_version",
        "9",
    )
    monkeypatch.setattr(
        "api.main.rollback_model",
        lambda model_name: "9",
    )
    monkeypatch.setattr(
        "api.main.reload_model",
        lambda: True,
    )
    monkeypatch.setattr(
        "api.main.model_version",
        "9",
    )

    response = client.post(
        "/admin/rollback-model",
        headers={"x-admin-api-key": "test-admin-api-key"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert (
        response.json()["message"]
        == "Model rollback completed successfully"
    )
    assert response.json()["model_version"] == "9"


def test_admin_rollback_model_without_key():
    response = client.post("/admin/rollback-model")

    assert response.status_code == 401
    assert (
        response.json()["detail"]
        == "Invalid or missing admin API key"
    )


def test_admin_rollback_model_without_history(monkeypatch):
    monkeypatch.setattr(
        "api.main.ADMIN_API_KEY",
        "test-admin-api-key",
    )
    monkeypatch.setattr(
        "api.main.rollback_model",
        lambda model_name: (
            (_ for _ in ()).throw(
                ValueError(
                    "Cannot rollback model customer-churn-model "
                    "version 9: previous production version "
                    "is not recorded."
                )
            )
        ),
    )

    response = client.post(
        "/admin/rollback-model",
        headers={"x-admin-api-key": "test-admin-api-key"},
    )

    assert response.status_code == 409
    assert (
        response.json()["detail"]
        == (
            "Cannot rollback model customer-churn-model "
            "version 9: previous production version "
            "is not recorded."
        )
    )
    
def test_canary_model_not_loaded_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "src.predict.CANARY_ENABLED",
        False,
    )

    from src.predict import load_canary_model

    result = load_canary_model()

    assert result == (None, None, None)


def test_canary_model_requires_mlflow(monkeypatch):
    monkeypatch.setattr(
        "src.predict.CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.predict.CANARY_MODEL_VERSION",
        10,
    )
    monkeypatch.setattr(
        "src.predict.MODEL_SOURCE",
        "local",
    )

    from src.predict import load_canary_model

    result = load_canary_model()

    assert result == (None, None, None)


def test_canary_model_loads_from_mlflow(monkeypatch):
    import src.predict as prediction_module

    monkeypatch.setattr(
        prediction_module,
        "CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        prediction_module,
        "CANARY_MODEL_VERSION",
        10,
    )
    monkeypatch.setattr(
        prediction_module,
        "MODEL_SOURCE",
        "mlflow",
    )

    class FakeCanaryModel:
        pass

    class FakeModelInfo:
        version = 10
        run_id = "test-run-id"

    class FakeClient:
        def get_model_version(
            self,
            model_name,
            model_version,
        ):
            assert model_name == "customer-churn-model"
            assert model_version == "10"

            return FakeModelInfo()

    class FakeMLflow:
        class xgboost:
            @staticmethod
            def load_model(model_uri):
                assert (
                    model_uri
                    == "models:/customer-churn-model/10"
                )

                return FakeCanaryModel()

        class MlflowClient:
            def __new__(cls):
                return FakeClient()

        class artifacts:
            @staticmethod
            def download_artifacts(
                run_id,
                artifact_path,
            ):
                assert run_id == "test-run-id"
                assert (
                    artifact_path
                    == "model/preprocessor.pkl"
                )

                return "fake-preprocessor.pkl"

    monkeypatch.setitem(
        __import__("sys").modules,
        "mlflow",
        FakeMLflow,
    )

    monkeypatch.setattr(
        prediction_module.joblib,
        "load",
        lambda path: "fake-preprocessor",
    )

    result = prediction_module.load_canary_model()

    assert result[0].__class__.__name__ == "FakeCanaryModel"
    assert result[1] == "fake-preprocessor"
    assert result[2] == "10"


def test_canary_model_load_failure_returns_none(monkeypatch):
    import src.predict as prediction_module

    monkeypatch.setattr(
        prediction_module,
        "CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        prediction_module,
        "CANARY_MODEL_VERSION",
        10,
    )
    monkeypatch.setattr(
        prediction_module,
        "MODEL_SOURCE",
        "mlflow",
    )

    class FakeMLflow:
        class xgboost:
            @staticmethod
            def load_model(model_uri):
                raise RuntimeError(
                    "test canary load failure"
                )

    monkeypatch.setitem(
        __import__("sys").modules,
        "mlflow",
        FakeMLflow,
    )

    result = prediction_module.load_canary_model()

    assert result == (None, None, None)
    
def test_canary_routing_zero_percent(monkeypatch):
    import src.predict as prediction_module

    monkeypatch.setattr(
        prediction_module,
        "CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        prediction_module,
        "CANARY_TRAFFIC_PERCENT",
        0,
    )
    monkeypatch.setattr(
        prediction_module,
        "model_version",
        "9",
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model_version",
        "10",
    )

    selected = prediction_module._select_prediction_model()

    assert selected[2] == "9"
    assert selected[3] == "production"


def test_canary_routing_hundred_percent(monkeypatch):
    import src.predict as prediction_module

    monkeypatch.setattr(
        prediction_module,
        "CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        prediction_module,
        "CANARY_TRAFFIC_PERCENT",
        100,
    )
    monkeypatch.setattr(
        prediction_module,
        "model_version",
        "9",
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model",
        object(),
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_preprocessor",
        object(),
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model_version",
        "10",
    )

    selected = prediction_module._select_prediction_model()

    assert selected[2] == "10"
    assert selected[3] == "canary"


def test_canary_routing_fifty_percent_canary(monkeypatch):
    import src.predict as prediction_module

    monkeypatch.setattr(
        prediction_module,
        "CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        prediction_module,
        "CANARY_TRAFFIC_PERCENT",
        50,
    )
    monkeypatch.setattr(
        prediction_module,
        "model_version",
        "9",
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model",
        object(),
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_preprocessor",
        object(),
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model_version",
        "10",
    )
    monkeypatch.setattr(
        prediction_module.random,
        "uniform",
        lambda start, end: 25,
    )

    selected = prediction_module._select_prediction_model()

    assert selected[2] == "10"
    assert selected[3] == "canary"


def test_canary_routing_fifty_percent_production(monkeypatch):
    import src.predict as prediction_module

    monkeypatch.setattr(
        prediction_module,
        "CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        prediction_module,
        "CANARY_TRAFFIC_PERCENT",
        50,
    )
    monkeypatch.setattr(
        prediction_module,
        "model_version",
        "9",
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model",
        object(),
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_preprocessor",
        object(),
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model_version",
        "10",
    )
    monkeypatch.setattr(
        prediction_module.random,
        "uniform",
        lambda start, end: 75,
    )

    selected = prediction_module._select_prediction_model()

    assert selected[2] == "9"
    assert selected[3] == "production"
    
def test_canary_routing_falls_back_to_production_when_unavailable(
    monkeypatch,
):
    import src.predict as prediction_module

    monkeypatch.setattr(
        prediction_module,
        "CANARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        prediction_module,
        "CANARY_TRAFFIC_PERCENT",
        50,
    )
    monkeypatch.setattr(
        prediction_module,
        "CANARY_MODEL_VERSION",
        10,
    )
    monkeypatch.setattr(
        prediction_module,
        "model_version",
        "9",
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model",
        None,
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_preprocessor",
        None,
    )
    monkeypatch.setattr(
        prediction_module,
        "canary_model_version",
        None,
    )

    selected = prediction_module._select_prediction_model()

    assert selected[2] == "9"
    assert selected[3] == "production"