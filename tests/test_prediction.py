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
    
def test_explain_without_api_key():
    response = client.post("/explain", json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"
    
def test_health_is_public():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
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