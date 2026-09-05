import json
import logging
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends, Request, APIRouter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Literal
from src.predict import predict_churn, model, preprocessor, reload_model
from src.config import API_KEY, RATE_LIMIT, API_VERSION
from src.explain import explain_prediction


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def verify_api_key(x_api_key: str = Depends(api_key_header)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

class CustomerData(BaseModel):

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int

limiter = Limiter(key_func=lambda request: request.headers.get("x-api-key", "anonymous"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: attempting model load")

    if reload_model():
        logger.info("Application startup: model loaded successfully")
    else:
        logger.error("Application startup: model unavailable")

    yield

app = FastAPI(
    lifespan=lifespan,
    title="Customer Churn Prediction API",
    description=(
        "Production-ready machine learning API for predicting "
        "customer churn probability and providing SHAP-based explanations."
    ),
    version=API_VERSION,
)

v1_router = APIRouter(prefix="/v1")

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    duration = time.time() - start_time

    logger.info(
        json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_seconds": round(duration, 3)
        })
    )

    return response

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.get("/ready")
def readiness():
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "preprocessor_loaded": True
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Predict customer churn",
    description=(
        "Predicts the probability that a customer will churn "
        "and returns the final churn prediction."
    ),
    responses={
        200: {
            "description": "Successful churn prediction",
            "content": {
                "application/json": {
                    "example": {
                        "churn_probability": 0.7139,
                        "prediction": 1
                    }
                }
            }
        },
        401: {
            "description": "Invalid or missing API key"
        },
        429: {
            "description": "Rate limit exceeded"
        }
    }
)

@limiter.limit(RATE_LIMIT)

def predict(request: Request, customer_data: CustomerData):
    try:
        result = predict_churn(
            customer_data.model_dump(),
            request_id=request.state.request_id
        )
        return result
    except Exception:
        logger.exception(
            "Prediction failed | request_id=%s",
            request.state.request_id,
        )
    raise HTTPException(
        status_code=500,
        detail="Prediction failed. Please try again later."
    )
@v1_router.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Predict customer churn (v1)",
    description="Version 1 of the customer churn prediction endpoint.",
)
@limiter.limit(RATE_LIMIT)
def predict_v1(request: Request, customer_data: CustomerData):
    return predict(request, customer_data)



@app.post(
    "/explain",
    dependencies=[Depends(verify_api_key)],
    summary="Explain churn prediction",
    description=(
        "Provides SHAP-based explanations showing which customer "
        "features increase or decrease the predicted churn risk."
    ),
    responses={
        200: {
            "description": "Successful SHAP explanation",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": [
                            {
                                "feature": "tenure",
                                "value": 5,
                                "shap_value": 0.7123,
                                "impact": "increases_churn",
                                "explanation": "tenure = 5 increases churn risk."
                            }
                        ]
                    }
                }
            }
        },
        401: {
            "description": "Invalid or missing API key"
        },
        429: {
            "description": "Rate limit exceeded"
        }
    }
)
@limiter.limit(RATE_LIMIT)
def explain(request: Request, customer_data: CustomerData):
    try:
        return explain_prediction(customer_data.model_dump())
    except Exception:
        logger.exception(
            "SHAP explanation failed | request_id=%s",
            request.state.request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="SHAP explanation failed. Please try again later."
        )

@v1_router.post(
    "/explain",
    dependencies=[Depends(verify_api_key)],
    summary="Explain churn prediction (v1)",
    description="Version 1 of the SHAP-based churn explanation endpoint.",
)
@limiter.limit(RATE_LIMIT)
def explain_v1(request: Request, customer_data: CustomerData):
    return explain(request, customer_data)

app.include_router(v1_router)
