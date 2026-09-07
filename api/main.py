import json
import logging
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    APIRouter,
)
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from slowapi import Limiter
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Literal

import src.predict as prediction_module

from src.config import (
    API_KEY,
    ADMIN_API_KEY,
    RATE_LIMIT,
    API_VERSION,
    CANARY_ENABLED,
    CANARY_TRAFFIC_PERCENT,
    CANARY_MODEL_VERSION,
)
from src.explain import explain_prediction
from src.train import rollback_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


api_key_header = APIKeyHeader(
    name="x-api-key",
    auto_error=False,
)

admin_api_key_header = APIKeyHeader(
    name="x-admin-api-key",
    auto_error=False,
)


def verify_api_key(
    x_api_key: str = Depends(api_key_header),
):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


def verify_admin_api_key(
    x_admin_api_key: str = Depends(admin_api_key_header),
):
    if not ADMIN_API_KEY or x_admin_api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing admin API key",
        )


class CustomerData(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]
    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]
    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]
    PaperlessBilling: Literal[
        "Yes",
        "No",
    ]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int


limiter = Limiter(
    key_func=lambda request: request.headers.get(
        "x-api-key",
        "anonymous",
    )
)


# --------------------------------------------------
# Backward-compatible references for tests
# --------------------------------------------------

model = prediction_module.model
preprocessor = prediction_module.preprocessor
model_version = prediction_module.model_version

predict_churn = prediction_module.predict_churn
reload_model = prediction_module.reload_model


def sync_prediction_references():
    global model, preprocessor, model_version

    model = prediction_module.model
    preprocessor = prediction_module.preprocessor
    model_version = prediction_module.model_version


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: attempting model load")

    if prediction_module.reload_model():
        sync_prediction_references()

        logger.info(
            "Application startup: model loaded successfully | "
            "model_version=%s",
            model_version,
        )
    else:
        sync_prediction_references()
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
    _rate_limit_exceeded_handler,
)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    start_time = time.time()

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    duration = time.time() - start_time

    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_seconds": round(duration, 3),
            }
        )
    )

    return response


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_version": model_version,
        "canary_enabled": CANARY_ENABLED,
        "canary_traffic_percent": CANARY_TRAFFIC_PERCENT,
        "canary_model_version": CANARY_MODEL_VERSION,
        "canary_model_loaded": (
            prediction_module.canary_model is not None
            and prediction_module.canary_preprocessor is not None
        ),
        "canary_loaded_version": prediction_module.canary_model_version,
    }


@app.get("/ready")
def readiness():
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready",
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "preprocessor_loaded": True,
        "model_version": model_version,
        "canary_enabled": CANARY_ENABLED,
        "canary_traffic_percent": CANARY_TRAFFIC_PERCENT,
        "canary_model_version": CANARY_MODEL_VERSION,
        "canary_model_loaded": (
            prediction_module.canary_model is not None
            and prediction_module.canary_preprocessor is not None
        ),
        "canary_loaded_version": prediction_module.canary_model_version,
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
    responses={},
)
@limiter.limit(RATE_LIMIT)
def predict(
    request: Request,
    customer_data: CustomerData,
):
    try:
        result = predict_churn(
            customer_data.model_dump(),
            request_id=request.state.request_id,
        )

        return result

    except Exception:
        logger.exception(
            "Prediction failed | request_id=%s",
            request.state.request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later.",
        )


@app.post(
    "/admin/mlflow-debug",
    dependencies=[Depends(verify_admin_api_key)],
)
def mlflow_debug():
    try:
        import mlflow

        client = mlflow.MlflowClient()

        production_model = (
            client.get_model_version_by_alias(
                "customer-churn-model",
                "production",
            )
        )

        return {
            "mlflow_version": mlflow.__version__,
            "production_alias_version": str(
                production_model.version
            ),
            "production_run_id": production_model.run_id,
            "application_model_version": model_version,
        }

    except Exception:
        logger.exception(
            "MLflow diagnostic failed"
        )

        raise HTTPException(
            status_code=500,
            detail="MLflow diagnostic failed",
        )


@app.post(
    "/admin/reload-model",
    dependencies=[Depends(verify_admin_api_key)],
)
def reload_model_endpoint():
    success = reload_model()

    sync_prediction_references()

    if not success:
        raise HTTPException(
            status_code=503,
            detail="Model reload failed",
        )

    logger.info(
        "Model reloaded successfully | model_version=%s",
        model_version,
    )

    return {
        "status": "success",
        "message": "Model reloaded successfully",
        "model_version": model_version,
    }


@app.post(
    "/admin/rollback-model",
    dependencies=[Depends(verify_admin_api_key)],
)
def rollback_model_endpoint():
    try:
        rollback_model("customer-churn-model")

        success = reload_model()

        sync_prediction_references()

        if not success:
            raise HTTPException(
                status_code=503,
                detail="Model reload failed after rollback",
            )

        logger.info(
            "Model rollback completed successfully | "
            "model_version=%s",
            model_version,
        )

        return {
            "status": "success",
            "message": "Model rollback completed successfully",
            "model_version": model_version,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Model rollback failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Model rollback failed. Please try again later.",
        )


@v1_router.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Predict customer churn",
    description=(
        "Version 1 endpoint for predicting customer churn probability "
        "and returning the final churn prediction."
    ),
)
@limiter.limit(RATE_LIMIT)
def predict_v1(
    request: Request,
    customer_data: CustomerData,
):
    try:
        result = predict_churn(
            customer_data.model_dump(),
            request_id=request.state.request_id,
        )

        return result

    except Exception:
        logger.exception(
            "Prediction failed | request_id=%s",
            request.state.request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later.",
        )


@v1_router.post(
    "/explain",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT)
def explain_v1(
    request: Request,
    customer_data: CustomerData,
):
    try:
        return explain_prediction(
            customer_data.model_dump()
        )

    except Exception:
        logger.exception(
            "Explanation failed | request_id=%s",
            request.state.request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Explanation failed. Please try again later.",
        )


@app.post(
    "/explain",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT)
def explain(
    request: Request,
    customer_data: CustomerData,
):
    try:
        return explain_prediction(
            customer_data.model_dump()
        )

    except Exception:
        logger.exception(
            "Explanation failed | request_id=%s",
            request.state.request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Explanation failed. Please try again later.",
        )


app.include_router(v1_router)