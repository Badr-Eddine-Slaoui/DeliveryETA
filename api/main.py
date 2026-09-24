import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    DeliveryPredictionRequest,
    DeliveryPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .utils import build_feature_row

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "hist_gradient_optimized_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "hist_gradient_optimized_metadata.json"

ml_resources: dict = {"pipeline": None, "metadata": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run train.py first."
        )

    ml_resources["pipeline"] = joblib.load(MODEL_PATH)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            ml_resources["metadata"] = json.load(f)
    else:
        ml_resources["metadata"] = {}

    yield

    ml_resources.clear()


app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Predicts total delivery time (in minutes) from order, "
    "driver, weather and traffic information.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline():
    pipeline = ml_resources.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return pipeline


def get_metadata() -> dict:
    return ml_resources.get("metadata") or {}


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Food Delivery Time Prediction API — see /docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=ml_resources.get("pipeline") is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    metadata = get_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="No metadata file found.")

    return ModelInfoResponse(
        model_name=metadata.get("model_name", "unknown"),
        features=metadata.get("features", []),
        metrics=metadata.get("metrics", {}),
    )


@app.post("/predict", response_model=DeliveryPredictionResponse)
def predict(request: DeliveryPredictionRequest):
    pipeline = get_pipeline()
    metadata = get_metadata()

    try:
        features_df = build_feature_row(request)
        prediction = pipeline.predict(features_df)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not compute a prediction from the given input: {exc}",
        ) from exc

    predicted_time_min = round(float(prediction), 1)

    mae = metadata.get("metrics", {}).get("test_mae_min", 0.0)
    predicted_range = (
        round(predicted_time_min - mae, 1),
        round(predicted_time_min + mae, 1),
    )

    return DeliveryPredictionResponse(
        predicted_time_min=predicted_time_min,
        predicted_time_range_min=predicted_range,
        computed_distance_km=round(float(features_df["Distance_km"].iloc[0]), 2),
        computed_preparation_time_min=float(
            features_df["Preparation_Time_min"].iloc[0]
        ),
        model_name=metadata.get("model_name", "HistGradientBoostingRegressor"),
        model_version="1.0.0",
    )