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
