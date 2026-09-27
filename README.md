# DeliveryETA 🛵⏱️

> **End-to-end food delivery time prediction** — from raw data to a production-ready REST API and interactive Streamlit dashboard.

---

## Overview

DeliveryETA is a machine-learning project that predicts the total delivery time (in minutes) for food orders. It covers the full ML lifecycle:

| Stage | What it does |
|---|---|
| **Data Cleaning** | Normalises types, imputes missing values, fixes GPS anomalies |
| **EDA** | Univariate & bivariate analysis, Spearman correlation, temporal patterns |
| **Feature Engineering** | Computes haversine distance, temporal/cyclical features and interaction terms |
| **Model Training** | Trains a tuned `HistGradientBoostingRegressor` (offline hyperparameter search via `RandomizedSearchCV` + `GridSearchCV`) |
| **REST API** | FastAPI service exposing `/predict`, `/health` and `/model-info` endpoints |
| **Dashboard** | Streamlit app with a prediction form, EDA charts and model-performance metrics |
| **Containerisation** | Docker Compose orchestration of all three services |

---

## Project Structure

```
DeliveryETA/
├── src/
│   ├── cleaning.py                # Data cleaning pipeline
│   ├── feature_engineering.py     # Feature extraction & validation pipeline
│   └── train.py                   # Model training & artefact persistence
├── api/
│   ├── main.py                    # FastAPI application & endpoints
│   ├── schemas.py                 # Pydantic request/response models
│   └── utils.py                   # Inference-time feature computation
├── app/
│   └── main.py                    # Streamlit dashboard
├── models/
│   ├── hist_gradient_optimized_pipeline.joblib
│   └── hist_gradient_optimized_metadata.json
├── notebooks/
│   ├── 01_cleaning.ipynb          # Data cleaning walkthrough
│   ├── 02_eda.ipynb               # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb          # Model selection & hyperparameter tuning
├── reports/
│   └── figures/
│       ├── eda/                   # 18 EDA charts (PNG)
│       └── feature_engineering/   # Distance distribution plot
├── data/
│   ├── raw/                       # Original CSV (not tracked by git)
│   └── processed/                 # Cleaned & feature-engineered CSVs (not tracked)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/Badr-Eddine-Slaoui/DeliveryETA.git
cd DeliveryETA
cp .env.example .env   # fill in FASTAPI_PORT, STREAMLIT_PORT, FASTAPI_URL
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI docs | `http://localhost:<FASTAPI_PORT>/docs` |
| Streamlit app | `http://localhost:<STREAMLIT_PORT>` |

### 3. Local development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full data pipeline (requires data/raw/food_delivery.csv)
python -m src.cleaning
python -m src.feature_engineering
python -m src.train

# Start the API
uvicorn api.main:app --reload

# Start the dashboard (separate terminal)
streamlit run app/main.py
```

---

## API Reference

### `POST /predict`

Accepts order details and returns a predicted delivery time.

**Request body example:**

```json
{
  "delivery_person_age": 29,
  "delivery_person_ratings": 4.6,
  "restaurant_latitude": 12.913041,
  "restaurant_longitude": 77.683237,
  "delivery_location_latitude": 13.043041,
  "delivery_location_longitude": 77.813237,
  "distance_km": null,
  "order_date": "2026-09-26",
  "time_ordered": "19:45:00",
  "time_order_picked": "19:50:00",
  "weather_conditions": "Stormy",
  "road_traffic_density": "Jam",
  "vehicle_condition": 2,
  "type_of_order": "Snack",
  "type_of_vehicle": "scooter",
  "multiple_deliveries": 1,
  "festival": "No",
  "city": "Metropolitian"
}
```

> Provide either `distance_km` **or** all four GPS coordinates — not both are required simultaneously.

**Response:**

```json
{
  "predicted_time_min": 34.5,
  "predicted_time_range_min": [30.2, 38.8],
  "computed_distance_km": 18.32,
  "computed_preparation_time_min": 5.0,
  "model_name": "HistGradientBoostingRegressor",
  "model_version": "1.0.0"
}
```

### `GET /health`

Returns `{ "status": "ok", "model_loaded": true }`.

### `GET /model-info`

Returns the model name, full feature list and test-set metrics from the metadata file.

---

## Notebooks & Reports

| Notebook | Description |
|---|---|
| `01_cleaning.ipynb` | Interactive walkthrough of all cleaning steps with per-step QA checks |
| `02_eda.ipynb` | Univariate/bivariate analysis, Spearman correlation matrix, temporal patterns |
| `03_feature_engineering.ipynb` | Haversine distance derivation, temporal feature extraction, feature group validation |
| `04_modeling.ipynb` | Baseline model, `RandomizedSearchCV` + `GridSearchCV` tuning, final evaluation |

All figures exported during EDA and feature engineering are saved under `reports/figures/` as PNG files.

---

## Model

| Property | Value |
|---|---|
| Algorithm | `HistGradientBoostingRegressor` |
| Hyperparameter search | `RandomizedSearchCV` then `GridSearchCV` (offline) |
| Key hyperparameters | `lr=0.076`, `max_iter=623`, `max_depth=15`, `max_bins=207` |

Features include ordinal traffic level, haversine distance, sin/cos cyclical encodings of hour/day/month, interaction terms (distance x traffic, distance per delivery, prep-to-distance ratio) and a rush-hour binary flag.

---

## Environment Variables

| Variable | Description |
|---|---|
| `FASTAPI_PORT` | Host port mapped to the FastAPI container (e.g. `8000`) |
| `STREAMLIT_PORT` | Host port mapped to the Streamlit container (e.g. `8501`) |
| `FASTAPI_URL` | Full URL of the `/predict` endpoint used by the Streamlit app |

---

## Tech Stack

- **Python 3.12**
- **pandas · numpy · scikit-learn** — data processing and modelling
- **FastAPI · Uvicorn · Pydantic** — REST API
- **Streamlit · Plotly** — interactive dashboard
- **joblib** — model serialisation
- **Docker · Docker Compose** — containerisation

---

## License

This project is released for educational and portfolio purposes.
