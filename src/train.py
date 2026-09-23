from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "food_delivery_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODEL_DIR / "hist_gradient_optimized_pipeline.joblib"
FINAL_METADATA_PATH = MODEL_DIR / "hist_gradient_optimized_metadata.json"

TARGET = "Time_taken(min)"

TRAFFIC_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

RUSH_HOURS = {12, 13, 14, 19, 20, 21}

BEST_PARAMS = {
    "learning_rate": 0.0762795063212169,
    "max_iter": 623,
    "max_leaf_nodes": 94,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "l2_regularization": 2.98,
    "max_bins": 207,
}


def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    return df


def add_ordinal_traffic_level(df):
    df = df.copy()
    df["Traffic_Level"] = df["Road_traffic_density"].map(TRAFFIC_ORDER)
    return df


def add_interaction_features(df):
    df = df.copy()
    df["Distance_x_Traffic"] = df["Distance_km"] * df["Traffic_Level"]
    df["Distance_per_Delivery"] = df["Distance_km"] / (df["multiple_deliveries"] + 1)
    df["Prep_to_Distance_Ratio"] = df["Preparation_Time_min"] / (df["Distance_km"] + 1)
    return df


def add_rush_hour_feature(df):
    df = df.copy()
    df["Is_Rush_Hour"] = df["Order_Hour"].isin(RUSH_HOURS).astype(int)
    return df


def add_cyclical_features(df):
    df = df.copy()

    df["Order_Hour_sin"] = np.sin(2 * np.pi * df["Order_Hour"] / 24)
    df["Order_Hour_cos"] = np.cos(2 * np.pi * df["Order_Hour"] / 24)

    df["Pickup_Hour_sin"] = np.sin(2 * np.pi * df["Pickup_Hour"] / 24)
    df["Pickup_Hour_cos"] = np.cos(2 * np.pi * df["Pickup_Hour"] / 24)

    df["Order_Hour_sin2"] = np.sin(2 * np.pi * 2 * df["Order_Hour"] / 24)
    df["Order_Hour_cos2"] = np.cos(2 * np.pi * 2 * df["Order_Hour"] / 24)

    df["Pickup_Hour_sin2"] = np.sin(2 * np.pi * 2 * df["Pickup_Hour"] / 24)
    df["Pickup_Hour_cos2"] = np.cos(2 * np.pi * 2 * df["Pickup_Hour"] / 24)

    df["Order_DayOfWeek_sin"] = np.sin(2 * np.pi * df["Order_DayOfWeek"] / 7)
    df["Order_DayOfWeek_cos"] = np.cos(2 * np.pi * df["Order_DayOfWeek"] / 7)

    df["Order_Month_sin"] = np.sin(2 * np.pi * (df["Order_Month"] - 1) / 12)
    df["Order_Month_cos"] = np.cos(2 * np.pi * (df["Order_Month"] - 1) / 12)

    return df


def cap_distance_outliers(df, quantile=0.995):
    cap = df["Distance_km"].quantile(quantile)
    df = df[df["Distance_km"] <= cap].reset_index(drop=True)
    return df


def engineer_features(df):
    df = add_ordinal_traffic_level(df)
    df = add_interaction_features(df)
    df = add_rush_hour_feature(df)
    df = add_cyclical_features(df)
    df = cap_distance_outliers(df)
    return df
