from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


TARGET = "Time_taken(min)"

INPUT_FILENAME = "food_delivery_cleaned.csv"
OUTPUT_FILENAME = "food_delivery_features.csv"


RAW_DATETIME_COLUMNS = [
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
]

RAW_GPS_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]


REQUIRED_INPUT_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Vehicle_condition",
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    TARGET,
]


NEW_FEATURES = [
    "Distance_km",
    "Order_Hour",
    "Order_Day",
    "Order_Month",
    "Order_DayOfWeek",
    "Is_Weekend",
    "Pickup_Hour",
    "Preparation_Time_min",
]


NUMERIC_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
    "Distance_km",
    "Order_Day",
    "Preparation_Time_min",
]


CATEGORICAL_FEATURES = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "Vehicle_condition",
    "Order_Hour",
    "Order_Month",
    "Order_DayOfWeek",
    "Pickup_Hour",
]


BINARY_FEATURES = [
    "Is_Weekend",
]


def resolve_project_root() -> Path:
    if "__file__" in globals():
        start = Path(__file__).resolve()
    else:
        start = Path.cwd()

    candidates = [start, *start.parents]

    for path in candidates:
        if (path / "data" / "processed").exists():
            return path

    return start


PROJECT_ROOT = resolve_project_root()

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / INPUT_FILENAME
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / OUTPUT_FILENAME
)


def load_cleaned_data(
    input_path: Path,
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(
            "The input dataset is empty."
        )

    return df


def validate_required_columns(
    df: pd.DataFrame,
) -> None:

    missing_columns = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )


def validate_target(
    df: pd.DataFrame,
) -> None:

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            f"Target '{TARGET}' contains missing values."
        )


def convert_datetime_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    for column in RAW_DATETIME_COLUMNS:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    invalid_datetime_values = {
        column: int(df[column].isna().sum())
        for column in RAW_DATETIME_COLUMNS
        if df[column].isna().any()
    }

    if invalid_datetime_values:
        raise ValueError(
            "Invalid or missing datetime values detected: "
            f"{invalid_datetime_values}"
        )

    return df


def create_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["Order_Hour"] = (
        df["Time_Orderd"]
        .dt.hour
        .astype("int8")
    )

    df["Order_Day"] = (
        df["Order_Date"]
        .dt.day
        .astype("int8")
    )

    df["Order_Month"] = (
        df["Order_Date"]
        .dt.month
        .astype("int8")
    )

    df["Order_DayOfWeek"] = (
        df["Order_Date"]
        .dt.dayofweek
        .astype("int8")
    )

    df["Is_Weekend"] = (
        df["Order_DayOfWeek"]
        .isin([5, 6])
        .astype("int8")
    )

    df["Pickup_Hour"] = (
        df["Time_Order_picked"]
        .dt.hour
        .astype("int8")
    )

    order_minutes = (
        df["Time_Orderd"].dt.hour * 60
        + df["Time_Orderd"].dt.minute
        + df["Time_Orderd"].dt.second / 60
    )

    pickup_minutes = (
        df["Time_Order_picked"].dt.hour * 60
        + df["Time_Order_picked"].dt.minute
        + df["Time_Order_picked"].dt.second / 60
    )

    preparation_time = (
        pickup_minutes - order_minutes
    ) % (24 * 60)

    df["Preparation_Time_min"] = (
        preparation_time
        .round()
        .astype("float32")
    )

    return df


def haversine_distance_km(
    latitude_1: pd.Series,
    longitude_1: pd.Series,
    latitude_2: pd.Series,
    longitude_2: pd.Series,
) -> pd.Series:

    earth_radius_km = 6371.0

    lat1 = np.radians(latitude_1)
    lon1 = np.radians(longitude_1)

    lat2 = np.radians(latitude_2)
    lon2 = np.radians(longitude_2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return earth_radius_km * c


def create_distance_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["Distance_km"] = haversine_distance_km(
        latitude_1=df["Restaurant_latitude"],
        longitude_1=df["Restaurant_longitude"],
        latitude_2=df["Delivery_location_latitude"],
        longitude_2=df["Delivery_location_longitude"],
    )

    return df


def validate_distance_feature(
    df: pd.DataFrame,
) -> None:

    if "Distance_km" not in df.columns:
        raise ValueError(
            "Distance_km was not created."
        )

    if df["Distance_km"].isna().any():
        raise ValueError(
            "Distance_km contains missing values."
        )

    if not np.isfinite(
        df["Distance_km"]
    ).all():
        raise ValueError(
            "Distance_km contains non-finite values."
        )

    if (
        df["Distance_km"] < 0
    ).any():
        raise ValueError(
            "Distance_km contains negative values."
        )
