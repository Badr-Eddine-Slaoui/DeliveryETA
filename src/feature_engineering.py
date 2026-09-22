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
