from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "food_delivery.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "food_delivery_cleaned.csv"
)

IDENTIFIER_COLUMNS = [
    "ID",
    "Delivery_person_ID",
]

CATEGORICAL_COLUMNS = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

GPS_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]

FINAL_COLUMNS = [
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
    "Time_taken(min)",
]


def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)
