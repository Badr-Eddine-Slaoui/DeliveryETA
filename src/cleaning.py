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


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["Weatherconditions"] = (
        df["Weatherconditions"]
        .str.replace("conditions", "", regex=False)
        .str.strip()
    )

    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "NaN": pd.NA,
                    "nan": pd.NA,
                    "": pd.NA,
                }
            )
        )

    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Delivery_person_Age"] = pd.to_numeric(
        df["Delivery_person_Age"],
        errors="coerce",
        downcast="integer",
    )

    df["Delivery_person_Ratings"] = pd.to_numeric(
        df["Delivery_person_Ratings"]
        .astype("string")
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
        downcast="float",
    )

    for column in GPS_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["Vehicle_condition"] = pd.to_numeric(
        df["Vehicle_condition"],
        errors="coerce",
        downcast="integer",
    )

    df["multiple_deliveries"] = pd.to_numeric(
        df["multiple_deliveries"],
        errors="coerce",
        downcast="integer",
    )

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce",
        format="%d-%m-%Y",
    )

    df["Time_Orderd"] = pd.to_datetime(
        df["Time_Orderd"],
        errors="coerce",
        format="%H:%M:%S",
    )

    df["Time_Order_picked"] = pd.to_datetime(
        df["Time_Order_picked"],
        errors="coerce",
        format="%H:%M:%S",
    )

    df["Time_taken(min)"] = pd.to_numeric(
        df["Time_taken(min)"]
        .astype("string")
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
        downcast="integer",
    )

    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def validate_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df["ID"].duplicated().any():
        raise ValueError("Duplicate ID values detected.")

    return df.copy()


def clean_invalid_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[
        df["Delivery_person_Ratings"] > 5,
        "Delivery_person_Ratings",
    ] = np.nan

    return df


def remove_invalid_gps_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    restaurant_zero_coordinates = (
        (df["Restaurant_latitude"] == 0)
        & (df["Restaurant_longitude"] == 0)
    )

    df = df.loc[~restaurant_zero_coordinates].copy()

    negative_gps = (
        df[GPS_COLUMNS] < 0
    ).any(axis=1)

    df.loc[negative_gps, GPS_COLUMNS] = df.loc[negative_gps, GPS_COLUMNS].abs()

    return df

def clip_delivery_person_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Delivery_person_Age"] = df["Delivery_person_Age"].clip(
        lower=20,
        upper=40,
    )

    return df


def remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=IDENTIFIER_COLUMNS,
        errors="ignore",
    ).copy()


def encode_vehicle_condition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Vehicle_condition"] = df["Vehicle_condition"].astype("category")

    return df
