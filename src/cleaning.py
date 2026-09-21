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


def compute_preparation_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    diff = (
        df["Time_Order_picked"] - df["Time_Orderd"]
    ).dt.total_seconds() / 60
    diff = np.where(diff < 0, diff + 24 * 60, diff)

    df["Preparation_Time_min"] = diff

    return df


def impute_time_orderd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    median_prep_time = df["Preparation_Time_min"].median()

    missing_order_time = df["Time_Orderd"].isna()

    df["Time_Orderd_Was_Missing"] = missing_order_time.astype(int)

    df.loc[missing_order_time, "Time_Orderd"] = (
        df.loc[missing_order_time, "Time_Order_picked"]
        - pd.to_timedelta(median_prep_time, unit="m")
    )

    diff = (
        df["Time_Order_picked"] - df["Time_Orderd"]
    ).dt.total_seconds() / 60
    diff = np.where(diff < 0, diff + 24 * 60, diff)

    df["Preparation_Time_min"] = diff

    return df


def impute_delivery_person_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    median_value = df["Delivery_person_Age"].median()

    df["Delivery_person_Age"] = df[
        "Delivery_person_Age"
    ].fillna(median_value)

    return df


def impute_delivery_person_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    median_value = df["Delivery_person_Ratings"].median()

    df["Delivery_person_Ratings"] = df[
        "Delivery_person_Ratings"
    ].fillna(median_value)

    return df


def impute_multiple_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    median_value = df["multiple_deliveries"].median()

    df["multiple_deliveries"] = df[
        "multiple_deliveries"
    ].fillna(median_value)

    return df


def impute_weather_conditions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mode_value = df["Weatherconditions"].mode(dropna=True)

    if not mode_value.empty:
        df["Weatherconditions"] = df[
            "Weatherconditions"
        ].fillna(mode_value.iloc[0])

    return df


def impute_road_traffic_density(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mode_value = df["Road_traffic_density"].mode(dropna=True)

    if not mode_value.empty:
        df["Road_traffic_density"] = df[
            "Road_traffic_density"
        ].fillna(mode_value.iloc[0])

    return df


def impute_festival(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mode_value = df["Festival"].mode(dropna=True)

    if not mode_value.empty:
        df["Festival"] = df["Festival"].fillna(
            mode_value.iloc[0]
        )

    return df


def impute_city(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["City"] = df["City"].fillna("Unknown")

    return df


def preserve_valid_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def validate_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df["Time_taken(min)"].isna().any():
        raise ValueError(
            "Missing values detected in Time_taken(min)."
        )

    return df


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[FINAL_COLUMNS].copy()


def verify_cleaned_data(df: pd.DataFrame) -> None:
    print("\n--- Cleaning verification ---")

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())
    
    print("\nDelivery person age below 20 or above 40:")
    print(
        (
            (df["Delivery_person_Age"] < 20)
            | (df["Delivery_person_Age"] > 40)
        ).sum()
    )

    print("\nRatings above 5:")
    print(
        (
            df["Delivery_person_Ratings"] > 5
        ).sum()
    )

    print("\nNegative GPS coordinates:")
    print(
        (
            df[GPS_COLUMNS] < 0
        ).any(axis=1).sum()
    )

    print("\nRestaurant coordinates equal to (0, 0):")
    print(
        (
            (df["Restaurant_latitude"] == 0)
            & (df["Restaurant_longitude"] == 0)
        ).sum()
    )

    print("\nUnique City values:")
    print(df["City"].unique())

    print("\nTarget statistics:")
    print(df["Time_taken(min)"].describe())

    print("\nData types:")
    print(df.dtypes)


def save_cleaned_data(
    df: pd.DataFrame,
    file_path: Path,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        file_path,
        index=False,
    )


def cleaning_data_pipeline() -> pd.DataFrame:
    df = load_data(RAW_DATA_PATH)

    df = clean_categorical_columns(df)

    df = convert_data_types(df)

    df = remove_duplicate_rows(df)

    df = validate_ids(df)

    df = clean_invalid_ratings(df)

    df = remove_invalid_gps_coordinates(df)
    
    df = clip_delivery_person_age(df)

    df = remove_identifiers(df)

    df = encode_vehicle_condition(df)

    df = compute_preparation_time(df)

    df = impute_time_orderd(df)

    df = impute_delivery_person_age(df)

    df = impute_delivery_person_ratings(df)

    df = impute_multiple_deliveries(df)

    df = impute_weather_conditions(df)

    df = impute_road_traffic_density(df)

    df = impute_festival(df)

    df = impute_city(df)

    df = preserve_valid_outliers(df)

    df = validate_target(df)

    df = finalize_columns(df)

    verify_cleaned_data(df)

    save_cleaned_data(
        df,
        PROCESSED_DATA_PATH,
    )

    return df


if __name__ == "__main__":
    cleaning_data_pipeline()
