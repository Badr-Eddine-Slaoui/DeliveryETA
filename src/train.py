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


def get_feature_lists():
    numeric_features = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "Distance_km",
        "Order_Day",
        "Preparation_Time_min",
        "Vehicle_condition",
        "Traffic_Level",
        "Distance_x_Traffic",
        "Distance_per_Delivery",
        "Prep_to_Distance_Ratio",
        "Order_Hour_sin",
        "Order_Hour_cos",
        "Order_Hour_sin2",
        "Order_Hour_cos2",
        "Pickup_Hour_sin",
        "Pickup_Hour_cos",
        "Pickup_Hour_sin2",
        "Pickup_Hour_cos2",
        "Order_DayOfWeek_sin",
        "Order_DayOfWeek_cos",
        "Order_Month_sin",
        "Order_Month_cos",
    ]

    categorical_features = [
        "Weatherconditions",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
        "Order_Hour",
        "Order_Month",
        "Order_DayOfWeek",
        "Pickup_Hour",
    ]

    binary_features = [
        "Is_Weekend",
        "Is_Rush_Hour",
    ]

    all_features = numeric_features + categorical_features + binary_features

    return numeric_features, categorical_features, binary_features, all_features


def split_features_target(df, all_features, target=TARGET):
    X = df[all_features].copy()
    y = df[target].copy()
    return X, y


def split_train_test(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_hist_preprocessor(numeric_features, categorical_features, binary_features):
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
            ("bin", binary_pipeline, binary_features),
        ],
        remainder="drop",
    )

    return preprocessor


def build_categorical_mask(numeric_features, categorical_features, binary_features):
    mask = (
        [False] * len(numeric_features)
        + [True] * len(categorical_features)
        + [False] * len(binary_features)
    )
    return mask


def build_model(categorical_mask, params=BEST_PARAMS, random_state=RANDOM_STATE):
    model = HistGradientBoostingRegressor(
        random_state=random_state,
        categorical_features=categorical_mask,
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        max_bins=params["max_bins"],
    )
    return model


def build_pipeline(preprocessor, model):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    X_train_transformed = pipeline.named_steps["preprocessor"].transform(X_train)
    p = X_train_transformed.shape[1]

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    metrics = {
        "Train_MAE": mean_absolute_error(y_train, y_train_pred),
        "Test_MAE": mean_absolute_error(y_test, y_test_pred),
        "Train_MSE": train_mse,
        "Test_MSE": test_mse,
        "Train_RMSE": float(np.sqrt(train_mse)),
        "Test_RMSE": float(np.sqrt(test_mse)),
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Train_Adjusted_R2": adjusted_r2(train_r2, len(y_train), p),
        "Test_Adjusted_R2": adjusted_r2(test_r2, len(y_test), p),
        "n_features_after_encoding": p,
    }

    return metrics


def save_pipeline(pipeline, model_path=FINAL_MODEL_PATH):
    joblib.dump(pipeline, model_path)


def save_metadata(metrics, all_features, params=BEST_PARAMS, metadata_path=FINAL_METADATA_PATH, random_state=RANDOM_STATE):
    metadata = {
        "model_name": "HistGradientBoostingRegressor",
        "optimization_method": "RandomizedSearchCV + GridSearchCV (offline)",
        "best_hyperparameters": params,
        "features": all_features,
        "target": TARGET,
        "metrics": {
            "test_mae_min": float(metrics["Test_MAE"]),
            "test_rmse_min": float(metrics["Test_RMSE"]),
            "test_r2": float(metrics["Test_R2"]),
            "test_adjusted_r2": float(metrics["Test_Adjusted_R2"]),
            "train_rmse_min": float(metrics["Train_RMSE"]),
            "train_r2": float(metrics["Train_R2"]),
            "rmse_gap_min": float(metrics["Test_RMSE"] - metrics["Train_RMSE"]),
            "r2_gap": float(metrics["Train_R2"] - metrics["Test_R2"]),
        },
        "random_state": random_state,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def training_pipeline(data_path=DATA_PATH):
    df = load_data(data_path)
    df = engineer_features(df)

    numeric_features, categorical_features, binary_features, all_features = get_feature_lists()

    X, y = split_features_target(df, all_features)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_hist_preprocessor(numeric_features, categorical_features, binary_features)
    categorical_mask = build_categorical_mask(numeric_features, categorical_features, binary_features)
    model = build_model(categorical_mask)
    pipeline = build_pipeline(preprocessor, model)

    pipeline.fit(X_train, y_train)

    metrics = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)

    save_pipeline(pipeline)
    save_metadata(metrics, all_features)

    return pipeline, metrics


if __name__ == "__main__":
    trained_pipeline, final_metrics = training_pipeline()
    print(final_metrics)
