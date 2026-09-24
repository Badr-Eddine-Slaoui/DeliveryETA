import numpy as np
import pandas as pd

from .schemas import DeliveryPredictionRequest

TRAFFIC_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
RUSH_HOURS = {12, 13, 14, 19, 20, 21}

AGE_CLIP_MIN, AGE_CLIP_MAX = 20, 40


def haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    earth_radius_km = 6371.0

    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    delta_lat = lat2_r - lat1_r
    delta_lon = lon2_r - lon1_r

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(delta_lon / 2) ** 2
    )
    a = np.clip(a, 0, 1)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return float(earth_radius_km * c)


def resolve_distance_km(request: DeliveryPredictionRequest) -> float:
    if request.distance_km is not None:
        return request.distance_km

    return haversine_distance_km(
        lat1=request.restaurant_latitude,
        lon1=request.restaurant_longitude,
        lat2=request.delivery_location_latitude,
        lon2=request.delivery_location_longitude,
    )


def compute_temporal_features(request: DeliveryPredictionRequest) -> dict:

    order_hour = request.time_ordered.hour
    pickup_hour = request.time_order_picked.hour

    order_minutes = (
        request.time_ordered.hour * 60
        + request.time_ordered.minute
        + request.time_ordered.second / 60
    )
    pickup_minutes = (
        request.time_order_picked.hour * 60
        + request.time_order_picked.minute
        + request.time_order_picked.second / 60
    )
    
    preparation_time_min = round((pickup_minutes - order_minutes) % (24 * 60))

    day_of_week = request.order_date.weekday()

    return {
        "Order_Hour": order_hour,
        "Order_Day": request.order_date.day,
        "Order_Month": request.order_date.month,
        "Order_DayOfWeek": day_of_week,
        "Is_Weekend": int(day_of_week in (5, 6)),
        "Pickup_Hour": pickup_hour,
        "Preparation_Time_min": float(preparation_time_min),
    }


def compute_traffic_level(road_traffic_density: str) -> int:
    return TRAFFIC_ORDER[road_traffic_density]


def compute_interaction_features(
    distance_km: float,
    traffic_level: int,
    multiple_deliveries: int,
    preparation_time_min: float,
) -> dict:
    return {
        "Distance_x_Traffic": distance_km * traffic_level,
        "Distance_per_Delivery": distance_km / (multiple_deliveries + 1),
        "Prep_to_Distance_Ratio": preparation_time_min / (distance_km + 1),
    }


def compute_rush_hour(order_hour: int) -> int:
    return int(order_hour in RUSH_HOURS)


def compute_cyclical_features(
    order_hour: int, pickup_hour: int, day_of_week: int, month: int
) -> dict:
    return {
        "Order_Hour_sin": np.sin(2 * np.pi * order_hour / 24),
        "Order_Hour_cos": np.cos(2 * np.pi * order_hour / 24),
        "Order_Hour_sin2": np.sin(2 * np.pi * 2 * order_hour / 24),
        "Order_Hour_cos2": np.cos(2 * np.pi * 2 * order_hour / 24),
        "Pickup_Hour_sin": np.sin(2 * np.pi * pickup_hour / 24),
        "Pickup_Hour_cos": np.cos(2 * np.pi * pickup_hour / 24),
        "Pickup_Hour_sin2": np.sin(2 * np.pi * 2 * pickup_hour / 24),
        "Pickup_Hour_cos2": np.cos(2 * np.pi * 2 * pickup_hour / 24),
        "Order_DayOfWeek_sin": np.sin(2 * np.pi * day_of_week / 7),
        "Order_DayOfWeek_cos": np.cos(2 * np.pi * day_of_week / 7),
        "Order_Month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "Order_Month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }


def clip_age(age: int) -> int:
    return int(np.clip(age, AGE_CLIP_MIN, AGE_CLIP_MAX))
