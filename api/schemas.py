from datetime import date, time
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class WeatherCondition(str, Enum):
    sunny = "Sunny"
    cloudy = "Cloudy"
    fog = "Fog"
    sandstorms = "Sandstorms"
    stormy = "Stormy"
    windy = "Windy"


class TrafficDensity(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    jam = "Jam"


class OrderType(str, Enum):
    snack = "Snack"
    meal = "Meal"
    drinks = "Drinks"
    buffet = "Buffet"


class VehicleType(str, Enum):
    motorcycle = "motorcycle"
    scooter = "scooter"
    electric_scooter = "electric_scooter"
    bicycle = "bicycle"


class FestivalStatus(str, Enum):
    yes = "Yes"
    no = "No"


class CityType(str, Enum):
    urban = "Urban"
    metropolitian = "Metropolitian"
    semi_urban = "Semi-Urban"


class DeliveryPredictionRequest(BaseModel):
    
    delivery_person_age: int = Field(
        ...,
        ge=18,
        le=65,
        description="Delivery person's age in years (18-65). "
        "The model was trained on drivers aged 20-40; values outside that "
        "range are clipped internally before inference (see utils.py).",
    )
    delivery_person_ratings: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Delivery person's average rating (1.0 to 5.0).",
    )

    restaurant_latitude: float | None = Field(
        None, ge=-90, le=90, description="Restaurant latitude."
    )
    restaurant_longitude: float | None = Field(
        None, ge=-180, le=180, description="Restaurant longitude."
    )
    delivery_location_latitude: float | None = Field(
        None, ge=-90, le=90, description="Delivery address latitude."
    )
    delivery_location_longitude: float | None = Field(
        None, ge=-180, le=180, description="Delivery address longitude."
    )

    distance_km: float | None = Field(
        None,
        gt=0,
        le=100,
        description="Straight-line distance in km. Provide this OR the "
        "four GPS coordinates above, not necessarily both. If both are "
        "given, distance_km takes precedence.",
    )
    
    order_date: date = Field(..., description="Date the order was placed.")
    time_ordered: time = Field(..., description="Local time the order was placed (HH:MM:SS).")
    time_order_picked: time = Field(
        ..., description="Local time the order was picked up by the driver (HH:MM:SS)."
    )

    weather_conditions: WeatherCondition
    road_traffic_density: TrafficDensity
    vehicle_condition: int = Field(
        ..., ge=0, le=3, description="Vehicle condition score, 0 (worst) to 3 (best)."
    )
    type_of_order: OrderType
    type_of_vehicle: VehicleType
    multiple_deliveries: int = Field(
        ..., ge=0, le=3, description="Number of other orders bundled in the same trip."
    )
    festival: FestivalStatus
    city: CityType


    @field_validator("delivery_person_ratings")
    @classmethod
    def round_ratings(cls, value: float) -> float:
        return round(value, 2)
    
    @model_validator(mode="after")
    def check_location_provided(self) -> "DeliveryPredictionRequest":
        has_distance = self.distance_km is not None
        has_full_gps = None not in (
            self.restaurant_latitude,
            self.restaurant_longitude,
            self.delivery_location_latitude,
            self.delivery_location_longitude,
        )

        if not has_distance and not has_full_gps:
            raise ValueError(
                "Provide either 'distance_km' or all four GPS coordinates "
                "(restaurant_latitude, restaurant_longitude, "
                "delivery_location_latitude, delivery_location_longitude)."
            )
            
        if has_full_gps and not has_distance:
            same_point = (
                self.restaurant_latitude == self.delivery_location_latitude
                and self.restaurant_longitude == self.delivery_location_longitude
            )
            if same_point:
                raise ValueError(
                    "Restaurant and delivery coordinates are identical; "
                    "this would produce a distance of 0 km."
                )

        return self

    @model_validator(mode="after")
    def check_preparation_time_is_realistic(self) -> "DeliveryPredictionRequest":
        order_minutes = self.time_ordered.hour * 60 + self.time_ordered.minute
        pickup_minutes = self.time_order_picked.hour * 60 + self.time_order_picked.minute
        prep_minutes = (pickup_minutes - order_minutes) % (24 * 60)

        if prep_minutes > 180:
            raise ValueError(
                "The gap between 'time_ordered' and 'time_order_picked' "
                f"is {prep_minutes:.0f} minutes, which is unrealistic for a "
                "food order (expected under 180 minutes)."
            )

        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "delivery_person_age": 29,
                "delivery_person_ratings": 4.6,
                "restaurant_latitude": 12.913041,
                "restaurant_longitude": 77.683237,
                "delivery_location_latitude": 13.043041,
                "delivery_location_longitude": 77.813237,
                "distance_km": None,
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
                "city": "Metropolitian",
            }
        }
    }

class DeliveryPredictionResponse(BaseModel):
    predicted_time_min: float = Field(
        ..., description="Predicted total delivery time, in minutes."
    )
    predicted_time_range_min: tuple[float, float] = Field(
        ...,
        description="Approximate (prediction - MAE, prediction + MAE) range "
        "in minutes, based on the test-set MAE of the final model.",
    )
    computed_distance_km: float = Field(
        ..., description="Distance used by the model (provided or computed from GPS)."
    )
    computed_preparation_time_min: float = Field(
        ..., description="Kitchen preparation time used by the model, in minutes."
    )
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    features: list[str]
    metrics: dict