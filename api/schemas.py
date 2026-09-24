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
