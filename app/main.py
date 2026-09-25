from datetime import date, time
import json
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()

API_URL = os.getenv("FASTAPI_URL")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "food_delivery_features.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "hist_gradient_optimized_metadata.json"

st.title("Prédiction du temps de livraison")

tab1, tab2, tab3 = st.tabs(["Prédiction", "Visualisation des données", "Performance du modèle"])


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
