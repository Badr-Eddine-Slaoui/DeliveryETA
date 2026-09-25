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


with tab1:
    col1, col2 = st.columns(2)

    with col1:
        delivery_person_age = st.number_input("Âge du livreur", 18, 65, 30)
        delivery_person_ratings = st.slider("Note du livreur", 1.0, 5.0, 4.5, 0.1)
        vehicle_condition = st.selectbox("État du véhicule", [0, 1, 2, 3], index=2)
        type_of_vehicle = st.selectbox("Type de véhicule", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
        type_of_order = st.selectbox("Type de commande", ["Snack", "Meal", "Drinks", "Buffet"])
        multiple_deliveries = st.selectbox("Nombre de livraisons groupées", [0, 1, 2, 3])
        festival = st.selectbox("Jour de festival", ["No", "Yes"])
        city = st.selectbox("Type de ville", ["Urban", "Metropolitian", "Semi-Urban"])

    with col2:
        weather_conditions = st.selectbox("Météo", ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy"])
        road_traffic_density = st.selectbox("Trafic", ["Low", "Medium", "High", "Jam"])
        order_date = st.date_input("Date de la commande", date.today())
        time_ordered = st.time_input("Heure de la commande", time(12, 0))
        time_order_picked = st.time_input("Heure de récupération", time(12, 15))

        location_mode = st.radio("Localisation", ["Distance directe", "Coordonnées GPS"])

        distance_km = None
        restaurant_latitude = None
        restaurant_longitude = None
        delivery_location_latitude = None
        delivery_location_longitude = None

        if location_mode == "Distance directe":
            distance_km = st.number_input("Distance (km)", 0.1, 100.0, 5.0)
        else:
            restaurant_latitude = st.number_input("Latitude restaurant", -90.0, 90.0, 12.9)
            restaurant_longitude = st.number_input("Longitude restaurant", -180.0, 180.0, 77.6)
            delivery_location_latitude = st.number_input("Latitude client", -90.0, 90.0, 13.0)
            delivery_location_longitude = st.number_input("Longitude client", -180.0, 180.0, 77.8)

    if st.button("Prédire le temps de livraison"):
        payload = {
            "delivery_person_age": delivery_person_age,
            "delivery_person_ratings": delivery_person_ratings,
            "restaurant_latitude": restaurant_latitude,
            "restaurant_longitude": restaurant_longitude,
            "delivery_location_latitude": delivery_location_latitude,
            "delivery_location_longitude": delivery_location_longitude,
            "distance_km": distance_km,
            "order_date": order_date.isoformat(),
            "time_ordered": time_ordered.isoformat(),
            "time_order_picked": time_order_picked.isoformat(),
            "weather_conditions": weather_conditions,
            "road_traffic_density": road_traffic_density,
            "vehicle_condition": vehicle_condition,
            "type_of_order": type_of_order,
            "type_of_vehicle": type_of_vehicle,
            "multiple_deliveries": multiple_deliveries,
            "festival": festival,
            "city": city,
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Temps de livraison estimé : {result['predicted_time_min']} minutes")
                low, high = result["predicted_time_range_min"]
                st.write(f"Intervalle probable : {low} à {high} minutes")
                st.write(f"Distance utilisée : {result['computed_distance_km']} km")
                st.write(f"Temps de préparation estimé : {result['computed_preparation_time_min']} min")
            else:
                st.error(response.json().get("detail", "Erreur lors de la prédiction"))
        except requests.exceptions.ConnectionError:
            st.error(f"Impossible de contacter l'API. Vérifiez qu'elle est lancée sur {API_URL}.")

with tab2:
    df = load_data()

    st.plotly_chart(px.histogram(df, x="Time_taken(min)", nbins=30, title="Distribution du temps de livraison"))

    st.plotly_chart(px.box(df, x="Road_traffic_density", y="Time_taken(min)", title="Temps de livraison selon le trafic"))

    st.plotly_chart(px.box(df, x="Weatherconditions", y="Time_taken(min)", title="Temps de livraison selon la météo"))

    st.plotly_chart(px.scatter(df, x="Distance_km", y="Time_taken(min)", title="Temps de livraison selon la distance"))

    st.plotly_chart(px.box(df, x="City", y="Time_taken(min)", title="Temps de livraison selon le type de ville"))

with tab3:
    metadata = load_metadata()
    metrics = metadata["metrics"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE (test)", f"{metrics['test_mae_min']:.2f} min")
    col2.metric("RMSE (test)", f"{metrics['test_rmse_min']:.2f} min")
    col3.metric("R² (test)", f"{metrics['test_r2']:.3f}")
    col4.metric("R² ajusté (test)", f"{metrics['test_adjusted_r2']:.3f}")

    st.write(f"Le modèle {metadata['model_name']} se trompe en moyenne de {metrics['test_mae_min']:.1f} minutes sur des données jamais vues.")

    comparison_df = pd.DataFrame({
        "Ensemble": ["Entraînement", "Test"],
        "RMSE (min)": [metrics["train_rmse_min"], metrics["test_rmse_min"]],
        "R²": [metrics["train_r2"], metrics["test_r2"]],
    })

    st.plotly_chart(px.bar(comparison_df, x="Ensemble", y="RMSE (min)", title="RMSE entraînement vs test"))
    st.plotly_chart(px.bar(comparison_df, x="Ensemble", y="R²", title="R² entraînement vs test"))

    st.write(f"Écart RMSE train/test : {metrics['rmse_gap_min']:.2f} min")
    st.write(f"Écart R² train/test : {metrics['r2_gap']:.3f}")