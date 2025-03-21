import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model, scaler, and label encoder
with open(r'C:\Users\91930\Desktop\quality\xgboost_air_quality.pkl', "rb") as model_file:
    model = pickle.load(model_file)
with open(r'C:\Users\91930\Desktop\quality\scaler.pkl', "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open(r'C:\Users\91930\Desktop\quality\label_encoder.pkl', "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Set up page configuration
st.set_page_config(page_title="Air Quality Prediction", page_icon="🌍", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main { background: linear-gradient(135deg, #d7e1ec, #f5f7fa); }
        .stButton>button { background-color: #1565c0; color: white; border-radius: 10px; }
        .stButton>button:hover { background-color: #0d47a1; }
        .result-box { background: #1565c0; color: white; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Section
st.markdown("""
    <h1 style='text-align: center; color: #1565c0;'>🌍 Air Quality Classification using XGBoost</h1>
    <p style='text-align: center; font-size: 1.2em; color: #666;'>Predict air quality based on environmental factors</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input("🌡 Temperature (°C)", min_value=-30.0, max_value=500.0, value=25.0, step=0.1)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    pm25 = st.number_input("🌀 PM2.5 Concentration (µg/m³)", min_value=0.0, max_value=500.0, value=35.0, step=0.1)

with col2:
    pm10 = st.number_input("🌫 PM10 Concentration (µg/m³)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
    no2 = st.number_input("🛑 NO2 Concentration (ppb)", min_value=0.0, max_value=500.0, value=30.0, step=0.1)
    so2 = st.number_input("⚠ SO2 Concentration (ppb)", min_value=0.0, max_value=500.0, value=10.0, step=0.1)

with col3:
    co = st.number_input("🔥 CO Concentration (ppm)", min_value=0.0, max_value=50.0, value=0.8, step=0.1)
    industrial_proximity = st.number_input("🏭 Proximity to Industrial Areas (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    population_density = st.number_input("👥 Population Density (people/km²)", min_value=0, max_value=1000000, value=1000, step=100)

st.markdown("---")

# Prediction function
def predict_air_quality():
    input_data = np.array([[temperature, humidity, pm25, pm10, no2, so2, co, industrial_proximity, population_density]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

# Predict button
if st.button("🚀 Predict Air Quality"):
    result = predict_air_quality()
    
    # Define colors and icons
    air_quality_styles = {
        "Good": ("🟢", "#4CAF50"),
        "Moderate": ("🟡", "#FFC107"),
        "Poor": ("🟠", "#FF9800"),
        "Hazardous": ("🔴", "#F44336")
    }
    icon, color = air_quality_styles.get(result, ("❓", "#607D8B"))
    
    # Display prediction results in a styled box
    st.markdown(f"""
        <div class='result-box' style='background-color: {color};'>
            <h2>{icon} Predicted Air Quality: {result}</h2>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 1.1em; color: #666;'>📌 Developed using <b>XGBoost</b> & <b>Streamlit</b> | 🚀 <i>Machine Learning Model for Air Quality Classification</i></p>", unsafe_allow_html=True)