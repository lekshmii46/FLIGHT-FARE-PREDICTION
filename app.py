import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model, Scaler, and Feature Names
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('model_features.pkl')
    return model, scaler, features

model, scaler, feature_names = load_assets()

# Page Config
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d2ff;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Containers / Cards */
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
    
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="stForm"]:hover {
        transform: translateY(-5px);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 10px 24px;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.6);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        color: #00ea8d;
        text-shadow: 0 0 10px rgba(0, 234, 141, 0.5);
    }
    
    /* Inputs */
    .stSelectbox label, .stNumberInput label, .stDateInput label, .stTimeInput label {
        color: #a8b2d1;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.title("✈️ SkyPredict: Intelligent Flight Fare Estimation")
st.markdown("### Powered by Ensemble Learning (Random Forest)")

col1, col2 = st.columns([2, 1])

with col1:
    with st.form("prediction_form"):
        st.subheader("Flight Details")
        
        c1, c2 = st.columns(2)
        with c1:
            airline = st.selectbox("Airline", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia'])
            source = st.selectbox("Source", ['Bangalore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
            date_journey = st.date_input("Date of Journey")
            duration_hrs = st.number_input("Duration (Hours)", min_value=1, max_value=50, value=2)
            
        with c2:
            destinations = ['New Delhi', 'Bangalore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad']
            destinations = [d for d in destinations if d != source and not (source == 'Delhi' and d == 'New Delhi')]
            destination = st.selectbox("Destination", destinations)
            total_stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
            dep_time = st.time_input("Departure Time")
            duration_mins = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=30)
            
        submit_button = st.form_submit_button("Predict Fare 🚀")

with col2:
    st.markdown("### 📊 About the Model")
    st.info("""
    This prediction engine uses a **Random Forest Regressor**, a powerful Ensemble Learning technique that builds multiple decision trees and merges them together for a more accurate and stable prediction.
    
    **Features analyzed:**
    - Airline Pricing Trends
    - Route Popularity
    - Time of Departure
    - Journey Duration
    - Stopovers
    """)
    
    if submit_button:
        # Preprocess input data to match model features
        stops_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
        
        input_data = {
            "Total_Stops": stops_map[total_stops],
            "Journey_day": date_journey.day,
            "Journey_month": date_journey.month,
            "Dep_hour": dep_time.hour,
            "Dep_min": dep_time.minute,
            "Duration_hours": duration_hrs,
            "Duration_mins": duration_mins
        }
        
        # Initialize dictionary with zeros for one-hot encoding
        processed_input = {col: 0 for col in feature_names}
        
        # Update known numeric features
        for k, v in input_data.items():
            if k in processed_input:
                processed_input[k] = v
                
        # Update one-hot features
        if f"{airline}" in processed_input: processed_input[f"{airline}"] = 1
        if f"{source}" in processed_input: processed_input[f"{source}"] = 1
        if f"{destination}" in processed_input: processed_input[f"{destination}"] = 1
            
        # Create DataFrame
        input_df = pd.DataFrame([processed_input])
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        st.success("Prediction Successful!")
        st.metric(label="Estimated Fare", value=f"₹ {prediction:,.2f}")
