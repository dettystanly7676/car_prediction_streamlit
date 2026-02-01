import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page Config for a better browser tab title and icon
st.set_page_config(page_title="CarValue Pro", page_icon="🚗", layout="wide")

# Custom CSS to make the button look better
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data["model"]
model_columns = data["columns"]

st.title("🚗 Car Selling Price Predictor")
st.markdown("---")

# Use Columns to organize the UI so it's not one long list
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Details")
    present_price = st.number_input("Showroom Price (Lakhs)", 0.1, 50.0, 5.0)
    kms_driven = st.number_input("Total Kilometers Driven", 0, 300000, 10000)
    year = st.slider("Year of Purchase", 2000, 2024, 2016)
    owner = st.radio("Previous Owners", [0, 1, 3], horizontal=True)

with col2:
    st.subheader("Specifications")
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

st.markdown("---")

# Centered Predict Button
if st.button("Calculate Estimated Value"):
    age = 2026 - year # Updated to current year
    
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    input_df['Present_Price'] = present_price
    input_df['Kms_Driven'] = kms_driven
    input_df['Owner'] = owner
    input_df['Age'] = age
    
    if f"Fuel_Type_{fuel_type}" in input_df.columns: input_df[f"Fuel_Type_{fuel_type}"] = 1
    if f"Seller_Type_{seller_type}" in input_df.columns: input_df[f"Seller_Type_{seller_type}"] = 1
    if f"Transmission_{transmission}" in input_df.columns: input_df[f"Transmission_{transmission}"] = 1

    prediction = model.predict(input_df)
    
    # Large, clear result display
    st.balloons()
    st.metric(label="Estimated Selling Price", value=f"₹{round(prediction[0], 2)} Lakhs")