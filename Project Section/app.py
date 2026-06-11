import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Home Price Estimator", layout="centered")
st.title("Home Price Estimator")

# Block 1: Load model and columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Block 2: User inputs
with st.sidebar:
    st.header("Property Details")
    house_size = st.number_input("House Size", 500, 10000, 1500)
    bedrooms = st.slider("Bedrooms", 1, 8, 3)
    baths = st.slider("Total Baths", 1, 6, 2)

    lot_size = st.number_input("Lot Size", 1000, 100000, 5000)
    latitude = st.number_input("Latitude", 47.0, 48.0, 47.25)

    fireplace = st.checkbox("Fireplace")
    waterfront = st.checkbox("Waterfront")
    air_conditioning = st.checkbox("Air Conditioning")
    new_construction = st.checkbox("New Construction")

    property_condition = st.selectbox(
        "Property Condition",
        ["Average", "Good", "V.Good", "Unk", "Fair", "Fixer", "Remod"]
    )
# Block 3: Prepare input
log_lot_size = np.log(lot_size)

input_data = {
    "house_size": house_size,
    "Latitude": latitude,
    "Total Baths": baths,
    "log_lot_size": log_lot_size,
    "New Construction": "Yes" if new_construction else "No",
    "Air Conditioning Y/N": int(air_conditioning),
    "Fireplace Y/N": int(fireplace),
    "Waterfront Y/N": int(waterfront),
    "Total Bedrooms": bedrooms,
    "Property Condition": property_condition
}

input_df = pd.DataFrame([input_data])

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# Block 4: Predict and display
input_scaled = scaler.transform(input_encoded)

log_prediction = model.predict(input_scaled)[0]
price_prediction = np.exp(log_prediction)

st.subheader("Estimated Home Price")
st.metric("Predicted Price", f"${price_prediction:,.2f}")