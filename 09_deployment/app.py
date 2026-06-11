import streamlit as st
import joblib
import pandas as pd

# ── App title (must be first Streamlit command) ──
st.set_page_config(page_title="Home Price Estimator", layout="centered")
st.title("Home Price Estimator")

# ── Block 1: Load model, scaler, and column names ──
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ── Block 2: Get input from user ──
with st.sidebar:
    st.header("Property Details")

    sqft = st.slider("Square Footage", 500, 5000, 1500)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    lot_size = st.number_input("Lot Size (sqft)", 1000, 50000, 5000)

    neighborhood = st.selectbox(
        "Neighborhood",
        ["Downtown", "Suburbs", "Rural", "Waterfront"]
    )

    has_fireplace = st.checkbox("Fireplace")

# ── Block 3: Prepare data and predict ──

# Step 1: Build DataFrame
input_data = {
    "sqft": sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "lot_size": lot_size,
    "neighborhood": neighborhood,
    "has_fireplace": int(has_fireplace),
}
input_df = pd.DataFrame([input_data])

# Step 2: Create dummy variables matching training
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# Step 3: Scale
input_scaled = scaler.transform(input_encoded)

# Step 4: Predict
prediction = model.predict(input_scaled)[0]

# ── Block 4: Display result ──
st.metric("Estimated Price", f"${prediction:,.2f}")