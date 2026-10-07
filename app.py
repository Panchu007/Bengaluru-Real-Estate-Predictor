"""Streamlit app: Bengaluru House Price Predictor."""

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------- Paths ----------
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_FILE = ROOT / "data" / "engineered_data.csv"


# ---------- Cached loaders ----------
@st.cache_resource
def load_artifacts():
    """Load the trained model and the list of training columns."""
    with open(MODELS_DIR / "bengaluru_house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "columns.json", "r") as f:
        columns = json.load(f)["data_columns"]
    return model, columns


@st.cache_data
def load_locations():
    """Unique locations seen during training, sorted alphabetically."""
    df = pd.read_csv(DATA_FILE, usecols=["location"])
    return sorted(df["location"].dropna().unique().tolist())


# ---------- Load once ----------
model, columns = load_artifacts()
locations = load_locations()

# ---------- Page setup ----------
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Bengaluru House Price Predictor")
st.write(
    "Estimate the price of a house in Bengaluru based on location, "
    "size, bedrooms, and bathrooms."
)
st.divider()

# ---------- Input widgets ----------
col_left, col_right = st.columns(2)

with col_left:
    location = st.selectbox("Location", options=locations, index=0)
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=20, value=2, step=1)

with col_right:
    total_sqft = st.number_input(
        "Total Square Feet", min_value=300, max_value=15000, value=1200, step=50
    )
    bath = st.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)

st.divider()

# ---------- Prediction ----------
if st.button("Predict Price", type="primary", use_container_width=True):
    # Start with all features = 0
    input_data = {col: 0 for col in columns}

    # Fill in the numeric features
    if "total_sqft" in input_data:
        input_data["total_sqft"] = total_sqft
    if "bath" in input_data:
        input_data["bath"] = bath
    if "bhk" in input_data:
        input_data["bhk"] = bhk

    # One-hot encode the location
    loc_key = location.lower()
    if loc_key in input_data:
        input_data[loc_key] = 1
    elif "other" in input_data:
        input_data["other"] = 1
        st.info(f"Location '{location}' wasn't seen during training — using 'other'.")

    # Build the feature row in the exact training column order
    X = pd.DataFrame([input_data], columns=columns)

    # Predict
    price_lakhs = float(model.predict(X)[0])
    price_rupees = price_lakhs * 100_000

    st.success(f"### Estimated Price: ₹ {price_lakhs:,.2f} lakhs")
    st.caption(f"≈ ₹ {price_rupees:,.0f}")

    # Small breakdown
    with st.expander("See input summary"):
        st.write(
            {
                "Location": location,
                "Total sqft": total_sqft,
                "BHK": bhk,
                "Bathrooms": bath,
            }
        )

st.divider()
st.caption(
    "Model: Linear Regression (R² = 0.7554) · "
    "Dataset: Bengaluru House Price Data · "
    "Built with Streamlit"
)
