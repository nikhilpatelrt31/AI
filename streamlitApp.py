import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# ✅ Streamlit app configuration
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

# ✅ Description
st.caption(
    "Timelytics is an ensemble model using XGBoost, Random Forests, and SVM to forecast Order to Delivery (OTD) times. "
    "It helps businesses identify supply chain delays, optimize inventory, and improve customer service."
)
st.caption(
    "Using historical data on order processing, production, shipping times, and more, "
    "Timelytics generates accurate delivery forecasts to streamline operations."
)

# ✅ Model Loader using joblib
@st.cache_resource
def load_model():
    model_path = "voting_model.pkl"
    return joblib.load(model_path)

voting_model = load_model()

# ✅ Prediction function
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    features = np.array([[
        purchase_dow,
        purchase_month,
        year,
        product_size_cm3,
        product_weight_g,
        geolocation_state_customer,
        geolocation_state_seller,
        distance,
    ]])
    prediction = voting_model.predict(features)
    return round(prediction[0])

# ✅ Sidebar Inputs
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")

    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)
    submit = st.button("Predict")

# ✅ Prediction Output
with st.container():
    st.header("Output: Wait Time in Days")
    if submit:
        with st.spinner("This may take a moment..."):
            prediction = waitime_predictor(
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            )
            st.success(f"⏱️ Predicted Wait Time: {prediction} day(s)")

# ✅ Sample Dataset Display
st.header("Sample Dataset")
data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm^3": [37206.0, 63714, 54816],
    "Product Weight in grams": [16250.0, 7249, 9600],
    "Geolocation State Customer": [25, 25, 25],
    "Geolocation State Seller": [20, 7, 20],
    "Distance": [247.94, 250.35, 4.915],
}
df = pd.DataFrame(data)
st.write(df)
