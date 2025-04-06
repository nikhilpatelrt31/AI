import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
# Set Streamlit page config
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")
# Title and description
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")
st.caption(
    "Timelytics is an ensemble model using XGBoost, Random Forest, and SVM to accurately forecast Order to Delivery (OTD) times. "
    "It helps identify bottlenecks and proactively improve supply chain performance."
)
# Sidebar with inputs
with st.sidebar:
    try:
        img = Image.open("./assets/supply_chain_optimisation.jpg")
        st.image(img)
    except FileNotFoundError:
        st.warning("Image not found. Please add 'supply_chain_optimisation.jpg' to the assets folder.")
    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance (in km)", value=475.35)
    submit = st.button(label="Predict Wait Time!")
# Function to load model
@st.cache_resource
def load_model():
    modelfile = "./voting_model.pkl"
    if not os.path.exists(modelfile):
        st.error("Model file 'voting_model.pkl' not found.")
        return None
    try:
        with open(modelfile, "rb") as f:
            model = pickle.load(f)
        return model
    except pickle.UnpicklingError:
        st.error("Failed to load the model. The file may be corrupted or not a pickle file.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model:\n{e}")
    return None
# Load the model
voting_model = load_model()
# Prediction logic
def waitime_predictor(purchase_dow, purchase_month, year, product_size_cm3,
                      product_weight_g, geolocation_state_customer, geolocation_state_seller, distance):
    input_array = np.array([[purchase_dow, purchase_month, year, product_size_cm3,
                             product_weight_g, geolocation_state_customer, geolocation_state_seller, distance]])
    prediction = voting_model.predict(input_array)
    return round(prediction[0])
# Main panel
with st.container():
    st.header("Output: Wait Time in Days")
    if submit and voting_model:
        with st.spinner("Predicting..."):
            try:
                prediction = waitime_predictor(
                    purchase_dow,
                    purchase_month,
                    year,
                    product_size_cm3,
                    product_weight_g,
                    geolocation_state_customer,
                    geolocation_state_seller,
                    distance
                )
                st.success(f"⏱️ Estimated Wait Time: **{prediction} days**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    # Show sample dataset
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
    st.dataframe(df)
