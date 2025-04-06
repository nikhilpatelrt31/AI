#!/usr/bin/python
# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import os

# Page Config
st.set_page_config(
    page_title='Timelytics',
    page_icon=':pencil:',
    layout='wide'
)

# App Header
st.title('Timelytics: Optimize your supply chain with advanced forecasting techniques.')

st.caption('Timelytics is an ensemble model that utilizes XGBoost, Random Forests, and SVM to forecast Order to Delivery (OTD) times. By combining the strengths of these algorithms, it provides reliable OTD time predictions to optimize supply chain operations.')

st.caption('It uses historical data on order processing, production, shipping, and other variables to forecast OTD times — helping reduce lead times and improve delivery accuracy.')

# Load the model with error handling
modelfile = './voting_model.pkl'

try:
    with open(modelfile, 'rb') as f:
        voting_model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Caching the model prediction function
@st.cache_resource
def waitime_predictor(purchase_dow, purchase_month, year, product_size_cm3,
                      product_weight_g, geolocation_state_customer, geolocation_state_seller,
                      distance):
    prediction = voting_model.predict(np.array([[
        purchase_dow,
        purchase_month,
        year,
        product_size_cm3,
        product_weight_g,
        geolocation_state_customer,
        geolocation_state_seller,
        distance,
    ]]))
    return round(prediction[0])

# Sidebar inputs
with st.sidebar:
    img = Image.open('./assets/supply_chain_optimisation.jpg')
    st.image(img)
    st.header('Input Parameters')

    purchase_dow = st.number_input('Purchased Day of the Week', min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input('Purchased Month', min_value=1, max_value=12, step=1, value=1)
    year = st.number_input('Purchased Year', value=2018)
    product_size_cm3 = st.number_input('Product Size in cm³', value=9328.0)
    product_weight_g = st.number_input('Product Weight in grams', value=1800.0)
    geolocation_state_customer = st.number_input('Geolocation State of the Customer', value=10)
    geolocation_state_seller = st.number_input('Geolocation State of the Seller', value=20)
    distance = st.number_input('Distance (km)', value=475.35)
    submit = st.button('Predict')

# Main container
with st.container():
    st.header('Output: Wait Time in Days')

    if submit:
        with st.spinner(text='This may take a moment...'):
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
            st.success(f"Predicted Wait Time: {prediction} days")

    # Display a sample dataset
    sample_data = {
        'Purchased Day of the Week': [0, 3, 1],
        'Purchased Month': [6, 3, 1],
        'Purchased Year': [2018, 2017, 2018],
        'Product Size in cm^3': [37206.0, 63714.0, 54816.0],
        'Product Weight in grams': [16250.0, 7249.0, 9600.0],
        'Geolocation State Customer': [25, 25, 25],
        'Geolocation State Seller': [20, 7, 20],
        'Distance': [247.94, 250.35, 4.915],
    }

    df = pd.DataFrame(sample_data)
    st.header('Sample Dataset')
    st.dataframe(df)
