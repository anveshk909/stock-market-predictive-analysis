# Example: Improved Streamlit App

import streamlit as st
import joblib
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
models_dir = os.getenv('MODELS_DIR', 'models')
model_path = os.path.join(models_dir, 'random_forest_model.pkl')

try:
    rf_model = joblib.load(model_path)
    logging.info(f"Model loaded from {model_path}")
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}")
    logging.error(f"Model file not found at {model_path}")
    st.stop()

st.title('Stock Price Movement Predictor')

# Collect user input with validation
MA10 = st.number_input('MA10 (10-day Moving Average)', min_value=0.0, max_value=1000.0, step=0.01, help="10-day moving average of the stock price")
MA50 = st.number_input('MA50 (50-day Moving Average)', min_value=0.0, max_value=1000.0, step=0.01, help="50-day moving average of the stock price")
RSI = st.number_input('RSI (Relative Strength Index)', min_value=0.0, max_value=100.0, step=0.01, help="Relative Strength Index of the stock")
BB_upper = st.number_input('BB_upper (Bollinger Band Upper)', min_value=0.0, max_value=1000.0, step=0.01, help="Upper Bollinger Band value")
BB_lower = st.number_input('BB_lower (Bollinger Band Lower)', min_value=0.0, max_value=1000.0, step=0.01, help="Lower Bollinger Band value")
Daily_Return = st.number_input('Daily Return', min_value=-1.0, max_value=1.0, step=0.01, help="Daily return of the stock")
Lagged_Close = st.number_input('Lagged Close', min_value=0.0, max_value=1000.0, step=0.01, help="Previous day's closing price")
Lagged_Volume = st.number_input('Lagged Volume', min_value=0.0, max_value=1e9, step=1.0, help="Previous day's trading volume")

features = np.array([MA10, MA50, RSI, BB_upper, BB_lower, Daily_Return, Lagged_Close, Lagged_Volume]).reshape(1, -1)

if st.button('Predict'):
    try:
        prediction = rf_model.predict(features)[0]
        probability = rf_model.predict_proba(features)[0][1]
        st.write(f'Prediction: {"Up" if prediction == 1 else "Down"}')
        st.write(f'Probability of Up: {probability:.2f}')
    except Exception as e:
        st.error("An error occurred during prediction.")
        logging.error(f"Error during prediction: {e}")