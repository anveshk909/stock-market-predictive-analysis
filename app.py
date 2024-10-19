# Example: Simple Streamlit App

import streamlit as st
import joblib
import numpy as np
import os

models_dir = 'models'
# Load the trained model
rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))

st.title('Stock Price Movement Predictor')

# Collect user input
MA10 = st.number_input('MA10')
MA50 = st.number_input('MA50')
RSI = st.number_input('RSI')
BB_upper = st.number_input('BB_upper')
BB_lower = st.number_input('BB_lower')
Daily_Return = st.number_input('Daily Return')
Lagged_Close = st.number_input('Lagged Close')
Lagged_Volume = st.number_input('Lagged Volume')

features = np.array([MA10, MA50, RSI, BB_upper, BB_lower, Daily_Return, Lagged_Close, Lagged_Volume]).reshape(1, -1)

if st.button('Predict'):
    prediction = rf_model.predict(features)[0]
    probability = rf_model.predict_proba(features)[0][1]
    st.write(f'Prediction: {"Up" if prediction == 1 else "Down"}')
    st.write(f'Probability of Up: {probability:.2f}')
