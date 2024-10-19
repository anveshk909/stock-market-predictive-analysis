# Stock Market Predictive Analysis Capstone Project

## Overview

This project aims to predict the movement of stock prices for major companies, specifically Apple (AAPL), Google (GOOGL), and Amazon (AMZN), using machine learning models. We used a combination of traditional machine learning models and advanced deep learning techniques to determine whether the closing price of a stock would move up or down. This project was developed as part of a capstone project to demonstrate proficiency in data collection, feature engineering, modeling, and analysis using Python and machine learning tools.

## Problem Statement

Stock price prediction is a challenging task due to the high volatility and multiple factors affecting market movements. Our goal was to use historical price data and technical indicators to predict the direction of future price movements. The predictions are aimed at helping investors make informed decisions about whether to buy or sell stocks.

## Data Collection

We collected historical stock data for Apple, Google, and Amazon from Yahoo Finance, covering the period from January 1, 2018, to December 31, 2023. The data included daily open, high, low, close, and volume values for each stock.

## Feature Engineering

Several technical indicators were calculated to enhance the predictive power of our models, including:

- **Moving Averages (MA10, MA50)**: To identify short-term and long-term trends.
- **Relative Strength Index (RSI)**: To identify overbought or oversold conditions.
- **Bollinger Bands**: To capture price volatility.
- **Daily Return**: Percentage change in the closing price to track price movements.
- **Lagged Features**: Lagged close and volume features to help the models learn from previous values.

## Exploratory Data Analysis (EDA)

- **Price Trends**: Visualized the closing price trends of AAPL, GOOGL, and AMZN.
- **Correlation Analysis**: Analyzed the correlation between different features, such as moving averages, returns, and volume.
- **Distribution of Returns**: Analyzed the distribution of daily returns for each stock to understand volatility.

## Modeling

We used a combination of traditional machine learning models and deep learning models to predict stock price movements:

### Traditional Machine Learning Models

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Decision Tree Classifier**
4. **Support Vector Classifier (SVC)**
5. **Gradient Boosting Classifier**

These models were trained using features derived from historical stock data, and their performance was evaluated using metrics such as **accuracy**, **AUC-ROC**, **precision**, and **recall**.

### Deep Learning Model

- **Long Short-Term Memory (LSTM)**: A type of recurrent neural network used to capture sequential dependencies in time-series data. The LSTM model was trained to predict the closing price trend based on historical data, using 50-day sequences of data points.

## Evaluation and Findings

- **Model Performance**: The **Random Forest** and **Gradient Boosting** models performed best with respect to AUC-ROC, indicating their robustness in predicting stock movements.
- **LSTM Results**: The LSTM model effectively captured sequential patterns in the data and showed promise for time-series forecasting, although it requires further tuning for optimal performance.
- **Feature Importance**: **SHAP analysis** was used to interpret the feature importance of the Random Forest model. Moving Averages (MA10, MA50) and RSI were identified as the most influential features.

## Summary of Insights

- **Random Forest and Gradient Boosting** performed the best in terms of predictive accuracy and interpretability.
- **LSTM** demonstrated the ability to learn sequential dependencies, making it a promising approach for future work.
- **Key Features**: Moving Averages and RSI were the most impactful features, highlighting their importance in predicting price movements.

## Recommendations and Next Steps

- **Sentiment Analysis**: Future work could include incorporating sentiment analysis from news headlines and social media to better capture market sentiment.
- **Economic Indicators**: Integrate macroeconomic indicators (e.g., interest rates, GDP growth) for a more comprehensive model.
- **Hyperparameter Tuning**: Further optimize hyperparameters for the LSTM model to improve its performance.
- **Feature Expansion**: Explore additional technical indicators and composite features to boost model accuracy.

## How to Access the Technical Analysis

The complete technical analysis, including all the data collection, preprocessing, feature engineering, modeling, evaluation, and visualization steps, is available in the [Jupyter Notebook here](./Notebooks/Stock_Market_Predictive_Analysis.ipynb).

## Requirements

To run the Jupyter Notebook, you will need the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `yfinance`
- `scikit-learn`
- `shap`
- `keras`
- `tensorflow`
- `joblib`

To install the required libraries, run:

```sh
pip install -r requirements.txt
