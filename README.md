

# Stock Market Predictive Analysis Capstone Project

## Table of Contents

1. [Overview](#overview)
2. [Executive Summary](#executive-summary)
3. [Problem Statement](#problem-statement)
4. [Technical Report](#technical-report)
    - [Data Collection](#data-collection)
    - [Feature Engineering](#feature-engineering)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Modeling](#modeling)
    - [Model Evaluation](#model-evaluation)
    - [Model Interpretation](#model-interpretation)
5. [Non-Technical Report](#non-technical-report)
    - [Introduction](#introduction)
    - [Methodology](#methodology)
    - [Results](#results)
    - [Key Findings](#key-findings)
    - [Recommendations](#recommendations)
    - [Conclusion](#conclusion)
6. [Technical Implementation](#technical-implementation)
    - [Flask API](#flask-api)
    - [Streamlit App](#streamlit-app)
7. [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Cloning the Repository](#cloning-the-repository)
    - [Creating and Activating a Virtual Environment](#creating-and-activating-a-virtual-environment)
    - [Installing Dependencies](#installing-dependencies)
    - [Running the Jupyter Notebook](#running-the-jupyter-notebook)
    - [Setting Up and Running the Flask API](#setting-up-and-running-the-flask-api)
    - [Setting Up and Running the Streamlit App](#setting-up-and-running-the-streamlit-app)
8. [Repository Structure](#repository-structure)
9. [Model Saving](#model-saving)
10. [Final Checklist](#final-checklist)

---

## Overview

This project aims to predict the daily movement of stock prices for three major technology companies: **Apple (AAPL)**, **Google (GOOGL)**, and **Amazon (AMZN)**. By leveraging historical stock data and various technical indicators, the project employs a combination of machine learning and deep learning models to forecast whether a stock's closing price will increase the next trading day. The analysis encompasses data collection, feature engineering, exploratory data analysis (EDA), model training, evaluation, interpretation, and final recommendations. Additionally, the project includes the development of a Flask API for model predictions and a Streamlit app for user-friendly interactions.

---

## Executive Summary

Accurately predicting stock price movements is a significant challenge in the financial industry due to the market's inherent volatility and the multitude of influencing factors. This project explores the efficacy of different machine learning classifiers and a deep learning LSTM model in forecasting daily stock price increases for **Apple (AAPL)**, **Google (GOOGL)**, and **Amazon (AMZN)**. The findings indicate that ensemble models, particularly **Random Forest** and **Gradient Boosting**, outperform individual classifiers, achieving the highest **AUC-ROC** scores. The **Stacking Classifier**, which combines multiple models, further enhances predictive performance. The **LSTM** model, while promising in capturing temporal dependencies, requires additional optimization to match the ensemble methods' performance. **SHAP** analysis reveals that technical indicators like **Moving Averages** and **RSI** are the most influential features driving predictions. These insights can aid investors in making informed decisions, potentially enhancing returns and mitigating risks.

---

## Problem Statement

**Research Question:**

*Can machine learning and deep learning models accurately predict the daily movement of stock prices for Apple (AAPL), Google (GOOGL), and Amazon (AMZN) based on historical data and technical indicators?*

Accurately predicting stock price movements is a fundamental challenge in the financial industry, offering significant implications for investment strategies and portfolio management. Traditional analysis methods may not fully capture the intricate patterns and indicators influencing stock prices. This project aims to harness machine learning and deep learning techniques to enhance prediction accuracy, providing actionable insights for investors and financial analysts.

---

## Technical Report

For a detailed walkthrough of the analysis, including data preprocessing, modeling, evaluation, and interpretation, please refer to the [Stock_Market_Predictive_Analysis.ipynb](notebooks/Stock_Market_Predictive_Analysis.ipynb) Jupyter Notebook.

### Data Collection

Historical stock data for **Apple (AAPL)**, **Google (GOOGL)**, and **Amazon (AMZN)** was collected using the `yfinance` Python library, which retrieves data directly from Yahoo Finance. The dataset spans from **January 1, 2018**, to **December 31, 2023**, encompassing:

- **Price Data:** Open, Close, High, Low
- **Trading Volume**

This comprehensive dataset provides a solid foundation for analyzing stock performance and identifying patterns relevant to price movements.

### Feature Engineering

To enrich the dataset and capture market trends, several technical indicators were computed:

- **Moving Averages (MA10, MA50):** Simple moving averages over 10 and 50 days. These indicators help identify the trend direction and potential reversal points.
  
- **Relative Strength Index (RSI):** Momentum oscillator measuring the speed and change of price movements. RSI values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.
  
- **Bollinger Bands (BB_upper, BB_lower):** Volatility bands placed above and below a moving average, typically 20-day. They help identify periods of high or low volatility.
  
- **Daily Return:** Percentage change in closing price from the previous day, providing insight into daily volatility and momentum.
  
- **Lagged Features:** Previous day's closing price and volume, capturing recent performance and trading activity trends.

These features provide insights into price trends, momentum, and volatility, which are crucial for predicting future price movements.

### Exploratory Data Analysis (EDA)

Comprehensive EDA was conducted to understand data distributions, trends, and correlations:

- **Closing Price Trends:** Visualized over the study period to identify patterns, trends, and potential anomalies.
  
- **Correlation Heatmaps:** Assessed relationships between technical indicators to identify multicollinearity and feature interdependencies.
  
- **Distribution of Daily Returns:** Analyzed to understand the volatility and skewness of returns, aiding in model selection and evaluation.
  
- **Box Plots and Histograms:** Used to visualize the distribution of individual features and identify outliers.
  
- **Time Series Decomposition:** Decomposed the closing price series into trend, seasonality, and residuals to better understand underlying patterns.

**Key EDA Findings:**

- **Trend Analysis:** Significant positive correlations were observed between MA10 and MA50, indicating trending behavior. Periods of price consolidation were also identified.
  
- **Momentum Indicators:** RSI showed a strong inverse relationship with overbought and oversold conditions, correlating with price reversals. High RSI values often preceded price declines, while low RSI values preceded price increases.
  
- **Volatility Patterns:** Bollinger Bands and ATR highlighted periods of high volatility, essential for risk management. Narrow bands indicated low volatility periods, while wide bands indicated high volatility.
  
- **Return Distribution:** Daily returns exhibited a normal distribution with occasional spikes corresponding to market events, informing the choice of evaluation metrics. The presence of fat tails suggests that extreme events, though rare, have a significant impact on stock prices.

### Modeling

Multiple classification models were implemented to predict stock price increases. The target variable was binary, indicating whether the stock's closing price would go **Up** (1) or **Down** (0) the next day.

1. **Random Forest Classifier**
2. **Gradient Boosting Classifier**
3. **Support Vector Classifier (SVC)**
4. **Decision Tree Classifier**
5. **Logistic Regression**
6. **Stacking Classifier (Ensemble of the above models)**

Additionally, a **Long Short-Term Memory (LSTM)** neural network was developed to capture temporal dependencies in the data.

**Hyperparameter Tuning:**

- **Grid Search Cross-Validation:** Employed to optimize model parameters, enhancing performance and ensuring robustness. Parameters such as the number of trees in Random Forest, learning rate in Gradient Boosting, kernel types in SVC, and regularization strengths in Logistic Regression were fine-tuned.

**Handling Class Imbalance:**

- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset, preventing model bias towards the majority class and improving the ability to detect upward movements.

### Model Evaluation

Models were evaluated using the following metrics:

- **Accuracy:** Proportion of correct predictions, indicating overall effectiveness.
  
- **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to distinguish between classes across different threshold settings, providing insight into the trade-off between sensitivity and specificity.

**Evaluation Strategy:**

- **Train-Test Split:** Data was split into 80% for training and 20% for testing to assess model performance on unseen data.
  
- **Cross-Validation:** 5-fold cross-validation was employed to ensure model stability and reduce variance in performance estimates.
  
- **Confusion Matrix:** Analyzed to understand the distribution of true positives, true negatives, false positives, and false negatives.

**Performance Highlights:**

- **Random Forest** and **Gradient Boosting** classifiers achieved the highest **AUC-ROC** scores, indicating strong predictive capabilities.
  
- **Stacking Classifier** further enhanced performance by leveraging the strengths of multiple models, achieving a balanced trade-off between bias and variance.
  
- **SVC** and **Logistic Regression** demonstrated robust performance, suitable for scenarios requiring interpretability and faster computation.
  
- **Decision Tree** provided moderate performance, benefiting significantly from ensemble techniques to improve stability and accuracy.
  
- **LSTM** showed potential in capturing temporal patterns inherent in stock data but lagged behind ensemble methods, suggesting the need for further optimization and potentially more complex architectures.

### Model Interpretation

**SHAP (SHapley Additive exPlanations)** was employed to interpret feature importance across different models, providing insights into which technical indicators most significantly influence predictions.

**Interpretation Highlights:**

- **Moving Averages (MA10, MA50):** Highest impact on predicting stock price increases, reflecting trend strength.
  
- **RSI:** Significant for identifying overbought or oversold conditions, aiding in reversal predictions.
  
- **Daily Return and Lagged Features:** Important for capturing recent performance and momentum.
  
- **Bollinger Bands:** Provided insights into market volatility and potential breakout or reversal points.

**SHAP Summary Plots:** Illustrated the overall importance and impact of each feature across the dataset, highlighting consistent patterns in feature influence.

---

## Non-Technical Report

### Introduction

Predicting stock price movements is a critical task for investors seeking to maximize returns and minimize risks. This project explores the use of advanced analytical techniques to forecast whether a stock's closing price will increase the next day. Focusing on three tech giants—Apple, Google, and Amazon—we aim to develop models that can provide actionable insights for investment decisions.

### Methodology

1. **Data Collection:** We gathered five years of historical stock data for AAPL, GOOGL, and AMZN, including daily prices and trading volumes. This data forms the basis for analyzing stock performance and identifying trends.
   
2. **Feature Engineering:** By calculating various technical indicators such as Moving Averages, RSI, Bollinger Bands, and Daily Returns, we enhanced the dataset to capture market trends and momentum, essential for accurate predictions.
   
3. **Exploratory Data Analysis (EDA):** Through visualizations and statistical analysis, we explored the relationships between different technical indicators and stock price movements, identifying key patterns and anomalies.
   
4. **Model Development:** Utilizing a suite of machine learning models, including ensemble methods and deep learning architectures, we trained classifiers to predict stock price movements based on the engineered features.
   
5. **Evaluation:** Models were assessed based on their accuracy and ability to distinguish between positive and negative movements, ensuring that the predictions are both reliable and actionable.
   
6. **Interpretation:** Through interpretative tools like SHAP, we identified the most influential factors driving the predictions, enhancing the transparency and trustworthiness of the models.

### Results

Our analysis revealed that ensemble models like Random Forest and Gradient Boosting achieved the highest predictive performance, effectively identifying days when stock prices would rise. Key technical indicators such as Moving Averages and RSI were instrumental in these predictions. The deep learning model, while promising in capturing temporal patterns, showed potential but requires further refinement to match the performance of ensemble methods.

### Key Findings

- **Effective Predictors:** Technical indicators such as Moving Averages (MA10, MA50) and RSI are strong predictors of stock price movements, aligning with financial theories about market trends and momentum.
  
- **Model Performance:** Ensemble models outperform individual classifiers, offering robust and reliable predictions. The Stacking Classifier, which combines multiple models, achieved the highest AUC-ROC score, indicating enhanced predictive capabilities.
  
- **Interpretability:** Understanding which factors influence predictions enhances trust and facilitates informed decision-making. SHAP analysis highlighted the critical role of Moving Averages and RSI in driving predictions.
  
- **Deep Learning Potential:** While the LSTM model demonstrated the ability to capture temporal dependencies, its performance was slightly below ensemble methods, suggesting that further optimization and potentially more complex architectures are needed.

### Recommendations

1. **Model Enhancement:** Further optimize existing models and explore additional features to improve accuracy. Techniques like Randomized Search or Bayesian Optimization can be employed for more efficient hyperparameter tuning.
   
2. **Data Integration:** Incorporate sentiment analysis from news articles and social media platforms to capture market sentiments, potentially improving predictive accuracy by reflecting investor psychology and external factors influencing stock prices.
   
3. **Advanced Deep Learning Architectures:** Experiment with more complex architectures such as Bidirectional LSTMs or Attention Mechanisms to better capture temporal dependencies and long-term trends in the data, aiming to bridge the performance gap with ensemble models.
   
4. **Deployment:** Develop a user-friendly dashboard or API for real-time predictions to facilitate practical trading applications. Real-time deployment allows investors to input current data and receive immediate predictions, aiding in timely decision-making.
   
5. **Continuous Monitoring:** Implement systems for ongoing model evaluation and retraining to adapt to evolving market conditions. Financial markets are dynamic, and continuous monitoring ensures that models remain relevant and accurate over time.
   
6. **Comprehensive Documentation:** Maintain thorough documentation and use version control effectively to ensure reproducibility and facilitate collaboration. Clear documentation aids in understanding model decisions and methodologies, crucial for both technical and non-technical stakeholders.

### Conclusion

This project demonstrates the potential of machine learning and deep learning in forecasting stock price movements. By leveraging technical indicators and advanced models, investors can gain valuable insights to inform their investment strategies, ultimately aiming to enhance returns and manage risks more effectively. While ensemble models have shown superior performance, there is room for improvement in deep learning approaches, especially with further optimization and integration of additional data sources. The development of a Flask API and Streamlit app further enhances the project's applicability, providing tools for real-time predictions and user-friendly interactions.

---

## Technical Implementation

### Flask API

The Flask API enables external applications to interact with the trained **Random Forest** model for making predictions. This facilitates seamless integration with other systems, such as automated trading platforms or analytical dashboards.

**Key Features:**

- **Endpoint:** `/predict` (POST)
  
- **Payload:** JSON object containing the required features.
  
- **Response:** JSON object with the prediction (`1` for "Up", `0` for "Down") and the probability of the stock price increasing.

**Usage Example:**

- **Request:**

  ```json
  {
      "features": [150.0, 145.0, 60.0, 155.0, 140.0, 0.02, 148.0, 3000000]
  }
  ```

- **Response:**

  ```json
  {
      "prediction": 1,
      "probability": 0.85
  }
  ```

**Implementation Highlights:**

- **Error Handling:** Ensures that the input data is correctly formatted and contains the required number of features. Returns meaningful error messages for invalid inputs.
  
- **Logging:** Captures and logs errors for troubleshooting and maintaining the API. Logs successful model loading and prediction requests.
  
- **Scalability:** Designed to handle multiple requests efficiently, suitable for deployment in production environments with appropriate scaling strategies.

**Integration Steps:**

1. **Setup Environment Variables:**
   - `MODELS_DIR`: Path to the directory containing the trained models.
   - `PORT`: Port number for the Flask server (default is `5001`).

2. **Deploying the API:**
   - Ensure that the Flask server is running continuously using process managers like **Gunicorn** or **Docker** for containerization.

3. **Security Considerations:**
   - Implement authentication and authorization mechanisms to secure the API endpoints.
   - Use HTTPS to encrypt data transmission.

### Streamlit App

The Streamlit app provides a user-friendly interface for users to input feature values and receive immediate predictions from the trained **Random Forest** model. This empowers users with an interactive tool for decision-making without requiring technical expertise.

**Key Features:**

- **User Inputs:** Provides input fields for all required features with validation to ensure data integrity.
  
- **Prediction Display:** Shows the prediction ("Up" or "Down") and the probability of the stock price increasing.
  
- **Error Handling:** Displays error messages for invalid inputs or prediction issues, enhancing user experience.
  
- **Visualization:** Incorporates visual elements to display prediction probabilities, aiding in intuitive understanding.

**Usage Flow:**

1. **Input Features:** Users enter values for MA10, MA50, RSI, BB_upper, BB_lower, Daily_Return, Lagged_Close, and Lagged_Volume.
  
2. **Generate Prediction:** Upon clicking the "Predict" button, the app processes the input and displays the prediction along with the associated probability.
  
3. **Interpret Results:** Users can interpret the results to make informed investment decisions.

**Implementation Highlights:**

- **Real-Time Interaction:** Provides instant feedback based on user inputs, enhancing interactivity.
  
- **User-Friendly Design:** Simplifies the prediction process, making it accessible to non-technical users.
  
- **Integration with Trained Model:** Seamlessly connects to the trained **Random Forest** model to generate accurate predictions.

**Deployment Considerations:**

- **Hosting:** Deploy the Streamlit app on platforms like **Streamlit Sharing**, **Heroku**, or **AWS** for broader accessibility.
  
- **UI Enhancements:** Incorporate additional UI elements such as sliders, dropdowns, and tooltips to improve user experience.
  
- **Feedback Mechanism:** Allow users to provide feedback on predictions to facilitate continuous improvement of the model.

---

## Setup Instructions

### Prerequisites

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.7 or higher
- **Git:** Installed on your system
- **Internet Connection:** Required for downloading stock data via `yfinance`

### Cloning the Repository

Begin by cloning the GitHub repository to your local machine. Open your terminal or command prompt and execute:

```bash
git clone https://github.com/yourusername/Stock-Market-Predictive-Analysis.git
cd Stock-Market-Predictive-Analysis
```

### Creating and Activating a Virtual Environment

It's good practice to create a virtual environment to manage your project's dependencies. Run the following commands:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **On macOS and Linux:**

  ```bash
  source venv/bin/activate
  ```

### Installing Dependencies

Ensure that you have all the necessary Python libraries installed. Install them using `pip` with the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Note:** If `requirements.txt` is not present, create one with the following content:

```plaintext
pandas
numpy
matplotlib
seaborn
yfinance
scikit-learn
imbalanced-learn
joblib
shap
tensorflow
keras
flask
streamlit
```

**Key Steps in the Notebook:**

1. **Data Collection:** Downloads and saves data in the `data/` directory.
2. **Feature Engineering:** Calculates technical indicators to enhance model inputs.
3. **EDA:** Creates visualizations to understand data distributions and correlations.
4. **Modeling:** Trains multiple classifiers and an LSTM model.
5. **Model Evaluation:** Assesses model performance using Accuracy and AUC-ROC metrics.
6. **Model Interpretation:** Utilizes SHAP for feature importance analysis.
7. **Model Saving:** Saves trained models in the `Models/` directory.

### Setting Up and Running the Flask API

The Flask API enables external applications to interact with the trained **Random Forest** model for making predictions.

1. **Run the Flask API:**

   Execute the following command to start the Flask server:

   ```bash
   python api.py
   ```

   The API will be accessible at `http://localhost:5001/predict`. It accepts POST requests with a JSON payload containing the features and returns the prediction and probability.

**API Usage Example:**

- **Request:**

  ```json
  {
      "features": [150.0, 145.0, 60.0, 155.0, 140.0, 0.02, 148.0, 3000000]
  }
  ```

- **Response:**

  ```json
  {
      "prediction": 1,
      "probability": 0.85
  }
  ```

**Implementation Highlights:**

- **Error Handling:** Ensures that the input data is correctly formatted and contains the required number of features. Returns meaningful error messages for invalid inputs.
  
- **Logging:** Captures and logs errors for troubleshooting and maintaining the API. Logs successful model loading and prediction requests.
  
- **Scalability:** Designed to handle multiple requests efficiently, suitable for deployment in production environments with appropriate scaling strategies.

### Setting Up and Running the Streamlit App

The Streamlit app provides a user-friendly interface for users to input feature values and receive immediate predictions from the trained **Random Forest** model. This empowers users with an interactive tool for decision-making without requiring technical expertise.


1. **Run the Streamlit App:**

   Execute the following command to start the Streamlit app:

   ```bash
   streamlit run smpaui.py
   ```

   The app will open in your default web browser, allowing you to input feature values and receive predictions interactively.

**Streamlit App Features:**

- **User Inputs:** Provides input fields for all required features with validation to ensure data integrity.
  
- **Prediction Display:** Shows the prediction ("Up" or "Down") and the probability of the stock price increasing.
  
- **Error Handling:** Displays error messages for invalid inputs or prediction issues, enhancing user experience.
  
- **Visualization:** Incorporates visual elements to display prediction probabilities, aiding in intuitive understanding.

**Implementation Highlights:**

- **Real-Time Interaction:** Provides instant feedback based on user inputs, enhancing interactivity.
  
- **User-Friendly Design:** Simplifies the prediction process, making it accessible to non-technical users.
  
- **Integration with Trained Model:** Seamlessly connects to the trained **Random Forest** model to generate accurate predictions.

**Deployment Considerations:**

- **Hosting:** Deploy the Streamlit app on platforms like **Streamlit Sharing**, **Heroku**, or **AWS** for broader accessibility.
  
- **UI Enhancements:** Incorporate additional UI elements such as sliders, dropdowns, and tooltips to improve user experience.
  
- **Feedback Mechanism:** Allow users to provide feedback on predictions to facilitate continuous improvement of the model.

---

## Repository Structure

```
Stock-Market-Predictive-Analysis/
│
├── Models/
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── svc_model.pkl
│   ├── decision_tree_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── stacking_classifier_model.pkl
│   └── lstm_stock_predictor.keras
├── data/
│   ├── AAPL.csv
│   ├── GOOGL.csv
│   ├── AMZN.csv
├── smpa.ipynb
│   
├── api/
│   └── api.py
├── streamlit_app/
│   └── smpaui.py
├── README.md
├── Final_Report.md
├── requirements.txt
└── .gitignore
```

- **Models/**: Contains all trained models saved in `.pkl` format and the LSTM model in `.keras` format.
  
- **data/**: Stores downloaded CSV files for each stock symbol, including data such as Open, Close, High, Low, and Volume.
  
- **notebooks/**: Includes the main Jupyter Notebook detailing the entire analysis, from data collection to model evaluation.
  
- **api/**: Contains the Flask API script (`api.py`) for model predictions, enabling integration with other applications.
  
- **streamlit_app/**: Contains the Streamlit app script (`app.py`) for user-friendly interactions, allowing users to input features and receive predictions.
  
- **README.md**: Provides an overview, summary of findings, setup instructions, and repository structure.
  
- **Final_Report.md**: Combines technical and non-technical reports, summarizing the project's scope, methodology, results, and recommendations.
  
- **requirements.txt**: Lists all the Python libraries required to run the project, ensuring easy setup and reproducibility.
  
- **.gitignore**: Specifies files and directories to be ignored by Git (e.g., virtual environments, data files), maintaining repository cleanliness.

---

## Model Saving

All trained models are saved in the `Models/` directory for future use and deployment. These include:

- **Random Forest:** `random_forest_model.pkl`
- **Gradient Boosting:** `gradient_boosting_model.pkl`
- **Support Vector Classifier (SVC):** `svc_model.pkl`
- **Decision Tree:** `decision_tree_model.pkl`
- **Logistic Regression:** `logistic_regression_model.pkl`
- **Stacking Classifier:** `stacking_classifier_model.pkl`
- **LSTM Neural Network:** `lstm_stock_predictor.keras`

**Model Loading Examples:**

- **Random Forest:**

  ```python
  import joblib

  # Load Random Forest model
  rf_model = joblib.load('Models/random_forest_model.pkl')
  ```

- **LSTM Neural Network:**

  ```python
  from keras.models import load_model

  # Load LSTM model
  lstm_model = load_model('Models/lstm_stock_predictor.keras')
  ```

These models can be integrated into applications or used for further analysis, facilitating scalability and practical deployment.

---

# Conclusion
This project demonstrates the potential of machine learning and deep learning in forecasting stock price movements. By leveraging technical indicators and advanced models, investors can gain valuable insights to inform their investment strategies, ultimately aiming to enhance returns and manage risks more effectively. While ensemble models have shown superior performance, there is room for improvement in deep learning approaches, especially with further optimization and integration of additional data sources. The development of a Flask API and Streamlit app further enhances the project's applicability, providing tools for real-time predictions and user-friendly interactions.
