# Stock Market Predictive Analysis Capstone Project

## Executive Summary

This project explores the predictive capabilities of machine learning and deep learning models in forecasting daily stock price movements for Apple (AAPL), Google (GOOGL), and Amazon (AMZN). By leveraging historical stock data and technical indicators, various models were developed to predict whether a stock's closing price will increase the next trading day. The analysis encompasses data collection, feature engineering, exploratory data analysis (EDA), model training, evaluation, interpretation, and final recommendations.

## Problem Statement

Accurately predicting stock price movements is a fundamental challenge in the financial industry, offering significant implications for investment strategies and portfolio management. Traditional analysis methods may not fully capture the intricate patterns and indicators influencing stock prices. This project aims to harness machine learning and deep learning techniques to enhance prediction accuracy, providing actionable insights for investors and financial analysts.

**Research Question:**

Can machine learning and deep learning models accurately predict the daily movement of stock prices for Apple (AAPL), Google (GOOGL), and Amazon (AMZN) based on historical data and technical indicators?

## Technical Report

For a detailed walkthrough of the analysis, including data preprocessing, modeling, evaluation, and interpretation, please refer to the [Stock_Market_Predictive_Analysis.ipynb](notebooks/Stock_Market_Predictive_Analysis.ipynb) Jupyter Notebook.

### **Key Components:**

1. **Data Collection:**
   - Utilized the `yfinance` library to gather historical stock data for AAPL, GOOGL, and AMZN from January 1, 2018, to December 31, 2023.

2. **Feature Engineering:**
   - Calculated technical indicators such as Moving Averages (MA10, MA50), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Average True Range (ATR).
   - Added lagged features to incorporate historical price movements into the predictive models.

3. **Exploratory Data Analysis (EDA):**
   - Visualized closing price trends, distribution of daily returns, and correlation between technical indicators.
   - Identified key relationships and patterns that inform model selection and feature importance.

4. **Modeling:**
   - Implemented multiple classification models: Random Forest, Gradient Boosting, Support Vector Classifier (SVC), Decision Tree, Logistic Regression, and Stacking Classifier.
   - Developed an LSTM (Long Short-Term Memory) neural network to capture temporal dependencies in the data.
   - Performed hyperparameter tuning using Grid Search with cross-validation to optimize model performance.

5. **Model Evaluation:**
   - Assessed models using metrics such as Accuracy and AUC-ROC.
   - Ensemble methods like Random Forest and Gradient Boosting demonstrated superior performance.
   - The Stacking Classifier, combining multiple models, achieved the highest AUC-ROC score, indicating enhanced predictive capabilities.

6. **Model Interpretation:**
   - Applied SHAP (SHapley Additive exPlanations) to interpret feature importance across different models.
   - Moving Averages and RSI emerged as the most influential features in predicting stock price movements.

7. **Feature Importance:**
   - Visualized feature importances for Random Forest, Gradient Boosting, and Stacking Classifier models, highlighting the significance of key technical indicators.

8. **Model Saving:**
   - Saved all trained models in the `Models/` directory for future use and deployment.

### **Findings and Recommendations:**

- **Effective Predictors:** Technical indicators such as Moving Averages and RSI are strong predictors of stock price movements, aligning with financial theories.
- **Ensemble Superiority:** Ensemble models, particularly Random Forest and Gradient Boosting, outperform individual classifiers, offering robust predictive performance.
- **Deep Learning Potential:** While LSTM models capture temporal patterns, they require further optimization to match the performance of ensemble methods.
- **Actionable Insights:** The predictive models can aid investors in making informed decisions, potentially enhancing portfolio returns and mitigating risks.

**Recommendations:**

1. **Hyperparameter Optimization:** Utilize more extensive hyperparameter tuning techniques like Randomized Search or Bayesian Optimization to further enhance model performance.
2. **Incorporate Sentiment Analysis:** Integrate sentiment scores from news articles and social media to capture market sentiment, potentially improving predictive accuracy.
3. **Advanced Deep Learning Architectures:** Experiment with more complex architectures, such as Bidirectional LSTMs or Attention Mechanisms, to better capture temporal dependencies.
4. **Real-Time Deployment:** Implement real-time data fetching and model deployment to facilitate live trading strategies.
5. **Continuous Monitoring:** Establish systems for continuous model monitoring and retraining to adapt to evolving market conditions.

## Non-Technical Report

### **Introduction**

Predicting stock price movements is a critical task for investors seeking to maximize returns and minimize risks. This project explores the use of advanced analytical techniques to forecast whether a stock's closing price will increase the next day. Focusing on three tech giants—Apple, Google, and Amazon—we aim to develop models that can provide actionable insights for investment decisions.

### **Methodology**

1. **Data Collection:** We gathered five years of historical stock data for AAPL, GOOGL, and AMZN, including daily prices and trading volumes.
2. **Feature Engineering:** By calculating various technical indicators, we enhanced the dataset to capture market trends and momentum.
3. **Model Development:** Utilizing a suite of machine learning models, we trained classifiers to predict stock price movements.
4. **Evaluation:** Models were assessed based on their accuracy and ability to distinguish between positive and negative movements.
5. **Interpretation:** Through interpretative tools, we identified the most influential factors driving the predictions.

### **Results**

Our analysis revealed that ensemble models like Random Forest and Gradient Boosting achieved the highest predictive performance, effectively identifying days when stock prices would rise. Key technical indicators such as Moving Averages and RSI were instrumental in these predictions. The deep learning model, while promising, showed potential but requires further refinement to match ensemble methods.

### **Key Findings**

- **Technical Indicators:** Moving Averages (MA10, MA50) and RSI are significant predictors of stock price increases.
- **Model Performance:** Ensemble models outperform individual classifiers, offering robust and reliable predictions.
- **Interpretability:** Understanding which factors influence predictions enhances trust and facilitates informed decision-making.

### **Recommendations**

1. **Model Enhancement:** Further optimize existing models and explore additional features to improve accuracy.
2. **Data Integration:** Incorporate sentiment analysis from news and social media to capture broader market sentiments.
3. **Deployment:** Develop a user-friendly dashboard for real-time predictions to aid investors in making timely decisions.
4. **Continuous Improvement:** Regularly update models with new data to maintain and enhance predictive performance.

### **Conclusion**

This project demonstrates the potential of machine learning and deep learning in forecasting stock price movements. By leveraging technical indicators and advanced models, investors can gain valuable insights to inform their investment strategies, ultimately aiming to enhance returns and manage risks more effectively.
