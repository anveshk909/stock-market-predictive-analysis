
# ## 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


import shap
import joblib
import datetime
import os

# Set plot style
sns.set(style='whitegrid')
%matplotlib inline

# ## 2. Data Collection

# Function to download stock data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# Define date range
start_date = '2018-01-01'
end_date = '2023-12-31'

# List of stock symbols
symbols = ['AAPL', 'GOOGL', 'AMZN']

# Dictionary to hold data for each symbol
all_data = {}
for symbol in symbols:
    print(f"Downloading data for {symbol}")
    all_data[symbol] = load_data(symbol, start=start_date, end=end_date)
    print(f"Downloaded {len(all_data[symbol])} records for {symbol}")

# ## 3. Feature Engineering

# Function to compute RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute Bollinger Bands
def bollinger_bands(series, window=20):
    sma = series.rolling(window=window).mean()
    std_dev = series.rolling(window=window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

# Function to add features to the dataframe
def add_features(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['BB_upper'], df['BB_lower'] = bollinger_bands(df['Close'], window=20)
    df['Daily Return'] = df['Close'].pct_change()
    df['Lagged Close'] = df['Close'].shift(1)
    df['Lagged Volume'] = df['Volume'].shift(1)
    df.dropna(inplace=True)
    return df

# Apply feature engineering to each stock's data
for symbol in symbols:
    print(f"Adding features for {symbol}")
    all_data[symbol] = add_features(all_data[symbol])
    print(f"Features added for {symbol}")

# ## 4. Exploratory Data Analysis (EDA)

# Function to plot closing price trends
def plot_price_trends(data, symbols):
    plt.figure(figsize=(14, 7))
    for symbol in symbols:
        plt.plot(data[symbol].index, data[symbol]['Close'], label=symbol)
    plt.title('Closing Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Closing Price USD ($)')
    plt.legend()
    plt.show()

plot_price_trends(all_data, symbols)

# Function to plot correlation heatmap
def plot_correlation(data, symbols):
    for symbol in symbols:
        plt.figure(figsize=(10, 8))
        corr = data[symbol].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title(f'Correlation Heatmap for {symbol}')
        plt.show()

plot_correlation(all_data, symbols)

# Function to plot distribution of returns
def plot_return_distribution(data, symbols):
    plt.figure(figsize=(14, 7))
    for symbol in symbols:
        sns.histplot(data[symbol]['Daily Return'], bins=100, label=symbol, kde=True)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

plot_return_distribution(all_data, symbols)

# ## 5. Preparing Data for Modeling

# Combine data for all symbols
all_features = []
all_targets = []

for symbol in symbols:
    df = all_data[symbol].copy()
    X = df[['MA10', 'MA50', 'RSI', 'BB_upper', 'BB_lower', 'Daily Return', 'Lagged Close', 'Lagged Volume']]
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = X[:-1]  # Remove last row as it has NaN target
    y = y[:-1]
    all_features.append(X)
    all_targets.append(y)

# Concatenate all data
X = pd.concat(all_features)
y = pd.concat(all_targets)

print(f"Total samples: {X.shape[0]}")
print(f"Total features: {X.shape[1]}")

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ## 6. Modeling

# ### 6.1 Random Forest Classifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest AUC-ROC: ", roc_auc_score(y_test, y_prob_rf))

# ### 6.2 Gradient Boosting Classifier

gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)
y_prob_gbc = gbc.predict_proba(X_test)[:, 1]

print("Gradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gbc))
print("Gradient Boosting AUC-ROC: ", roc_auc_score(y_test, y_prob_gbc))

# ### 6.3 Support Vector Classifier (SVC)

from sklearn.svm import SVC

svc = SVC(probability=True, random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
y_prob_svc = svc.predict_proba(X_test)[:, 1]

print("Support Vector Classifier Report:\n", classification_report(y_test, y_pred_svc))
print("SVC AUC-ROC: ", roc_auc_score(y_test, y_prob_svc))

# ### 6.4 Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
y_prob_dtc = dtc.predict_proba(X_test)[:, 1]

print("Decision Tree Classifier Report:\n", classification_report(y_test, y_pred_dtc))
print("Decision Tree AUC-ROC: ", roc_auc_score(y_test, y_prob_dtc))

# ### 6.5 Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))
print("Logistic Regression AUC-ROC: ", roc_auc_score(y_test, y_prob_lr))

# ## 7. Model Evaluation and Comparison

# Create a DataFrame to compare models
models = ['Random Forest', 'Gradient Boosting', 'SVC', 'Decision Tree', 'Logistic Regression']
accuracy = [
    rf.score(X_test, y_test),
    gbc.score(X_test, y_test),
    svc.score(X_test, y_test),
    dtc.score(X_test, y_test),
    lr.score(X_test, y_test)
]
auc_roc = [
    roc_auc_score(y_test, y_prob_rf),
    roc_auc_score(y_test, y_prob_gbc),
    roc_auc_score(y_test, y_prob_svc),
    roc_auc_score(y_test, y_prob_dtc),
    roc_auc_score(y_test, y_prob_lr)
]

comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'AUC-ROC': auc_roc
})

print(comparison_df)

# ## 8. Deep Learning Model: LSTM

# Preparing data for LSTM
# For LSTM, we'll create sequences of 50 days

sequence_length = 50

def create_sequences(X, y, seq_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y.iloc[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Reshape X for LSTM
X_lstm, y_lstm = create_sequences(X_scaled, y, sequence_length)

# Split into training and testing sets
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42
)

print(f"LSTM Training samples: {X_train_lstm.shape[0]}")
print(f"LSTM Testing samples: {X_test_lstm.shape[0]}")

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_lstm, y_test_lstm)
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate LSTM model
y_pred_lstm_prob = lstm_model.predict(X_test_lstm).flatten()
y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)

print("LSTM Classification Report:\n", classification_report(y_test_lstm, y_pred_lstm))
print("LSTM AUC-ROC: ", roc_auc_score(y_test_lstm, y_pred_lstm_prob))

# ## 9. Model Interpretation Using SHAP

# Using SHAP for Random Forest
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values[1], X_test, feature_names=X.columns)

# ## 10. Feature Importance

# Plot Random Forest Feature Importance
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# ## 11. Save the Models

# Create Models directory if it doesn't exist
models_dir = '../Models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save Random Forest model
joblib.dump(rf, os.path.join(models_dir, 'random_forest_model.pkl'))
print("Random Forest model saved.")

# Save Gradient Boosting model
joblib.dump(gbc, os.path.join(models_dir, 'gradient_boosting_model.pkl'))
print("Gradient Boosting model saved.")

# Save LSTM model
lstm_model.save(os.path.join(models_dir, 'lstm_stock_predictor.h5'))
print("LSTM model saved.")

# ## 12. Summary and Recommendations

print("\nSummary of Insights:")
print("- Random Forest and Gradient Boosting performed the best in terms of predictive accuracy and AUC-ROC.")
print("- Moving Averages (MA10, MA50) and RSI were among the most important features.")
print("- LSTM model requires more fine-tuning but shows promise in learning sequential patterns.")
