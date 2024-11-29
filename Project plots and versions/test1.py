import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np

# ------------------- Step 1: Load and Preprocess Data ------------------- #

def load_data(file_path):
    """Load S&P 500 stock data."""
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def preprocess_data(data):
    """Preprocess stock data: calculate daily returns and rolling features."""
    data['daily_return'] = data['Close'].pct_change()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['volatility'] = data['daily_return'].rolling(window=10).std()
    return data.dropna()

# ---------------- Step 2: GARCH Volatility Modeling ---------------- #

def detect_garch_anomalies(data):
    """Fit GARCH model and detect high-volatility periods."""
    model = arch_model(data['daily_return'], vol='Garch', p=1, q=1)
    garch_results = model.fit(disp="off")
    data['garch_volatility'] = garch_results.conditional_volatility
    threshold = data['garch_volatility'].mean() + 2 * data['garch_volatility'].std()
    data['garch_anomaly'] = data['garch_volatility'] > threshold
    return data

# ---------------- Step 3: ARIMA Residual Analysis ---------------- #

def detect_arima_anomalies(data):
    """Fit ARIMA model and detect residual anomalies."""
    model = ARIMA(data['Close'], order=(1, 1, 1))
    arima_results = model.fit()
    data['arima_residual'] = arima_results.resid
    threshold = data['arima_residual'].std() * 2
    data['arima_anomaly'] = data['arima_residual'].abs() > threshold
    return data

# ---------------- Step 4: Isolation Forest Anomaly Detection ---------------- #

def detect_isolation_forest_anomalies(data):
    """Apply Isolation Forest to detect anomalies."""
    features = data[['daily_return', 'MA_10', 'MA_50', 'volatility']]
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    data['isolation_anomaly'] = isolation_forest.fit_predict(features)
    data['isolation_anomaly'] = data['isolation_anomaly'] == -1
    return data

# ---------------- Step 5: Visualization ---------------- #

def visualize_garch(data):
    """Visualize GARCH volatility and anomalies."""
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['garch_volatility'], label='Volatility')
    plt.axhline(y=data['garch_volatility'].mean() + 2 * data['garch_volatility'].std(), color='red', linestyle='--', label='Threshold')
    plt.scatter(data.loc[data['garch_anomaly'], 'Date'], data.loc[data['garch_anomaly'], 'garch_volatility'], color='red', label='Anomalies')
    plt.title('GARCH Volatility and Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

def visualize_arima(data):
    """Visualize ARIMA residuals and anomalies."""
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['arima_residual'], label='Residuals')
    plt.axhline(y=2 * data['arima_residual'].std(), color='green', linestyle='--', label='Threshold')
    plt.axhline(y=-2 * data['arima_residual'].std(), color='green', linestyle='--')
    anomalies = data.loc[data['arima_anomaly']]
    plt.scatter(anomalies['Date'], anomalies['arima_residual'], color='red', label='Anomalies')
    plt.title('ARIMA Residuals and Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

def visualize_isolation(data):
    """Visualize Isolation Forest results."""
    plt.figure(figsize=(10, 6))
    normal_data = data[data['isolation_anomaly'] == False]
    anomalous_data = data[data['isolation_anomaly'] == True]
    plt.scatter(normal_data['daily_return'], normal_data['volatility'], alpha=0.5, label='Normal')
    plt.scatter(anomalous_data['daily_return'], anomalous_data['volatility'], color='red', alpha=0.7, label='Anomalies')
    plt.title('Isolation Forest: Anomaly Detection')
    plt.xlabel('Daily Return')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

# ---------------- Step 6: Main Execution ---------------- #

if __name__ == "__main__":
    # Load and preprocess data
    file_path = "sp500_index.csv"  # Path to your stock dataset
    sp500_stocks = load_data(file_path)
    sp500_stocks = preprocess_data(sp500_stocks)

    # Apply methodologies
    sp500_stocks = detect_garch_anomalies(sp500_stocks)
    sp500_stocks = detect_arima_anomalies(sp500_stocks)
    sp500_stocks = detect_isolation_forest_anomalies(sp500_stocks)

    # Visualize results
    visualize_garch(sp500_stocks)
    visualize_arima(sp500_stocks)
    visualize_isolation(sp500_stocks)

    # Save the processed data
    sp500_stocks.to_csv("processed_sp500_anomalies.csv", index=False)
