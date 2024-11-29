import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
import numpy as np

# ------------------- Step 1: Load and Preprocess Data ------------------- #

def load_data(file_path):
    """Load and preprocess S&P 500 index data."""
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['daily_return'] = data['S&P500'].pct_change()
    data = data.dropna()  # Drop rows with missing values
    return data

# ---------------- Step 2: GARCH Volatility Modeling ---------------- #

def detect_garch_anomalies(data):
    """Fit GARCH model and detect high-volatility periods."""
    model = arch_model(data['daily_return'], vol='Garch', p=1, q=1)
    garch_results = model.fit(disp="off")
    data['garch_volatility'] = garch_results.conditional_volatility

    # Define anomaly threshold
    threshold = data['garch_volatility'].mean() + 2 * data['garch_volatility'].std()
    data['garch_anomaly'] = data['garch_volatility'] > threshold
    return data, threshold

def plot_garch(data, threshold):
    """Plot GARCH volatility and anomalies."""
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['garch_volatility'], label='Volatility')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    plt.scatter(data[data['garch_anomaly']]['Date'], 
                data[data['garch_anomaly']]['garch_volatility'], 
                color='red', label='Anomalies')
    plt.title('GARCH Volatility on S&P 500 Index')
    plt.xlabel('Date')
    plt.ylabel('Conditional Volatility')
    plt.legend()
    plt.show()

# ---------------- Step 3: ARIMA Residual Analysis ---------------- #

def detect_arima_anomalies(data):
    """Fit ARIMA model and detect residual anomalies."""
    model = ARIMA(data['S&P500'], order=(1, 1, 1))
    arima_results = model.fit()
    data['arima_residual'] = arima_results.resid

    # Define anomaly threshold
    threshold = data['arima_residual'].std() * 2
    data['arima_anomaly'] = abs(data['arima_residual']) > threshold
    return data, threshold

def plot_arima(data, threshold):
    """Plot ARIMA residuals and anomalies."""
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['arima_residual'], label='Residuals')
    plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    plt.axhline(y=-threshold, color='green', linestyle='--')
    plt.scatter(data[data['arima_anomaly']]['Date'], 
                data[data['arima_anomaly']]['arima_residual'], 
                color='red', label='Anomalies')
    plt.title('ARIMA Residuals on S&P 500 Index')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

# ---------------- Step 4: Isolation Forest Anomaly Detection ---------------- #

def detect_isolation_forest_anomalies(data):
    """Apply Isolation Forest to detect anomalies."""
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    data['isolation_anomaly'] = isolation_forest.fit_predict(data[['daily_return']])
    data['isolation_anomaly'] = data['isolation_anomaly'] == -1  # Convert to boolean
    return data

def plot_isolation(data):
    """Plot Isolation Forest results."""
    plt.figure(figsize=(10, 6))
    normal_data = data[data['isolation_anomaly'] == False]
    anomalous_data = data[data['isolation_anomaly'] == True]
    plt.scatter(normal_data['Date'], normal_data['daily_return'], alpha=0.5, label='Normal')
    plt.scatter(anomalous_data['Date'], anomalous_data['daily_return'], color='red', label='Anomalies')
    plt.title('Isolation Forest: Anomaly Detection on S&P 500 Index')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.show()

# ---------------- Step 5: Main Execution ---------------- #

if __name__ == "__main__":
    # File path to S&P 500 index data
    file_path = "sp500_index.csv"

    # Load and preprocess the data
    index_data = load_data(file_path)

    # Apply GARCH
    index_data, garch_threshold = detect_garch_anomalies(index_data)
    plot_garch(index_data, garch_threshold)

    # Apply ARIMA
    index_data, arima_threshold = detect_arima_anomalies(index_data)
    plot_arima(index_data, arima_threshold)

    # Apply Isolation Forest
    index_data = detect_isolation_forest_anomalies(index_data)
    plot_isolation(index_data)

    # Save the processed data with anomalies
    index_data.to_csv("processed_sp500_index_anomalies.csv", index=False)
    print("Processed data saved to 'processed_sp500_index_anomalies.csv'.")
