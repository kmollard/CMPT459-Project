import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest

# ------------------- Step 1: Load and Preprocess Data ------------------- #

def load_data(file_path):
    """Load and preprocess S&P 500 index data."""
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['daily_return'] = data['S&P500'].pct_change()
    data = data.dropna()  # Drop rows with missing values
    return data

def split_data(data, split_date="2019-01-01"):
    """Split the data into training and testing sets."""
    train_data = data[data['Date'] < split_date]
    test_data = data[data['Date'] >= split_date]
    return train_data, test_data

# ---------------- Step 2: GARCH Volatility Modeling ---------------- #

def detect_garch_anomalies(train_data, test_data):
    """Train GARCH on training data and detect high-volatility periods in test data."""
    # Standardize daily returns
    train_data['std_daily_return'] = (train_data['daily_return'] - train_data['daily_return'].mean()) / train_data['daily_return'].std()
    garch_model = arch_model(train_data['std_daily_return'], vol='Garch', p=1, q=1)
    garch_results = garch_model.fit(disp="off")

    # Forecast conditional volatility
    test_data['std_daily_return'] = (test_data['daily_return'] - train_data['daily_return'].mean()) / train_data['daily_return'].std()
    test_data['garch_volatility'] = garch_results.forecast(horizon=len(test_data)).variance.iloc[-1].values

    # Threshold based on training data
    threshold = train_data['std_daily_return'].std() * 2
    test_data['garch_anomaly'] = test_data['garch_volatility'] > threshold
    return test_data, threshold

def plot_garch(test_data, threshold):
    """Plot GARCH volatility and anomalies."""
    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Date'], test_data['garch_volatility'], label='Volatility')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    plt.scatter(test_data[test_data['garch_anomaly']]['Date'], 
                test_data[test_data['garch_anomaly']]['garch_volatility'], 
                color='red', label='Anomalies')
    plt.title('GARCH Volatility on S&P 500 Index')
    plt.xlabel('Date')
    plt.ylabel('Conditional Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------- Step 3: ARIMA Residual Analysis ---------------- #

def detect_arima_anomalies(train_data, test_data):
    """Train ARIMA on training data and detect residual anomalies in test data."""
    # Ensure stationarity via differencing
    train_data['diff_S&P500'] = train_data['S&P500'].diff().dropna()
    arima_model = ARIMA(train_data['diff_S&P500'].dropna(), order=(1, 1, 1))
    arima_results = arima_model.fit()

    # Apply model to test data
    test_data['diff_S&P500'] = test_data['S&P500'].diff().dropna()
    test_data['arima_forecast'] = arima_results.forecast(steps=len(test_data))
    test_data['arima_residual'] = test_data['diff_S&P500'] - test_data['arima_forecast']

    # Rolling threshold
    test_data['rolling_std'] = test_data['arima_residual'].rolling(window=30).std()
    test_data['arima_anomaly'] = abs(test_data['arima_residual']) > 2 * test_data['rolling_std']
    return test_data

def plot_arima(test_data):
    """Plot ARIMA residuals and anomalies."""
    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Date'], test_data['arima_residual'], label='Residuals')
    plt.axhline(y=2 * test_data['rolling_std'].mean(), color='green', linestyle='--', label='Threshold')
    plt.scatter(test_data[test_data['arima_anomaly']]['Date'], 
                test_data[test_data['arima_anomaly']]['arima_residual'], 
                color='red', label='Anomalies')
    plt.title('ARIMA Residuals on S&P 500 Index')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------- Step 4: Isolation Forest Anomaly Detection ---------------- #

def detect_isolation_forest_anomalies(train_data, test_data):
    """Train Isolation Forest on training data and detect anomalies in test data."""
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    features_train = train_data[['daily_return']].dropna()
    isolation_forest.fit(features_train)
    features_test = test_data[['daily_return']].dropna()
    test_data['isolation_anomaly'] = isolation_forest.predict(features_test) == -1  # Convert to boolean
    return test_data

def plot_isolation(test_data):
    """Plot Isolation Forest results."""
    plt.figure(figsize=(10, 6))
    normal_data = test_data[test_data['isolation_anomaly'] == False]
    anomalous_data = test_data[test_data['isolation_anomaly'] == True]
    plt.scatter(normal_data['Date'], normal_data['daily_return'], alpha=0.5, label='Normal')
    plt.scatter(anomalous_data['Date'], anomalous_data['daily_return'], color='red', label='Anomalies')
    plt.title('Isolation Forest: Anomaly Detection on S&P 500 Index\nContamination Level: 5%')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------- Consolidated Overlay Plot ---------------- #

def plot_overlay(test_data):
    """Plot anomalies from all three methods on one chart."""
    plt.figure(figsize=(12, 6))
    plt.plot(test_data['Date'], test_data['S&P500'], label='S&P 500 Index', color='blue')
    
    # Add anomalies
    plt.scatter(test_data[test_data['garch_anomaly']]['Date'], 
                test_data[test_data['garch_anomaly']]['S&P500'], 
                color='red', label='GARCH Anomalies')
    plt.scatter(test_data[test_data['arima_anomaly']]['Date'], 
                test_data[test_data['arima_anomaly']]['S&P500'], 
                color='green', label='ARIMA Anomalies')
    plt.scatter(test_data[test_data['isolation_anomaly']]['Date'], 
                test_data[test_data['isolation_anomaly']]['S&P500'], 
                color='orange', label='Isolation Forest Anomalies')
    
    plt.title('Consolidated Anomaly Detection on S&P 500 Index')
    plt.xlabel('Date')
    plt.ylabel('S&P 500 Index')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------- Step 5: Main Execution ---------------- #

if __name__ == "__main__":
    # File path to S&P 500 index data
    file_path = "sp500_index.csv"

    # Load and preprocess the data
    index_data = load_data(file_path)
    train_data, test_data = split_data(index_data)

    # Apply GARCH
    test_data, garch_threshold = detect_garch_anomalies(train_data, test_data)
    plot_garch(test_data, garch_threshold)

    # Apply ARIMA
    test_data = detect_arima_anomalies(train_data, test_data)
    plot_arima(test_data)

    # Apply Isolation Forest
    test_data = detect_isolation_forest_anomalies(train_data, test_data)
    plot_isolation(test_data)

    # Consolidated visualization
    plot_overlay(test_data)

    # Save the processed test data
    test_data.to_csv("processed_sp500_test_anomalies.csv", index=False)
    print("Processed test data saved to 'processed_sp500_test_anomalies.csv'.")
