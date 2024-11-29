import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import silhouette_score

# Load datasets
sp500_companies = pd.read_csv('sp500_companies.csv')
sp500_index = pd.read_csv('sp500_index.csv')
sp500_stocks = pd.read_csv('sp500_stocks.csv')

# Parse dates for time-series
sp500_index['Date'] = pd.to_datetime(sp500_index['Date'])
sp500_stocks['Date'] = pd.to_datetime(sp500_stocks['Date'])

# ==========================================================
# 1. Data Preprocessing
# ==========================================================
# Handle missing values
sp500_stocks = sp500_stocks.dropna(subset=['Close'])  # Drop rows with missing Close prices
sp500_stocks['Return'] = sp500_stocks.groupby('Symbol')['Close'].pct_change()  # Calculate daily returns

# Merge stock and company data
sp500_data = sp500_stocks.merge(sp500_companies, on='Symbol', how='left')

# ==========================================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================================
# Plot S&P 500 index over time
# plt.figure(figsize=(10, 6))
# plt.plot(sp500_index['Date'], sp500_index['S&P500'], label='S&P 500 Index', color='blue')
# plt.title('S&P 500 Index Over Time')
# plt.xlabel('Date')
# plt.ylabel('Index Value')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Sector weights pie chart
# sector_weights = sp500_companies.groupby('Sector')['Weight'].sum().sort_values(ascending=False)
# sector_counts = sp500_companies['Sector'].value_counts()
# sector_labels = [f"{sector} ({sector_counts[sector]})" for sector in sector_weights.index]

# plt.figure(figsize=(10, 6))
# sector_colors = sns.color_palette('Set3', n_colors=len(sector_weights))
# plt.pie(sector_weights, labels=sector_labels, autopct='%1.1f%%', startangle=140, colors=sector_colors)
# plt.title('S&P 500 Sector Weights with Company Counts')
# plt.axis('equal')
# plt.show()

# Correlation heatmap
# plt.figure(figsize=(8, 6))
# stock_features = sp500_stocks[['Close', 'High', 'Low', 'Open', 'Volume']].dropna()
# corr_matrix = stock_features.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Heatmap of Stock Features')
# plt.show()

# ==========================================================
# 3. Clustering
# ==========================================================
# Feature selection for clustering
clustering_features = sp500_data[['Return', 'Volume']].dropna()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
sp500_data['Cluster'] = kmeans.fit_predict(clustering_features)

# Visualize clusters using PCA
pca = PCA(n_components=2)
clustering_features_pca = pca.fit_transform(clustering_features)
plt.figure(figsize=(10, 6))
plt.scatter(clustering_features_pca[:, 0], clustering_features_pca[:, 1], c=sp500_data['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Clusters of Stocks Based on Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()

# Evaluate clustering
sil_score = silhouette_score(clustering_features, sp500_data['Cluster'])
print(f"Silhouette Score for Clustering: {sil_score}")

# ==========================================================
# 4. Outlier Detection
# ==========================================================
# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
sp500_data['Anomaly_Score'] = iso_forest.fit_predict(sp500_data[['Close', 'Volume']].dropna())

# Visualize anomaly scores
anomaly_matrix = sp500_data.pivot(index='Symbol', columns='Date', values='Anomaly_Score')
plt.figure(figsize=(12, 8))
sns.heatmap(anomaly_matrix, cmap='coolwarm', center=0)
plt.title('Anomaly Scores Heatmap')
plt.xlabel('Date')
plt.ylabel('Stock Symbol')
plt.show()

# ==========================================================
# 5. Time-Series Analysis
# ==========================================================
# ARIMA: Forecasting S&P 500 index
sp500_index.set_index('Date', inplace=True)
sp500_index_diff = sp500_index['S&P500'].diff().dropna()
arima_model = ARIMA(sp500_index_diff, order=(1, 1, 1))
arima_result = arima_model.fit()

# Plot ARIMA forecast
forecast = arima_result.forecast(steps=10)
plt.figure(figsize=(10, 6))
plt.plot(sp500_index.index[-100:], sp500_index['S&P500'][-100:], label='Actual')
plt.plot(pd.date_range(sp500_index.index[-1], periods=10, freq='D'), forecast, label='Forecast', linestyle='--')
plt.title('ARIMA Forecast of S&P 500 Index')
plt.legend()
plt.show()

# GARCH: Volatility modeling on returns
returns = sp500_index['S&P500'].pct_change().dropna()
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_result = garch_model.fit(disp='off')

# Plot GARCH volatility
volatility = garch_result.conditional_volatility
plt.figure(figsize=(10, 6))
plt.plot(volatility, label='Volatility')
plt.title('GARCH Model Conditional Volatility')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# ==========================================================
# Areas for Optimization
# ==========================================================
# 1. **Clustering Features**:
#    - Experiment with additional features like moving averages, sector information.
# 2. **Hyperparameter Tuning**:
#    - Use GridSearchCV to optimize K-Means (e.g., n_clusters) and Isolation Forest parameters.
# 3. **Dimensionality Reduction**:
#    - Use t-SNE or UMAP for better visualizations if PCA doesn't capture meaningful variance.
# 4. **Cross-Validation**:
#    - Implement cross-validation for ARIMA and GARCH models to ensure robustness.

print("Project completed!")
