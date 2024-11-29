import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
sp500_index = pd.read_csv('sp500_index.csv')
sp500_stocks = pd.read_csv('sp500_stocks.csv')
sp500_companies = pd.read_csv('sp500_companies.csv')

# Parse date columns
sp500_index['Date'] = pd.to_datetime(sp500_index['Date'])
sp500_stocks['Date'] = pd.to_datetime(sp500_stocks['Date'])

# Visualization 1: S&P 500 Index Time Series
plt.figure(figsize=(10, 6))
plt.plot(sp500_index['Date'], sp500_index['S&P500'], label='S&P 500 Index', color='blue')
plt.title('S&P 500 Index Over Time')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 2: Individual Stock Prices vs. Index

# Identify the top 15 weighted stocks
top_15_stocks = sp500_companies.nlargest(15, 'Weight')['Symbol']

# Visualization 2.1: Top 15 Weighted Stocks vs. S&P 500 Index
plt.figure(figsize=(12, 8))
for symbol in top_15_stocks:
    stock_data = sp500_stocks[sp500_stocks['Symbol'] == symbol]
    plt.plot(stock_data['Date'], stock_data['Adj Close'], label=f'{symbol}')

plt.plot(sp500_index['Date'], sp500_index['S&P500'] / sp500_index['S&P500'].max() * 100, 
         label='S&P 500 Index (scaled)', linestyle='--', color='black')
plt.title('Top 15 Weighted Stocks vs. S&P 500 Index')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price / Scaled Index')
plt.legend(ncol=3)
plt.grid(True)
plt.show()

# Add sector information to stock data
sp500_stocks_with_sector = sp500_stocks.merge(sp500_companies[['Symbol', 'Sector']], on='Symbol')

# Calculate sector averages over time
sector_averages = sp500_stocks_with_sector.groupby(['Date', 'Sector'])['Adj Close'].mean().reset_index()

# Visualization 2.2: Sector-Averaged Performance vs. S&P 500 Index
plt.figure(figsize=(12, 8))
for sector in sector_averages['Sector'].unique():
    sector_data = sector_averages[sector_averages['Sector'] == sector]
    plt.plot(sector_data['Date'], sector_data['Adj Close'], label=sector)

plt.plot(sp500_index['Date'], sp500_index['S&P500'] / sp500_index['S&P500'].max() * 100, 
         label='S&P 500 Index (scaled)', linestyle='--', color='black')
plt.title('Sector-Averaged Performance vs. S&P 500 Index')
plt.xlabel('Date')
plt.ylabel('Average Adjusted Close Price / Scaled Index')
plt.legend(ncol=2)
plt.grid(True)
plt.show()

# Visualization 3: Sector Weights
# Refined Pie Chart for S&P 500 Sector Weights
plt.figure(figsize=(10, 6))

# Create the pie chart with sector weights
sector_weights = sp500_companies.groupby('Sector')['Weight'].sum().sort_values(ascending=False)

# Plot the pie chart
sector_colors = sns.color_palette('Set3', n_colors=len(sector_weights))
plt.pie(sector_weights, labels=sector_weights.index, autopct='%1.1f%%', startangle=140, colors=sector_colors, 
        explode=[0.3] * len(sector_weights))  # Explode all slices slightly for emphasis

# Add title and a clean layout
plt.title('S&P 500 Sector Weights', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

# Display the plot
plt.show()

# Visualization 4: Industry Distribution (Bar Chart)
industry_distribution = sp500_companies['Industry'].value_counts().head(10)
plt.figure(figsize=(12, 6))
industry_distribution.plot(kind='bar', color='orange')
plt.title('Top 10 Industries in the S&P 500')
plt.xlabel('Industry')
plt.ylabel('Number of Companies')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualization 5: Correlation Heatmap of Stock Features
stock_features = sp500_stocks[['Close', 'High', 'Low', 'Open', 'Volume']]
stock_correlation = stock_features.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(stock_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Stock Features')
plt.show()
