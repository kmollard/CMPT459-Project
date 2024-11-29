import pandas as pd
from sklearn.preprocessing import StandardScaler

# Helper function: Data acquisition
def load_data(stock_file, company_file, index_file):
    stocks = pd.read_csv(stock_file)
    companies = pd.read_csv(company_file)
    index = pd.read_csv(index_file)
    return stocks, companies, index

# Helper function: Data cleaning
def clean_data(stocks):
    # Drop rows with missing values or interpolate them
    stocks.interpolate(method='linear', inplace=True)
    return stocks

# Helper function: Normalize numerical features
def normalize_features(stocks, columns_to_scale):
    scaler = StandardScaler()
    stocks[columns_to_scale] = scaler.fit_transform(stocks[columns_to_scale])
    return stocks

# Helper function: Feature engineering
def add_features(stocks):
    # Calculate daily returns
    stocks['daily_return'] = stocks['close'].pct_change()

    # Add moving averages
    stocks['MA_10'] = stocks['close'].rolling(window=10).mean()
    stocks['MA_50'] = stocks['close'].rolling(window=50).mean()

    # Calculate rolling volatility
    stocks['volatility'] = stocks['daily_return'].rolling(window=10).std()

    return stocks

# Main execution
if __name__ == "__main__":
    # File paths (adjust as necessary)
    stock_file = "sp500_stocks.csv"
    company_file = "sp500_companies.csv"
    index_file = "sp500_index.csv"

    # Load datasets
    stocks, companies, index = load_data(stock_file, company_file, index_file)

    # Verify data structure
    print("Stocks Data Overview:")
    print(stocks.info())
    print("\nCompanies Data Overview:")
    print(companies.info())
    print("\nIndex Data Overview:")
    print(index.info())

    # Data cleaning
    print("\nCleaning stocks data...")
    stocks = clean_data(stocks)

    # Join with company data
    print("\nJoining stocks data with company metadata...")
    stocks = stocks.merge(companies, left_on='Symbol', right_on='Symbol', how='left')

    # Normalize numerical features
    print("\nNormalizing numerical features...")
    columns_to_scale = ['open', 'high', 'low', 'close', 'volume']
    stocks = normalize_features(stocks, columns_to_scale)

    # Feature engineering
    print("\nAdding new features...")
    stocks = add_features(stocks)

    # Save processed data for future use
    processed_file = "processed_sp500_data.csv"
    print(f"\nSaving processed data to {processed_file}...")
    stocks.to_csv(processed_file, index=False)

    print("Data preprocessing and feature engineering completed.")
