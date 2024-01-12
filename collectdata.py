import yfinance as yf

DEBUG = True

# Parameters
STOCK = 'AAPL'  # Example stock ticker
START_DATE = '2023-01-01'
END_DATE = '2023-10-01'
INTERVAL = '1d'  # 1-day interval

# Part 1: Data Collection
def collect_data(ticker, start_date, end_date, interval):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data

# Part 2: Data Preprocessing
def preprocess_data(data):
    # For simplicity, we'll just use closing prices; more features can be added here
    data = data[['Close']]
    data.dropna(inplace=True)  # Remove missing values
    return data