import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

DATA_FILE = "historical_data.parquet"

def initialize_historical_data():
    """Load or download complete Bitcoin historical data into Parquet."""
    if os.path.exists(DATA_FILE):
        df = pd.read_parquet(DATA_FILE)
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
        df = yf.download("BTC-USD", start="2010-01-01", end=end_date)
        df = df[["Close"]].reset_index()
        df.columns = ["date", "price"]
        df.to_parquet(DATA_FILE)
    return df

def update_historical_data(existing_data):
    """Fetch and append new days to existing data."""
    last_date = existing_data["date"].max()
    next_day = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    new_data = yf.download("BTC-USD", start=next_day, end=datetime.now().strftime("%Y-%m-%d"))

    if not new_data.empty:
        df_new = new_data[["Close"]].reset_index()
        df_new.columns = ["date", "price"]
        combined = pd.concat([existing_data, df_new])
        combined.to_parquet(DATA_FILE)
        return combined
    return existing_data

def fetch_realtime_price():
    """Get the latest price of Bitcoin."""
    try:
        ticker = yf.Ticker("BTC-USD")
        price = ticker.info.get("regularMarketPrice", 0.0)
        return price
    except Exception as e:
        print(f"Error fetching real-time price: {e}")
        return 0.0