from fastapi import FastAPI, Query, HTTPException
from typing import Literal
from datetime import datetime
import yfinance as yf
import time
import os

app = FastAPI(
    title="Yahoo Finance Historical Data API",
    description="Fetch historical stock data (days or minutes) with retry and CSV export.",
    version="1.0.0"
)

# Define valid intervals
INTERVALS = {
    "days": "1d",
    "minutes": "1m"
}

@app.get("/historical-data")
def get_historical_data(
    ticker: str = Query(..., description="Ticker symbol like 'AAPL', 'GOOGL', 'MSFT', etc."),
    period_type: Literal["days", "minutes"] = Query(..., description="Choose between 'days' or 'minutes'")
):
    """
    Fetches historical data from Yahoo Finance, handles rate limits, saves to DataFrame and CSV.
    """
    interval = INTERVALS[period_type]
    period = "60d" if period_type == "minutes" else "max"

    retry_attempts = 3
    retry_delay = 20  # seconds

    for attempt in range(retry_attempts):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                threads=False
            )

            if df.empty:
                raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'.")

            # Reset index to include Date as a column
            df.reset_index(inplace=True)

            # Display first 5 rows for validation
            print(f"[INFO] DataFrame preview for {ticker} ({interval}):")
            print(df.head())

            # Processing date string
            today_str = datetime.today().strftime("%Y-%m-%d")

            # Get the absolute path of the current script (main.py)
            script_path = os.path.abspath(__file__)

            # Get the project root: go up 1 level from notebooks/ to fase_04/
            project_root = os.path.dirname(os.path.dirname(script_path))

            # Define the target directory (fase_04/data)
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)

            # Filename with path
            filename = f"{ticker}_{interval}_{today_str}_historical_data.csv"
            filepath = os.path.join(data_dir, filename)
            print(filepath)

            print("filepath: ", filepath)

            # Save CSV
            df.to_csv(filepath, index=False)

            print(f"[INFO] File saved to: {filepath}")

            return {
                "message": f"Data fetched and saved to '{filename}'",
                "rows": len(df),
                "columns": list(df.columns)
            }

        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < retry_attempts - 1:
                print("[WARNING] Rate limit exceeded. Retrying in 20 seconds...")
                time.sleep(retry_delay)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")


# go to main.py path
# RUN IN BASH: uvicorn main:app --reload
# Then go to: http://127.0.0.1:8000/docs
# GET /historical-data?ticker=AAPL&period_type=days