import os
import pandas as pd
from polygon import StocksClient
from tqdm import tqdm  # For progress tracking
import os

API_KEY = os.getenv("POLYGON_API_KEY")

if API_KEY is None:
    raise EnvironmentError("POLYGON_API_KEY is not set in environment variables.")
OUTPUT_DIR = 'data/raw_1min_top_sp500_bulk'
TICKER_FILE = '/notebooks/Market_RV/data/info/all_sp500.csv'
START_DATE = "2015-04-13"
END_DATE = "2025-04-11"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tickers from file
tickers = pd.read_csv(TICKER_FILE)['Symbol'].tolist()

# Initialize the synchronous client
client = StocksClient(API_KEY)

def fetch_and_save(ticker):
    try:
        # Fetch full range aggregate bars synchronously
        bars = client.get_aggregate_bars(
            ticker,
            START_DATE,
            END_DATE,
            full_range=True,
            multiplier=1,
            timespan='minute'
        )
        
        # Convert to DataFrame directly from the list of dictionaries
        df = pd.DataFrame(bars)
        
        # Rename columns first, so that the original 't' column is renamed to 'timestamp'
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'volume_weighted_average_price',
            'n': 'trade_count',
            't': 'timestamp'
        })
        
        # Now convert the timestamp column from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Add additional metadata columns
        df['symbol'] = ticker
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day

        # Save to Parquet with partitioning
        df.to_parquet(
            path=OUTPUT_DIR,
            engine='pyarrow',
            partition_cols=["symbol", "year", "month"],
            compression="snappy",
            index=False
        )
        print(f"[{ticker}] Data saved successfully.")
    except Exception as e:
        print(f"[{ticker}] Error: {e}")

# Main function to process all tickers sequentially
def main():
    for ticker in tqdm(tickers, desc="Fetching and Saving Data", dynamic_ncols=True):
        fetch_and_save(ticker)

if __name__ == "__main__":
    main()
