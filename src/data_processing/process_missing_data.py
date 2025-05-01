import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pandas_market_calendars as mcal
from datetime import datetime, time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
pd.set_option('future.no_silent_downcasting', True)
RAW_DATA_DIR = "/notebooks/Market_RV/data/raw_1min_top_sp500_bulk"
PROCESSED_DATA_DIR = "/notebooks/Market_RV/data/processed_1min_top_sp500_bulk"
START_DATE = datetime(2015, 4, 13)
END_DATE = datetime(2025, 4, 11)
NUM_WORKERS = 8
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# --- Ensure output directory exists ---
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- NYSE trading days ---
def get_trading_days(start, end):
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start.strftime("%Y-%m-%d"),
                             end_date=end.strftime("%Y-%m-%d"))
    return [pd.Timestamp(d).tz_localize(None) for d in schedule.index]

# --- Enhanced fill logic for all columns ---
def fill_missing_minute_data(df, market_open, market_close):
    """Process missing data for all specified columns with market-aware logic"""
    df = df.copy().set_index('timestamp').sort_index()

    # Create complete market-minute index for this trading day
    full_index = pd.date_range(start=market_open, end=market_close, freq='1min')
    df = df.reindex(full_index)

    # Constant fields
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].ffill().bfill()

    # Volume and trade count - zero for missing minutes
    df['volume'] = df['volume'].fillna(0)
    df['trade_count'] = df['trade_count'].fillna(0)

    # Price columns - forward fill within same trading day
    price_cols = ['close', 'volume_weighted_average_price']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # OHLC relationships
    if 'close' in df.columns:
        df['open'] = df['open'].combine_first(df['close'])
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Handle market open special case (no trades at open)
    if len(df) > 0:
        first_valid = df[(df['volume'] > 0) & (df['trade_count'] > 0)].first_valid_index()
        if first_valid is not None and first_valid > df.index[0]:
            # Backfill prices from market open to the minute before the first valid trade
            fill_values = df.loc[first_valid, ['open', 'high', 'low', 'close', 'volume_weighted_average_price']]
            fill_start_time = df.index[0]
            fill_end_time = first_valid

            # Create a range of times to fill (exclusive of first_valid)
            fill_index = pd.date_range(start=fill_start_time, end=fill_end_time, freq='1min', inclusive='left')

            # Apply the fill values and set volume/trade_count to 0
            df.loc[fill_index, fill_values.index] = fill_values.values
            df.loc[fill_index, ['volume', 'trade_count']] = 0

    # Flag imputed data
    df['imputed'] = df['close'].isna().astype(int)

    return df.reset_index().rename(columns={'index': 'timestamp'})

# --- Get all symbols ---
def get_all_symbols(data_dir):
    return [d.split("=")[1] for d in os.listdir(data_dir) if d.startswith("symbol=")]

# --- Save monthly partition ---
def save_monthly_partition(df, symbol, year, month):
    out_dir = os.path.join(PROCESSED_DATA_DIR,
                            f"symbol={symbol}",
                            f"year={year}",
                            f"month={month:02d}")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{symbol}_{year}{month:02d}.parquet")
    df.to_parquet(filename, index=False, engine="pyarrow", compression="snappy")

# --- Worker: process one symbol ---
def process_symbol(symbol):
    try:
        symbol_dir = os.path.join(RAW_DATA_DIR, f"symbol={symbol}")
        if not os.path.exists(symbol_dir):
            return f"[{symbol}] Raw data directory not found."

        all_data = []
        # Iterate through year/month directories
        for root, dirs, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith(".parquet"):
                    try:
                        pq_file = os.path.join(root, file)
                        df = pd.read_parquet(pq_file)
                        if df.empty:
                            continue

                        # Standardize timestamp column
                        if 'timestamp' not in df.columns and 't' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                        elif 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        else:
                            continue

                        if 'symbol' not in df.columns:
                            df['symbol'] = symbol

                        all_data.append(df)
                    except Exception as e:
                        print(f"[{symbol}] Failed to read {pq_file}: {e}")
                        continue

        if not all_data:
            return f"[{symbol}] No data found."

        full_df = pd.concat(all_data).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Extract date components
        full_df['year'] = full_df['timestamp'].dt.year
        full_df['month'] = full_df['timestamp'].dt.month
        full_df['day'] = full_df['timestamp'].dt.date

        # Process by trading day
        filled_data = []
        trading_days = get_trading_days(START_DATE, END_DATE)

        for day in trading_days:
            day_str = day.strftime('%Y-%m-%d')
            day_data = full_df[full_df['day'] == pd.Timestamp(day_str).date()]

            if not day_data.empty:
                market_open = datetime.combine(day.date(), MARKET_OPEN)
                market_close = datetime.combine(day.date(), MARKET_CLOSE)
                filled_day = fill_missing_minute_data(day_data, market_open, market_close)
                filled_data.append(filled_day)

        if not filled_data:
            return f"[{symbol}] No filled data available."

        final_df = pd.concat(filled_data)
        final_df['year'] = final_df['timestamp'].dt.year
        final_df['month'] = final_df['timestamp'].dt.month

        # Save monthly partitions (drop 'year', 'month', and 'day' columns)
        for (year, month), df_month in final_df.groupby(['year', 'month']):
            save_monthly_partition(df_month.drop(columns=['year', 'month', 'day']), symbol, year, month)

        return f"[{symbol}] Successfully processed"
    except Exception as e:
        return f"[{symbol}] Failed: {str(e)}"

# --- Main execution ---
def process_all_data():
    symbols = get_all_symbols(RAW_DATA_DIR)
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_symbol, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing symbols"):
            results.append(future.result())
            print(results[-1])
    return results

if __name__ == "__main__":
    process_all_data()