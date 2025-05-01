import os
import gc
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ----------------------------
# CORE PIPELINE
# ----------------------------

def load_sp500_tickers(filepath):
    tickers = pd.read_csv(filepath)['Symbol'].tolist()
    print(f"[load_sp500_tickers] Loaded {len(tickers)} tickers from {filepath}")
    return tickers

def pass_a_chunked_market_avg(all_tickers, data_dir="data/processed_1min_top_sp500_bulk",
                             chunk_size=10, market_parquet="market_avg.parquet", max_workers=2):
    print("\n=== PASS A: Chunked approach to compute GLOBAL market average ===")
    pivot_files = []

    # Use ThreadPoolExecutor with limited workers to reduce memory usage
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(0, len(all_tickers), chunk_size):
            chunk_tickers = all_tickers[i : i + chunk_size]
            print(f"[PASS A] Submitting chunk {i//chunk_size+1}: {chunk_tickers}")
            futures[executor.submit(process_chunk_a, chunk_tickers, i, data_dir)] = i

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            chunk_file, tickers = future.result()
            if chunk_file is not None:
                pivot_files.append(chunk_file)
                print(f"[PASS A] Completed chunk for tickers: {tickers}")

    print("\n[PASS A] Unifying partial pivots to compute the global market average...")
    pivot_dfs = []
    for fpath in pivot_files:
        print(f"[PASS A] Loading pivot file: {fpath}")
        pivot_dfs.append(pd.read_parquet(fpath))
        os.remove(fpath)  # Remove temp file immediately after loading
        print(f"[PASS A] Removed temporary file: {fpath}")
    full_pivot = pd.concat(pivot_dfs, ignore_index=True).drop_duplicates()
    
    print(full_pivot.head())
    
    print(full_pivot.columns)
    
    market_avg = full_pivot.groupby('Date').mean(numeric_only=True).reset_index()
    market_avg.columns = ['Date'] + [f"market_{col}" for col in market_avg.columns if col != 'Date']
    market_avg.to_parquet(market_parquet)
    print(f"[PASS A] Global market average saved to {market_parquet}")

    # Cleanup
    del pivot_dfs, full_pivot, market_avg
    gc.collect()

def pass_b_chunked_pipeline(all_tickers, market_parquet="market_avg.parquet",
                            data_dir="data", chunk_size=10, max_workers=2):
    print("\n=== PASS B: Merging with Global Market Averages ===")
    global_market = pd.read_parquet(market_parquet)
    print(f"[PASS B] Loaded global market data from {market_parquet}")
    final_datasets = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(0, len(all_tickers), chunk_size):
            chunk_tickers = all_tickers[i : i + chunk_size]
            print(f"[PASS B] Submitting chunk {i//chunk_size+1}: {chunk_tickers}")
            future = executor.submit(process_chunk_b, chunk_tickers, global_market, data_dir)
            futures[future] = chunk_tickers

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            result = future.result()
            if result is not None:
                final_datasets.append(result)
                print(f"[PASS B] Completed chunk for tickers: {futures[future]}")

    if final_datasets:
        dataset = pd.concat(final_datasets, ignore_index=True)
        print(f"[PASS B] Merged {len(final_datasets)} datasets")
    else:
        dataset = pd.DataFrame()
        print("[PASS B] No datasets to merge")

    # Cleanup
    del final_datasets, global_market
    gc.collect()
    return dataset

# ----------------------------
# CHUNK PROCESSING FUNCTIONS
# ----------------------------

def process_chunk_a(chunk_tickers, i, data_dir):
    print(f"[process_chunk_a] Processing chunk: {chunk_tickers}")
    chunk_vol = compute_all_volatilities(chunk_tickers, data_dir=data_dir)
    
    if chunk_vol is None or chunk_vol.empty:
        print(f"[process_chunk_a] No volatility data for: {chunk_tickers}")
        return None, None
        
    chunk_pivot = process_intraday_data(chunk_vol)
    
    # Check if we got valid pivot data
    if chunk_pivot is None or chunk_pivot.empty:
        print(f"[process_chunk_a] No pivot data produced for: {chunk_tickers}")
        return None, None
        
    chunk_file = f"temp_pivot_{i}.parquet"
    chunk_pivot.to_parquet(chunk_file)
    print(f"[process_chunk_a] Saved pivot file: {chunk_file}")

    # Cleanup
    del chunk_vol, chunk_pivot
    gc.collect()
    return chunk_file, chunk_tickers

def process_chunk_b(chunk_tickers, global_market, data_dir):
    print(f"\n[process_chunk_b] Processing chunk: {chunk_tickers}")
    chunk_vol = compute_all_volatilities(chunk_tickers, data_dir=data_dir)
    if chunk_vol is None:
        print(f"[process_chunk_b] No volatility data for: {chunk_tickers}")
        return None
    chunk_daily_rv = compute_daily_rv(chunk_tickers, data_dir=data_dir)
    print(f"[process_chunk_b] Computed daily RV for: {chunk_tickers}")
    chunk_intraday_rv = process_intraday_data(chunk_vol)

    # Merge with global market data based on the day
    chunk_intraday_rv['date'] = chunk_intraday_rv['Date'].dt.date
    merged_intraday = pd.merge(chunk_intraday_rv, global_market, on='Date', how='left')
    #merged_intraday.drop(columns=['date'], inplace=True) # Remove the temporary date column

    chunk_daily_rv = process_daily_rv_target(chunk_daily_rv)

    # Merge intraday and daily data on symbol and the start of the 15-min interval's day
    merged_intraday['day_only'] = merged_intraday['Date'].dt.date
    chunk_daily_rv['date'] = pd.to_datetime(chunk_daily_rv['date']).dt.date

    chunk_dataset = pd.merge(merged_intraday, chunk_daily_rv,
                             left_on=['symbol', 'day_only'],
                             right_on=['symbol', 'date'],
                             how='inner')
    chunk_dataset.drop(columns=['day_only', 'date_y'], inplace=True)
    chunk_dataset.rename(columns={'date_x': 'date'}, inplace=True)

    print(f"[process_chunk_b] Merged intraday and daily data for: {chunk_tickers}")

    # Cleanup
    del chunk_vol, chunk_daily_rv, chunk_intraday_rv, merged_intraday
    gc.collect()
    return chunk_dataset

# ----------------------------
# DATA PROCESSING FUNCTIONS
# ----------------------------

def process_intraday_data(volatility_df):
    """
    Process intraday volatility data by resampling into 15-min intervals.
    Each row will contain the timestamp of the end of the 15-minute interval,
    the symbol, and the 15-min and 30-min RV calculated up to that point.
    """
    print("[process_intraday_data] Starting intraday processing")
    
    # Check if input is None or empty
    if volatility_df is None or volatility_df.empty:
        print("[process_intraday_data] Input DataFrame is None or empty")
        return pd.DataFrame()
    
    # Debug information about input
    print(f"[process_intraday_data] Input shape: {volatility_df.shape}")
    print(f"[process_intraday_data] Columns: {volatility_df.columns.tolist()}")
    print(f"[process_intraday_data] Index type: {type(volatility_df.index)}")
    
    vol_df = volatility_df.copy()
    
    # Ensure we have datetime index
    try:
        vol_df.index = pd.to_datetime(vol_df.index, errors='coerce')
        print(f"[process_intraday_data] After datetime conversion, valid indices: {vol_df.dropna().shape[0]}")
    except Exception as e:
        print(f"[process_intraday_data] Error converting index to datetime: {e}")
        # If the index isn't datetime already, move forward differently
        if 'Date' in vol_df.columns:
            print("[process_intraday_data] Using Date column instead of index")
            vol_df['Date'] = pd.to_datetime(vol_df['Date'], errors='coerce')
        else:
            print("[process_intraday_data] No datetime index or Date column found")
            return pd.DataFrame()
    
    # Only filter if we have a datetime index
    if isinstance(vol_df.index, pd.DatetimeIndex):
        try:
            vol_df = vol_df.between_time("09:30", "16:00")
            print(f"[process_intraday_data] After time filtering: {vol_df.shape[0]} rows")
        except Exception as e:
            print(f"[process_intraday_data] Error filtering by time: {e}")
    
    vol_df.reset_index(inplace=True)
    
    # Ensure Date column exists and is datetime
    if 'Date' not in vol_df.columns:
        print("[process_intraday_data] No Date column after reset_index")
        if 'index' in vol_df.columns:
            vol_df.rename(columns={'index': 'Date'}, inplace=True)
            print("[process_intraday_data] Renamed 'index' to 'Date'")
    
    vol_df['Date'] = pd.to_datetime(vol_df['Date'], errors='coerce')
    print(f"[process_intraday_data] Valid dates: {vol_df.dropna(subset=['Date']).shape[0]}")

    # Check if we have any symbol column
    if 'symbol' not in vol_df.columns:
        print("[process_intraday_data] No symbol column found")
        return pd.DataFrame()
    
    unique_symbols = vol_df['symbol'].unique()
    print(f"[process_intraday_data] Found {len(unique_symbols)} unique symbols")

    resampled_list = []
    for symbol, group in vol_df.groupby('symbol'):
        print(f"[process_intraday_data] Processing symbol: {symbol}, rows: {len(group)}")
        
        # Skip empty groups
        if group.empty:
            print(f"[process_intraday_data] Empty group for symbol: {symbol}")
            continue
            
        # Skip groups with no valid dates
        valid_dates = group.dropna(subset=['Date'])
        if valid_dates.empty:
            print(f"[process_intraday_data] No valid dates for symbol: {symbol}")
            continue
            
        try:
            group = group.set_index('Date').sort_index()
            resampled = group.resample('15min').last()
            print(f"[process_intraday_data] Resampled {symbol}: {len(resampled)} rows")
            
            # Skip empty resampled results
            if resampled.empty:
                print(f"[process_intraday_data] Empty resampled result for {symbol}")
                continue
                
            resampled['symbol'] = symbol
            resampled_list.append(resampled)
        except Exception as e:
            print(f"[process_intraday_data] Error processing symbol {symbol}: {e}")
            continue

    # Check if we have any resampled data
    if not resampled_list:
        print("[process_intraday_data] No resampled data collected")
        return pd.DataFrame()
        
    print(f"[process_intraday_data] Collected {len(resampled_list)} resampled groups")
    
    try:
        resampled_df = pd.concat(resampled_list).reset_index()
        resampled_df = resampled_df.sort_values(['symbol', 'Date'])
        print(f"[process_intraday_data] Final shape after resampling: {resampled_df.shape}")
    except ValueError as e:
        print(f"[process_intraday_data] Concat error: {e}")
        # Try inspecting what's in the list
        for i, df in enumerate(resampled_list):
            print(f"DataFrame {i} shape: {df.shape}, columns: {df.columns.tolist()}")
        return pd.DataFrame()

    print("[process_intraday_data] Completed intraday resampling")

    # Cleanup
    del resampled_list, vol_df
    gc.collect()
    return resampled_df

def process_daily_rv_target(daily_rv_df):
    """
    Process the daily realized volatility data by shifting the target (next day log RV).
    """
    print("[process_daily_rv_target] Shifting daily RV target")
    daily_rv_df['date'] = pd.to_datetime(daily_rv_df['date']).dt.date
    daily_rv_df = daily_rv_df.sort_values(['symbol', 'date'])
    daily_rv_df['next_day_rv'] = daily_rv_df.groupby('symbol')['daily_rv'].shift(-1)
    daily_rv_df = daily_rv_df[['symbol', 'date', 'next_day_rv']]
    print("[process_daily_rv_target] Completed target shifting")
    return daily_rv_df

# ----------------------------
# VOLATILITY COMPUTATIONS
# ----------------------------

def compute_daily_rv(tickers, data_dir="data", EPS=1e-8):
    """
    Compute daily log-realized volatility for each ticker using full in-memory load per ticker.
    """
    print("[compute_daily_rv] Starting computation of daily RV")
    daily_list = []
    dataset = ds.dataset(data_dir, format="parquet", partitioning="hive")

    for ticker in tqdm(tickers, desc="Computing daily RV"):
        print(f"[compute_daily_rv] Processing ticker: {ticker}")
        try:
            table = dataset.to_table(filter=(ds.field("symbol") == ticker),
                                     columns=["timestamp", "close", "symbol"])
            df_ticker = table.to_pandas()
            del table
            gc.collect()
            if df_ticker.empty:
                print(f"[compute_daily_rv] No data found for {ticker}")
                continue

            df_ticker['Date'] = pd.to_datetime(df_ticker['timestamp'])
            df_ticker.drop(columns=['timestamp'], inplace=True)
            df_ticker = df_ticker.sort_values('Date')

            df_ticker['log_return'] = np.log(df_ticker['close'] / df_ticker['close'].shift(1))
            df_ticker = df_ticker.dropna(subset=['log_return'])
            df_ticker['day'] = df_ticker['Date'].dt.date
            df_ticker['sq_log_return'] = df_ticker['log_return'] ** 2

            daily = df_ticker.groupby('day')['sq_log_return'].sum().reset_index()
            daily['daily_rv'] = np.log(daily['sq_log_return'] + EPS)
            daily['symbol'] = ticker

            daily_list.append(daily[['symbol', 'day', 'daily_rv']])
            print(f"[compute_daily_rv] Finished: {ticker} — {len(daily)} rows")

            # Cleanup per ticker
            del df_ticker, daily
            gc.collect()
        except Exception as e:
            print(f"[compute_daily_rv] Error processing {ticker}: {e}")
            continue

    if daily_list:
        daily_rv_df = pd.concat(daily_list).sort_values(['day'])
        daily_rv_df.rename(columns={'day': 'date'}, inplace=True)
        print(f"[compute_daily_rv] Completed daily RV computation for {len(daily_rv_df)} records")
        return daily_rv_df
    else:
        print("[compute_daily_rv] No daily RV data computed")
        return None

def compute_all_volatilities(tickers, data_dir="data", EPS=1e-8):
    """
    Load all data for the given tickers at once and compute 10m, 30m, and 65m realized volatility.
    """
    print(f"[compute_all_volatilities] Computing volatilities for tickers: {tickers}")
    try:
        dataset = ds.dataset(data_dir, format="parquet", partitioning="hive")
        filter_expr = ds.field("symbol").isin(tickers)

        # Load full table directly
        table = dataset.to_table(filter=filter_expr, columns=["timestamp", "close", "symbol"])
        df_ticker = table.to_pandas()
        del table
        gc.collect()
        print(f"[compute_all_volatilities] Loaded full dataset with shape {df_ticker.shape}")
        
        # Add early check
        if df_ticker.empty:
            print("[compute_all_volatilities] Empty dataset loaded")
            return None

        df_ticker['Date'] = pd.to_datetime(df_ticker['timestamp'])
        df_ticker.drop(columns=['timestamp'], inplace=True)
        df_ticker = df_ticker.sort_values('Date')

        # Check for potential issues with close prices
        print(f"[compute_all_volatilities] Close price stats: min={df_ticker['close'].min()}, max={df_ticker['close'].max()}, null count={df_ticker['close'].isnull().sum()}")
        
        # Handle missing values more carefully
        if df_ticker['close'].isnull().any():
            print("[compute_all_volatilities] Found null close prices, interpolating...")
            # First group by symbol to avoid interpolating across different stocks
            for symbol, group in df_ticker.groupby('symbol'):
                mask = df_ticker['symbol'] == symbol
                df_ticker.loc[mask, 'close'] = group['close'].interpolate(method='linear')
                
        # Calculate log returns - careful with zeros or negative prices
        df_ticker['log_return'] = np.log(df_ticker['close'] / df_ticker['close'].shift(1))
        print(f"[compute_all_volatilities] Log returns: null count={df_ticker['log_return'].isnull().sum()}")
        
        # Calculate realized volatility over different windows
        df_ticker['10min_RV'] = np.log(df_ticker['log_return'].pow(2).rolling(window=10).sum() + EPS)
        df_ticker['30min_RV'] = np.log(df_ticker['log_return'].pow(2).rolling(window=30).sum() + EPS)
        df_ticker['65min_RV'] = np.log(df_ticker['log_return'].pow(2).rolling(window=65).sum() + EPS)
        
        # Get just the columns we need
        result_df = df_ticker[['Date', '10min_RV', '30min_RV', '65min_RV', 'symbol']].copy()
        
        # Drop nulls more carefully
        null_counts = result_df.isnull().sum()
        print(f"[compute_all_volatilities] Null counts before dropping: {null_counts}")
        result_df = result_df.dropna()
        print(f"[compute_all_volatilities] Computed volatility data with shape {result_df.shape}")
        
        # Set the date as index for future operations
        result_df.set_index('Date', inplace=True)

        del df_ticker
        gc.collect()
        return result_df
    except Exception as e:
        print(f"[compute_all_volatilities] Error processing tickers: {e}")
        return None
    
def finish_processing(data):
    data['Time'] = data['Date'].dt.time
    data = data.drop('Date', axis=1)

    # Create an empty list to store dataframes (more efficient than repeated pd.concat)
    result_dfs = []

    # Get unique symbols
    unique_symbols = data['symbol'].unique()
    symbol_chunk_size = 100  # Desired chunk size for the number of unique days

    # Pre-compute the target variables for all symbols at once
    # This avoids repeated groupby operations
    all_targets = data.groupby(['date', 'symbol'])[['next_day_rv', 'next_day_market_rv']].first()

    for symbol in tqdm(unique_symbols, desc="Processing Symbols"):
        # Filter data for this symbol
        symbol_data = data[data['symbol'] == symbol].sort_values(by='date').copy()

        all_days = symbol_data['date'].unique()

        # Determine the number of chunks for the current symbol (based on unique days)
        num_symbol_chunks = (len(all_days) + symbol_chunk_size - 1) // symbol_chunk_size

        for i in range(num_symbol_chunks):
            print(f"{i+1}/{num_symbol_chunks}", end='\r')
            start_index = i * symbol_chunk_size
            end_index = min((i + 1) * symbol_chunk_size, len(all_days))
            chunk_dates = all_days[start_index:end_index]

            # Filter data for the current symbol where the date is in the current chunk of dates
            symbol_data_chunk = symbol_data[symbol_data['date'].isin(chunk_dates)].copy()

            # Create pivot table for this symbol and date chunk
            try:
                chunk_pivot = symbol_data_chunk.pivot_table(
                    index=['date', 'symbol'],
                    columns='Time',
                    values=['10min_RV', '30min_RV', '65min_RV',
                            'market_10min_RV', 'market_30min_RV', 'market_65min_RV']
                )
                # Flatten the column index
                chunk_pivot.columns = [f'{col[0]}_{col[1]}' if col[0] else col[1] for col in chunk_pivot.columns]
            except ValueError:
                print(f"Warning: Not enough unique values in 'Date' for pivot on symbol '{symbol}', dates {chunk_dates[0]} to {chunk_dates[-1]}. Skipping this date chunk.")
                continue  # Skip to the next date chunk if pivot fails

            # Extract just this symbol's targets for the current date chunk
            try:
                symbol_targets_chunk = all_targets.loc[pd.IndexSlice[chunk_dates, symbol], :]
            except KeyError:
                print(f"Warning: No target data found for symbol '{symbol}', dates {chunk_dates[0]} to {chunk_dates[-1]}. Skipping this date chunk.")
                continue

            # Join with targets
            chunk_result = chunk_pivot.join(symbol_targets_chunk, how='left') # Using left join to keep all pivoted data

            # Append to results list (faster than pd.concat on each iteration)
            result_dfs.append(chunk_result)

            # Clean up to free memory
            del symbol_data_chunk, chunk_pivot, chunk_result, symbol_targets_chunk
            gc.collect()

    # Combine all results at the end (much faster than concatenating in each loop)
    data = pd.concat(result_dfs).reset_index()

    # Clean up
    del result_dfs
    gc.collect()
    
    return data

# ----------------------------
# MAIN EXECUTION
# ----------------------------

if __name__ == "__main__":
    # Load only a subset of tickers if needed for lower memory usage.
    all_tickers = load_sp500_tickers('data/info/all_sp500.csv')
    top_tickers = all_tickers[:100]

    print("all_tickers: ", len(all_tickers))
    print("top_tickers: ", len(top_tickers))

    chunk_size = 20
    max_workers = 8

    print("chunk_size: ", chunk_size)
    print("max_workers: ", max_workers)

    # PASS A: Compute the global market average in chunks.
    pass_a_chunked_market_avg(top_tickers, data_dir="data/processed_1min_top_sp500_bulk",
                                 chunk_size=chunk_size, market_parquet="market_avg.parquet", max_workers=max_workers)

    # PASS B: Process and merge with the global market average.
    dataset = pass_b_chunked_pipeline(top_tickers, market_parquet="market_avg.parquet",
                                     data_dir="data/processed_1min_top_sp500_bulk", chunk_size=chunk_size, max_workers=max_workers)

    # Compute market target: For each date, compute next_day_market_rv as the average next_day_rv across stocks.
    print("[MAIN] Computing market target")
    dataset['date'] = pd.to_datetime(dataset['date'])
    market_target = dataset.groupby(dataset['date'].dt.date)['next_day_rv'].mean().reset_index()
    market_target.rename(columns={'next_day_rv': 'next_day_market_rv', 'date': 'day_only'}, inplace=True)
    dataset['day_only'] = dataset['date'].dt.date
    dataset = pd.merge(dataset, market_target, on='day_only', how='left')
    dataset.drop(columns=['day_only'], inplace=True)
    
    dataset = finish_processing(dataset)
    
    print(dataset.head())
    
    dataset.to_parquet("processed_dataset.parquet")
    print("[MAIN] Processed dataset saved to processed_dataset.parquet")