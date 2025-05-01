import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ----------------------------
# DATA LOADING FUNCTIONS
# ----------------------------

# ----------------------------
# PASS A (CHUNKED): Build Global Market Averages
# ----------------------------

def process_chunk_a(chunk_tickers, i, data_dir):
    print(f"  Processing chunk: {chunk_tickers}")
    chunk_vol = compute_all_volatilities(chunk_tickers, data_dir=data_dir)
    if chunk_vol is None:
        return None, None
    chunk_pivot = process_intraday_data(chunk_vol)
    chunk_file = f"temp_pivot_{i}.parquet"
    chunk_pivot.to_parquet(chunk_file)
    return chunk_file, chunk_tickers

def pass_a_chunked_market_avg(all_tickers, data_dir="data", chunk_size=10, market_parquet="market_avg.parquet"):
    print("\n=== PASS A: Chunked approach to compute GLOBAL market average ===")
    pivot_files = []
    
    # Use ThreadPoolExecutor to process chunks concurrently
    with ThreadPoolExecutor() as executor:
        futures = {}
        for i in range(0, len(all_tickers), chunk_size):
            chunk_tickers = all_tickers[i : i + chunk_size]
            futures[executor.submit(process_chunk_a, chunk_tickers, i, data_dir)] = i
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            chunk_file, tickers = future.result()
            if chunk_file is not None:
                pivot_files.append(chunk_file)
    
    print("\nUnifying partial pivots to compute the global market average...")
    pivot_dfs = [pd.read_parquet(fpath) for fpath in pivot_files]
    full_pivot = pd.concat(pivot_dfs, ignore_index=True).drop_duplicates()
    market_avg = full_pivot.groupby('date').mean(numeric_only=True).reset_index()
    market_avg.columns = ['date'] + [f"market_{col}" for col in market_avg.columns if col != 'date']
    market_avg.to_parquet(market_parquet)
    print(f"Global market average saved to {market_parquet}")
    
    for fpath in pivot_files:
        os.remove(fpath)

# ----------------------------
# PASS B: Process Tickers in Chunks, Merging with Global Market
# ----------------------------

def process_chunk_b(chunk_tickers, global_market, data_dir):
    print(f"\nProcessing chunk: {chunk_tickers}")
    chunk_vol = compute_all_volatilities(chunk_tickers, data_dir=data_dir)
    if chunk_vol is None:
        return None
    chunk_daily_rv = compute_daily_rv(chunk_tickers, data_dir=data_dir)
    chunk_pivot = process_intraday_data(chunk_vol)
    chunk_pivot = pd.merge(chunk_pivot, global_market, on='date', how='left')
    chunk_daily_rv = process_daily_rv_target(chunk_daily_rv)
    chunk_dataset = merge_intraday_and_daily(chunk_pivot, chunk_daily_rv)
    return chunk_dataset

def pass_b_chunked_pipeline(all_tickers, market_parquet="market_avg.parquet", data_dir="data", chunk_size=10):
    print("\n=== PASS B: Merging with Global Market Averages ===")
    global_market = pd.read_parquet(market_parquet)
    final_datasets = []

    with ThreadPoolExecutor() as executor:
        futures = {}
        for i in range(0, len(all_tickers), chunk_size):
            chunk_tickers = all_tickers[i : i + chunk_size]
            future = executor.submit(process_chunk_b, chunk_tickers, global_market, data_dir)
            futures[future] = chunk_tickers

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            result = future.result()
            if result is not None:
                final_datasets.append(result)

    dataset = pd.concat(final_datasets, ignore_index=True)
    return dataset


def load_sp500_tickers(filepath):
    tickers = pd.read_csv(filepath)['Symbol'].tolist()
    return tickers

def load_data(tickers, data_dir="data", batch_size=10000):
    """
    Load data for the specified tickers in chunks using PyArrow batches.
    Each batch is processed (timestamp conversion, sorting) and then concatenated.
    """
    print("\nLoading data for:", tickers)
    dataset = ds.dataset(data_dir, format="parquet", partitioning="hive")
    filter_expr = ds.field("symbol").isin(tickers)
    scanner = dataset.scanner(filter=filter_expr, batch_size=batch_size)
    
    batches = []
    for batch in scanner.to_batches():
        df_batch = batch.to_pandas()
        df_batch['Date'] = pd.to_datetime(df_batch['timestamp'])
        df_batch.drop(columns=['timestamp'], inplace=True)
        batches.append(df_batch)
    tickers_df = pd.concat(batches).sort_values(['Date', 'symbol'])
    tickers_df.set_index('Date', inplace=True)
    return tickers_df
# ----------------------------
# INCREMENTAL VOLATILITY COMPUTATIONS
# ----------------------------

def compute_volatility_for_ticker(ticker, data_dir="data", EPS=1e-8, batch_size=10000):
    """
    Incrementally load data for one ticker, compute its 15-minute and 30-minute realized volatility,
    and return a DataFrame with Date, 15min_RV, 30min_RV, and symbol.
    """
    try:
        dataset = ds.dataset(data_dir, format="parquet", partitioning="hive")
        filter_expr = ds.field("symbol") == ticker
        scanner = dataset.scanner(filter=filter_expr, batch_size=batch_size)
        
        ticker_batches = []
        for batch in scanner.to_batches():
            df_batch = batch.to_pandas()
            ticker_batches.append(df_batch)
        if not ticker_batches:
            return None
        
        df_ticker = pd.concat(ticker_batches)
        df_ticker['Date'] = pd.to_datetime(df_ticker['timestamp'])
        df_ticker.drop(columns=['timestamp'], inplace=True)
        df_ticker = df_ticker.sort_values('Date')
        
        df_ticker['close'] = df_ticker['close'].interpolate(method='linear')
        df_ticker['log_return'] = np.log(df_ticker['close'] / df_ticker['close'].shift(1))
        df_ticker['15min_RV'] = np.log(df_ticker['log_return'].pow(2).rolling(window=15).sum() + EPS)
        df_ticker['30min_RV'] = np.log(df_ticker['log_return'].pow(2).rolling(window=30).sum() + EPS)
        result_df = df_ticker[['Date', '15min_RV', '30min_RV']].dropna().copy()
        result_df['symbol'] = ticker
        result_df.set_index('Date', inplace=True)
        return result_df
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


def compute_all_volatilities(tickers, data_dir="data", n_threads=4, batch_size=10000, EPS=1e-8):
    """
    Incrementally load data for multiple tickers, compute the 15-minute and 30-minute realized volatility,
    and return a DataFrame with Date, 15min_RV, 30min_RV, and symbol.
    """
    try:
        dataset = ds.dataset(data_dir, format="parquet", partitioning="hive")
        filter_expr = ds.field("symbol").isin(tickers)
        scanner = dataset.scanner(filter=filter_expr, batch_size=batch_size)
        
        ticker_batches = []
        for batch in scanner.to_batches():
            df_batch = batch.to_pandas()
            ticker_batches.append(df_batch)
        if not ticker_batches:
            return None
        
        df_ticker = pd.concat(ticker_batches)
        df_ticker['Date'] = pd.to_datetime(df_ticker['timestamp'])
        df_ticker.drop(columns=['timestamp'], inplace=True)
        df_ticker = df_ticker.sort_values('Date')
        
        df_ticker['close'] = df_ticker['close'].interpolate(method='linear')
        df_ticker['log_return'] = np.log(df_ticker['close'] / df_ticker['close'].shift(1))
        df_ticker['15min_RV'] = np.log(df_ticker['log_return'].pow(2).rolling(window=15).sum() + EPS)
        df_ticker['30min_RV'] = np.log(df_ticker['log_return'].pow(2).rolling(window=30).sum() + EPS)
        result_df = df_ticker[['Date', '15min_RV', '30min_RV', 'symbol']].dropna().copy()
        result_df.set_index('Date', inplace=True)
        return result_df
    except Exception as e:
        print(f"Error processing tickers: {e}")
        return None


def compute_daily_rv(tickers, data_dir="data", EPS=1e-8, batch_size=10000):
    """
    Compute daily log-realized volatility for each ticker incrementally.
    For each ticker, load its data in chunks, compute 1-min log returns and group by day.
    """
    daily_list = []
    dataset = ds.dataset(data_dir, format="parquet", partitioning="hive")
    for ticker in tqdm(tickers, desc="Computing daily RV"):
        filter_expr = ds.field("symbol") == ticker
        scanner = dataset.scanner(filter=filter_expr, batch_size=batch_size)
        ticker_batches = []
        for batch in scanner.to_batches():
            df_batch = batch.to_pandas()
            ticker_batches.append(df_batch)
        if not ticker_batches:
            continue
        df_ticker = pd.concat(ticker_batches)
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
    
    if daily_list:
        daily_rv_df = pd.concat(daily_list).sort_values(['day'])
        daily_rv_df.rename(columns={'day': 'date'}, inplace=True)
        return daily_rv_df
    else:
        return None

# ----------------------------
# INTRADAY DATA PROCESSING FUNCTIONS
# ----------------------------

def process_intraday_data(volatility_df):
    """
    Process intraday volatility data by resampling into non-overlapping 15-min bins
    and pivoting so that each row is (symbol, date) with columns for both 15-min and 30-min RV.
    """
    vol_df = volatility_df.copy()
    vol_df.index = pd.to_datetime(vol_df.index, errors='coerce')
    vol_df = vol_df.between_time("09:30", "16:00")
    vol_df.reset_index(inplace=True)
    vol_df['Date'] = pd.to_datetime(vol_df['Date'], errors='coerce')
    vol_df['day_only'] = vol_df['Date'].dt.date

    resampled_list = []
    for (symbol, day), group in vol_df.groupby(['symbol', 'day_only']):
        group = group.set_index('Date').sort_index()
        # Resample into non-overlapping 15-min bins
        resampled = group.resample('15min').last()
        resampled['symbol'] = symbol
        resampled['day_only'] = day
        resampled_list.append(resampled)
    resampled_df = pd.concat(resampled_list).reset_index()
    resampled_df = resampled_df.sort_values(['symbol', 'Date'])
    resampled_df['bucket'] = resampled_df.groupby(['symbol', 'day_only']).cumcount()

    # Pivot for 15min RV
    pivot_15 = resampled_df.pivot(index=['symbol', 'day_only'], columns='bucket', values='15min_RV')
    pivot_15.columns = [f"rv_15m_{b}" for b in pivot_15.columns]
    
    # Pivot for 30min RV
    pivot_30 = resampled_df.pivot(index=['symbol', 'day_only'], columns='bucket', values='30min_RV')
    pivot_30.columns = [f"rv_30m_{b}" for b in pivot_30.columns]
    
    # Merge the two pivots on symbol and day_only
    pivot_df = pd.merge(pivot_15.reset_index(), pivot_30.reset_index(), on=['symbol', 'day_only'])
    pivot_df = pivot_df.rename(columns={'day_only': 'date'})
    return pivot_df


def augment_market_features(pivot_df):
    """
    Augment the pivoted intraday data with market features.
    For each date, compute the average for each 15-min and 30-min bucket across all stocks.
    """
    # Compute market average for 15min RV buckets
    market_15 = pivot_df.filter(like="rv_15m").groupby(pivot_df['date']).mean().reset_index()
    market_15.columns = ['date'] + [f"market_rv_15m_{i}" for i in range(len(market_15.columns)-1)]
    
    # Compute market average for 30min RV buckets
    market_30 = pivot_df.filter(like="rv_30m").groupby(pivot_df['date']).mean().reset_index()
    market_30.columns = ['date'] + [f"market_rv_30m_{i}" for i in range(len(market_30.columns)-1)]
    
    # Merge market features with the original pivot data
    pivot_df = pd.merge(pivot_df, market_15, on='date', how='inner')
    pivot_df = pd.merge(pivot_df, market_30, on='date', how='inner')
    return pivot_df


def process_daily_rv_target(daily_rv_df):
    """
    Process the daily realized volatility data by shifting the target (next day log RV).
    """
    daily_rv_df['date'] = pd.to_datetime(daily_rv_df['date']).dt.date
    daily_rv_df = daily_rv_df.sort_values(['symbol', 'date'])
    daily_rv_df['next_day_rv'] = daily_rv_df.groupby('symbol')['daily_rv'].shift(-1)
    daily_rv_df = daily_rv_df[['symbol', 'date', 'next_day_rv']]
    return daily_rv_df

def merge_intraday_and_daily(pivot_df, daily_rv_df):
    """
    Merge the intraday features (pivoted and augmented) with the daily target.
    """
    dataset = pd.merge(pivot_df, daily_rv_df, on=['symbol', 'date'], how='inner')
    dataset = dataset.dropna().reset_index(drop=True)
    dataset = dataset.sort_values(['date', 'symbol'])
    return dataset

# ----------------------------
# OTHER DATA PROCESSING FUNCTIONS (unchanged)
# ----------------------------

def create_sequences(features, target, lookback=30):
    """
    Create sequences. If lookback == 1, then each sample is one day,
    and no additional sequence dimension is added.
    Otherwise, uses a sliding window.
    """
    X, y = [], []
    if lookback == 1:
        X = features[lookback:]
        y = target[lookback:]
    else:
        for i in range(lookback, len(features)):
            X.append(features[i - lookback:i])
            y.append(target[i])
        X = np.array(X)
        y = np.array(y)
                    
    return X, y

def winsorize_data(train_df, val_df, columns):
    """
    Winsorize the DataFrames based on quantile thresholds computed from the training set.
    (Data leakage prevention: quantiles are computed on train_val_df only)
    """
    for col in columns:
        lower = train_df[col].quantile(0.005)
        upper = train_df[col].quantile(0.995)
        train_df[col] = train_df[col].clip(lower, upper)
        val_df[col] = val_df[col].clip(lower, upper)
    return train_df, val_df

# ----------------------------
# CUSTOM PYTORCH DATASET FOR STREAMING DATA
# ----------------------------

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset that loads data from a Parquet file on demand.
    This is useful when the processed training data is too large to fit in memory.
    """
    def __init__(self, data_file, feature_cols, target_col, transform=None):
        self.data = pd.read_parquet(data_file)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = sample[self.feature_cols].values.astype(np.float32)
        target = np.array(sample[self.target_col], dtype=np.float32)
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features), torch.tensor(target)