# MarketRV

## Volatility Forecasting with Machine Learning and Intraday Commonality

This repository contains an implementation of the methods described in the paper "Volatility Forecasting with Machine Learning and Intraday Commonality" by Zhang, Zhang, Cucuringu, and Qian.

The project focuses on leveraging machine learning models and exploiting intraday commonality to forecast realized volatility in financial markets.

## Paper Citation

Zhang, C., Zhang, Y., Cucuringu, M., & Qian, Z. (2024). Volatility forecasting with machine learning and intraday commonality. Journal of Financial Econometrics, 22(2), 492-530. https://www.google.com/search?q=https://doi.org/10.1093/jjfinec/nbad005


## Features

* Implementation of volatility forecasting models, including linear models (Linear Regression, Lasso) and potentially Neural Networks (based on the paper's focus and the use of PyTorch in the notebook).
* Handling and processing of intraday financial data.
* Integration of commonality features across different stocks.
* Utilizes time-series cross-validation (TimeSeriesSplit).
* Evaluation using standard regression metrics (R-squared, Mean Squared Error, Mean Absolute Error).
* Includes data preprocessing steps like scaling and one-hot encoding.

## Data Processing Pipeline (In-depth)

The raw intraday financial data undergoes a multi-stage processing pipeline handled by the provided Python scripts before being used for model training in the Jupyter notebook. This pipeline is designed to handle large volumes of minute-level data efficiently and prepare it according to the requirements of the forecasting models, including creating features related to intraday commonality and lagged volatilities.

The overall process is orchestrated by run_data_processing.py and involves the following key stages:

1.  **Raw Data Collection (bulk_polygon_data.py):**
    * This script is responsible for fetching the initial, raw 1-minute aggregate bar data for a specified list of stock tickers over a given date range.
    * It interacts with the Polygon.io API using the polygon-api-client.
    * For each ticker, it retrieves minute-level data including open, high, low, close, volume, volume-weighted average price (VWAP), and trade count.
    * The fetched data is converted into pandas DataFrames, columns are renamed for clarity (e.g., t to timestamp), and additional time-based columns (year, month, day, symbol) are added.
    * The raw data is saved into partitioned Parquet files. Parquet is used for its efficiency in storing tabular data and its ability to handle partitioning, which helps in managing and querying large datasets by symbol, year, and month. This structure facilitates subsequent processing steps that might operate on subsets of the data.

2.  **Missing Data Handling (process_missing_data.py):**
    * This script addresses the issue of missing minute-level observations in the raw data, which is crucial for accurately calculating realized volatility and other intraday features.
    * It reads the raw partitioned Parquet data.
    * It uses the pandas-market-calendars library to determine the exact NYSE trading minutes between market open (typically 9:30 AM) and market close (4:00 PM) for each trading day in the date range.
    * For each stock and each trading day, it generates a complete minute-by-minute timestamp index for the trading session.
    * It then merges the available raw data onto this complete index, effectively identifying gaps.
    * A function (fill_missing_minute_data) is applied to fill these missing minute data points. The exact filling logic (e.g., forward fill, backward fill, interpolation) is implemented within this function to create a continuous time series.
    * The cleaned and filled data for each symbol is saved into new partitioned Parquet files, maintaining the symbol, year, and month partitioning structure. This output serves as the input for the next processing stage. The script uses ProcessPoolExecutor to parallelize the processing across different stock symbols, speeding up the cleaning process.

3.  **Core Processing & Feature Engineering (data_processing.py):**
    * This script implements the core logic for transforming the cleaned intraday data into the features and target variable required for volatility forecasting. It follows a two-pass approach, referred to as Pass A and Pass B, likely to handle the computation of global market-level features which require processing data across *all* stocks before combining them with individual stock data.
    * **Pass A (Chunked Global Market Average Computation):**
        * The pass_a_chunked_market_avg function orchestrates this pass.
        * It processes the cleaned intraday data in chunks of tickers to manage memory usage.
        * For each chunk, it calls functions (compute_all_volatilities and process_intraday_data - details of these functions are in the script but not fully visible in the snippet, but they likely compute realized volatility from the minute data and aggregate/transform it to create intraday patterns or features).
        * The processed data for each chunk is saved temporarily.
        * Finally, the temporary files are combined to compute a "global market average" (market_avg.parquet). This market average likely represents the overall market volatility or a common component derived from aggregating intraday data across all stocks.
    * **Pass B (Chunked Stock Processing and Merging):**
        * The pass_b_chunked_pipeline function manages this pass.
        * It again processes stocks in chunks.
        * For each stock, it loads its cleaned intraday data.
        * It merges this individual stock data with the global market average computed in Pass A. This step integrates the commonality aspect into the dataset.
        * Stock-specific features are likely computed or finalized at this stage (e.g., lagged RVs, intraday pattern features).
        * The script includes winsorize_data, which applies winsorization to specified columns based on quantiles calculated from the training data. This is a common technique to limit the impact of extreme outliers.
        * The prepare_dataset function is used to structure the processed data into feature matrix X and target vector y pairs for model input.
        * The script also contains a TimeSeriesDataset class, suggesting support for loading this potentially large processed data efficiently for training, although the primary output of pass_b_chunked_pipeline appears to be a complete pandas DataFrame.
    * This script utilizes ThreadPoolExecutor for concurrent processing of ticker chunks in both Pass A and Pass B, aiming to improve performance.

4.  **Pipeline Orchestration (run_data_processing.py):**
    * This is the main script executed to run the entire data processing pipeline.
    * It loads the list of relevant tickers (e.g., S&P 500 constituents).
    * It defines parameters for chunking and parallel processing (chunk_size, max_workers).
    * It calls the pass_a_chunked_market_avg function from data_processing (3).py to compute the global market average.
    * It then calls the pass_b_chunked_pipeline function from data_processing (3).py to process individual stocks and merge the market average data, resulting in the final dataset.
    * Crucially, run_data_processing.py also computes the next_day_market_rv target variable. It groups the processed data by date and calculates the average next_day_rv across all stocks for that day. This aggregate market volatility is then merged back into the dataset as a potential feature or target component related to market commonality.
    * The final output dataset, ready for model training, is prepared by this script.

In summary, the data processing pipeline is a multi-step process that starts with fetching raw data, rigorously cleans it by handling missing points to create complete intraday series, computes both individual stock and global market volatility features, integrates the commonality aspect by merging market averages, applies outlier treatment, and finally structures the data into a format suitable for input into machine learning forecasting models. The use of chunking and parallel processing is implemented to manage the computational demands of high-frequency data.

## Requirements

To run this code, you need the following libraries installed:

* Python 3.x
* warnings
* numpy
* pandas
* matplotlib
* seaborn
* torch (PyTorch)
* scikit-learn
* scipy
* tqdm
* random
* polygon-api-client (for bulk_polygon_data.py)
* pyarrow (for Parquet handling)
* dask (potentially used in data_processing (3).py for larger-than-memory data, although ThreadPoolExecutor is primarily shown)
* pandas-market-calendars (for process_missing_data.py)

You can install most of the required libraries using pip:

bash
pip install numpy pandas matplotlib seaborn torch scikit-learn scipy tqdm polygon-api-client pyarrow dask pandas-market-calendars
Note that bulk_polygon_data.py requires setting the POLYGON_API_KEY environment variable to fetch data from Polygon.io.

Installation
Clone this repository:
Bash

git clone <repository_url>
cd <repository_folder>
(Replace <repository_url> and <repository_folder> with your actual repository details)
Install the required dependencies (see Requirements).
Set your POLYGON_API_KEY environment variable if you plan to run the data collection script:
Bash

export POLYGON_API_KEY="your_api_key_here"
(Replace "your_api_key_here" with your actual Polygon.io API key)
Data
The raw data is intended to be fetched using the bulk_polygon_data.py script from Polygon.io. The subsequent processing scripts assume the raw data is stored in the directory specified by RAW_DATA_DIR in process_missing_data.py (and similarly for processed data directories like PROCESSED_DATA_DIR in other scripts).

To use your own data:

If using Polygon.io, configure bulk_polygon_data.py with your ticker list and date range, set your API key, and run the script to generate raw data files in the specified OUTPUT_DIR.
If using a different data source, format your raw minute-level data to be compatible with the structure expected by process_missing_data.py (ideally, partitioned Parquet files by symbol, year, month with columns like timestamp, open, high, low, close, volume, symbol) and place it in the directory specified by RAW_DATA_DIR.
Configure input and output directories in process_missing_data.py and data_processing.py to match your file locations.
Run the data processing pipeline scripts (run_data_processing.py) to generate the final processed dataset.
Update the data loading path in the MARKET_RV.ipynb notebook to point to the location of the final processed dataset.
Usage
Ensure you have completed the Data Processing Pipeline steps and the final processed data is ready.
Open the Jupyter notebook MARKET_RV.ipynb.1   
1.
What is the ipynb Jupyter Notebook File Extension and How to Open It? | Saturn Cloud Blog

saturncloud.io

Update the data loading path in the notebook to point to your processed dataset file or directory.
Run the cells sequentially in the notebook.
The notebook performs the following steps:

Loads the processed data.
Sets up the time-series cross-validation splits.
Applies final scaling and encoding to features using methods appropriate for time-series validation (e.g., fitting scalers only on the training fold).
Trains and evaluates the specified machine learning models (Linear Models, potentially Neural Networks) for each cross-validation fold.
Calculates and prints performance metrics (R-squared, MSE, MAE) aggregated across folds and potentially per ticker.
Results
The data processing scripts produce processed data files used as input for the modeling notebook. The results of the model training and evaluation, including R-squared, MSE, and MAE for the models, are printed within the MARKET_RV (2).ipynb notebook cells. The notebook includes code to calculate and display average R-squared per ticker on the validation set across folds, providing insights into per-stock performance.

Project Structure
.
├── MARKET_RV.ipynb      # Main notebook for model training and evaluation
├── bulk_polygon_data.py     # Script for fetching raw data from Polygon.io API
├── data_processing.py   # Contains core data processing, feature engineering (Pass A & B), and data preparation logic
├── process_missing_data.py  # Script for handling missing minute-level data and ensuring data completeness
├── run_data_processing.py   # Orchestrates the data processing pipeline, including computing market targets
└── README.md                # This README file
(Include any specific data directories like data/raw_1min_top_sp500_bulk/ or data/processed_1min_top_sp500_bulk/ here if they are part of the repository structure, along with any example data files if applicable)