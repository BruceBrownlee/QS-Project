# QUANT SCIENCE LLC 
# THE QUANT SCIENTIST PRO ALGORITHMIC TRADER 
# LEVEL 2 PROGRAM
# ****
# CLINIC #5: QUANTAMENTAL FINANCIAL DATABASE SETUP WITH QSCONNECT


# * INSTALLATION INSTRUCTIONS ----
# - Install the `qsconnect` package from GitHub:
#     https://github.com/quant-science/QSConnect
# - Run These Commands For Fresh Install:
#     pip uninstall qsconnect
#     pip install git+https://github.com/quant-science/QSConnect.git

# * FINANCIAL DATA:
# - We use the Financial Modeling Prep (FMP) API to access financial data. 
# - We have a partnership with FMP for 30% OFF. 
#   DISCOUNT LINK: https://site.financialmodelingprep.com/developer/docs/pricing?couponCode=quantscience

# * LIBRARIES ----

# system imports
import os
from pathlib import Path

# Data Analysis
import pandas as pd
import pytimetk as tk
import polars as pl

# Files

import yaml


# QSConnect 
from qsconnect import Client


# * ENVIRONMENT VARIABLES ----

# Constants
TODAY = "2025-07-31"

# FMP API Key
# %%
os.environ['FMP_API_KEY'] = yaml.safe_load(open('credentials.yml', encoding='utf-8'))['fmp_api_key']

# Check to see that we have an FMP_API_KEY
os.environ['FMP_API_KEY']

# QSCONNECT PATH TO CACHE & DATABASES
os.environ['QSCONNECT_ROOT'] = str(Path.cwd() / 'data' / 'qsconnect')

os.environ['QSCONNECT_ROOT'] = str(Path.cwd() / 'data' / 'qsconnect-demo')

# * 1.0 QSCONNECT CLIENT ----

# Instantiate Client
# %%
client = Client()  # This will automatically create the data/qsconnect directory if it does not exist

client

# Cache Directories
client.cache_dir
# %%
client.user_cache_dir

client.fmp_cache_dir

client.yahoo_cache_dir

# Database Directory
client.database_dir
# %%

# * 2.0 BUILD YOUR FMP CACHE FILES (DATA LAKE) ----

# * Step 1: Get the list of all available stock tickers from FMP

# This fetches stock, etf, trust and fund tickers
# %%
stock_list = client.stock_list(
    security_type="stock", 
    override_date=TODAY
)
# %%
stock_list

stock_list.glimpse()

stock_list.shape
# stock_list.glimpse()

stock_list[stock_list['symbol'] == 'NVDA']

# Asset Types

stock_list['type'].value_counts()


# Exchange Types

stock_list[['exchange', 'exchangeShortName']].value_counts()

# What is stock_list?
type(stock_list)

# Create a list of symbols to get prices for
# %%
asset_types = ['stock'] 
exchange_types = ['NASDAQ', 'NYSE', 'AMEX']

symbols_to_get = []

for asset_type in asset_types:
    for exchange_type in exchange_types:
        filtered_symbols = stock_list[
            (stock_list['type'] == asset_type) & 
            (stock_list['exchangeShortName'] == exchange_type)
        ]['symbol'].tolist()
        symbols_to_get.extend(filtered_symbols)

symbols_to_get[0:10]  # Display the first 10 symbols

len(symbols_to_get)  # Total number of symbols to get prices for

# Add SPY and QQQ to the list for Benchmarking
symbols_to_get = ['SPY', 'QQQ'] + symbols_to_get

symbols_to_get
# %%
# * Step 2: Get the stock prices for all tickers

# * FMP Prices API

# FMP API Notes: 
#  1. This will take 10 minutes to run as it fetches historical prices for 12,000+ assets and convert to a 3 GB parquet file.
#  2. You can adjust the `api_calls_per_minute` parameter based on your FMP plan to avoid hitting rate limits. My limit is 3000 calls per minute, so I set it to 2000 here because I saw I was getting ERRORS when I set it to 3000.

prices_path = client.historical_prices(
    symbols=symbols_to_get[:2000],
    start_date="2000-01-01",
    end_date=TODAY,
    api_calls_per_minute=2000, # Adjust based on your FMP plan
    cache=True,  # This will cache the file in the FMP cache directory
    return_path=True,  # This will return the path to the cached file instead of the DataFrame (which is useful for large datasets)
)

# Inspect the cached file using Polars Lazy Evaluation for large datasets

# * Your cached file path will be different, so you need to adjust this path accordingly

prices_path =  "/Users/brucebrownlee/Dev/GitHub/Resident/QS-Project/Clinic-05/" + "data/qsconnect-demo/cache/fmp/historical-prices-2025-07-28.parquet"

print(prices_path)

# Detect Missing Symbols

# Get the head of the DataFrame to see the first few rows
lazy_df = pl.scan_parquet(prices_path)

lazy_df.head()

lazy_df.head().collect().to_pandas()

# Get the unique symbols in the DataFrame
lazy_df = (
    pl.scan_parquet(prices_path)
    .select(pl.col("symbol").unique())
)   

unique_symbols = lazy_df.collect().to_pandas()

unique_symbols

# Difference between symbols_to_get and unique_symbols
missing_symbols = list(set(symbols_to_get) - set(unique_symbols['symbol'].tolist()))

pd.DataFrame(missing_symbols, columns=['symbol'])


# Locate a single stock ticker, e.g., Apple (AAPL)

lazy_df = (
    pl.scan_parquet(prices_path)
    .filter(pl.col("symbol") == "AAPL")
)

df = lazy_df.collect().to_pandas()

df

df.glimpse()

df \
    .groupby('symbol') \
    .plot_timeseries(
        date_column='date',
        value_column='close',
        title='Stock Price',
    )


# * Yahoo Finance API Downloads

# Note: This is a long running operation (about 10 minutes)

client.historical_prices_yahoo(
    symbols=symbols_to_get[:2000],  
    start_date="2000-01-01",
    end_date=TODAY,
    batch_size=100,
)


# * Step 3: Get the fundamental data for all tickers

# - WARNING: Long running operation (134 minutes)
# - Note: Must wait 10 seconds between bulk API requests

client.fetch_bulk_financial_statements(
    statement_type='all', # all available statements: income, balance, cash flow, ...
    periods='all', # annual and quarter 
    
    # start_year=2000,
    start_year=2024,  
    
    end_year=pd.to_datetime(TODAY).year,
    api_buffer_seconds=12,  # Adjust based on your FMP plan to avoid hitting rate limits
)

# * Step 4: Detect Cached Files

client.detect_cached_files()

client.summarize_cached_files()


# Detect Missing Cached Files

df = client.detect_missing_cached_files(
    statement_type="all",
    periods='all',
    start_year=2001,
    end_year=2025, # Try with 2025
    override_date=TODAY,
)

# Clean up Cached Files
help(client.delete_cached_files)

# client.delete_cached_files(df.iloc[[0]])

# * 2.0 BUILD YOUR QUANTAMENTAL FINANCIAL DATABASE ----


# * Step 1: Identify the files you want to include in your database

all_cached_files = client.detect_cached_files(
    # start_date=TODAY,
    # end_date=pd.to_datetime(TODAY) + pd.DateOffset(days=1),  
)

all_cached_files[all_cached_files['file_name'].str.startswith('historical-prices')]

all_cached_files[all_cached_files['file_name'].str.contains('2006')]

# * Step 2: Create a database from the cached files

# This takes a 5 minutes to run as it processes all the cached files and creates the qsconnect DuckDB database

errors_df = client.load_cached_files_to_database(
    cached_files_df=all_cached_files, 
    fresh=True, # Set to True to create a fresh database; set to False to append to an existing database
)

errors_df # If this is empty, then all files were loaded successfully; otherwise a DataFrame with errors will be returned

# * Step 3: Inspect the database

# Connect to the database
client.connect_to_database()

# List All Database Tables
client.list_database_tables()

# Glimpse All Database Tables
client.glimpse_all_database_tables(max_colwidth=25)

# Missing Values Report
client.missing_values_all_database_tables(max_colwidth=25)

# Missing Calendar Years Report for FMP Fundamental Data
client.missing_calendar_years_all_database_tables(
    start_year=2000, 
    end_year=2025,
)

# Quality Report
client.database_quality_report(
    start_year=2000, 
    end_year=2025
)

# Run SQL Queries on the Database
conn = client.connect_to_database()

conn.execute("""
    SELECT *
    FROM historical_prices_fmp
    LIMIT 5
""").fetchdf()

# Disconnect from the database
client.close_database_connection()

# DBCode Extension Demo
# - Use the DBCode extension to examine the database tables


# * 3.0 BUILD YOUR ZIPLINE BUNDLE ----

# * Step 1: Create a Zipline Bundle from the FMP Database

# Connect to the database
client.connect_to_database()

# Create a Zipline Bundle from the database
client.ingest_zipline_bundle_from_fmp_tables(bundle_name="qspro_demo_historical_prices_fmp")

# Disconnect from the database
client.close_database_connection()

# * Step 2: Open the Zipline hidden folder

# Mac users can find the hidden folder at:
#   1. Open Finder.
#   2. Press `Command + Shift + .` (period) to toggle hidden files visibility.
#   3. Navigate to the `~/.zipline` directory.

# Windows users can find the hidden folder at:
#   C:\Users\<YourUsername>\.zipline


# %%
