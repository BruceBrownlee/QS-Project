# QUANT SCIENCE LLC 
# THE QUANT SCIENTIST PRO ALGORITHMIC TRADER 
# LEVEL 2 PROGRAM
# ****
# CLINIC #5: UNIVERSE SELECTION AND CLEANING PIPELINE WITH QSRESEARCH

# * Goal:
#   - Implement a universe selection and cleaning pipeline using the QSResearch library.
#   - This will help us filter and preprocess financial data use in the QSResearch Quant Lab coming in Clinic #6.

# * INSTALLATION INSTRUCTIONS ----
# - Install the `qsresearch` package from GitHub:
#     https://github.com/quant-science/QSResearch
# - Run These Commands For Fresh Install:
#     pip uninstall qsresearch
#     pip install git+https://github.com/quant-science/QSResearch.git


# * LIBRARIES ----
# %%
# Data Analysis
import pandas as pd
import pytimetk as tk
import logging 

# Files
import os
from pathlib import Path

# QSConnect
from qsconnect import Client
# %%


# QSResearch
from qsresearch.preprocessors import universe_screener, preprocess_price_data

# * ENVIRONMENT VARIABLES ----

# Constants 
TODAY = "2025-07-31"
print(TODAY)

# Set Environment Variables
os.environ["QSCONNECT_ROOT"] = str(Path.cwd() / "data" / "qsconnect-demo")

# ForPATH_DIR = Path.cwd() Checkpointing
# PATH_DIR = Path(__file__).parent
PATH_DIR = Path.home() / "Dev" / "GitHub" / "Resident" / "QS-Project" / "Clinic-05"

print(PATH_DIR)

# Connect to database
client = Client()

conn = client.connect_to_database()

conn

type(conn)

# Load data

data = client.collect_database_table("historical_prices_fmp")
data

client.close_database_connection()


# * 1.0 UNIVERSE SELECTION

#   - Screens a stock universe based on a lookback period from a given end date, applying filters for volume, momentum, abnormal price movements, volatility, minimum last price, and minimum average price

# - Function Definition: https://github.com/quant-science/QSResearch/blob/d3467148a6524bf342b6dcb707a0f66b7a4e0b8d/qsresearch/preprocessors/universe_screeners.py#L14

data_filtered = universe_screener(
    data,
    date_max=TODAY,
    lookback_days=2 * 365,
    volume_top_n=1000,
    momentum_top_n=None,
    percent_change_filter=False,
    max_percent_change=0.35,
    min_avg_volume=100_000,
    min_avg_price=4.0,
    min_last_price=5.0,
    symbol_column="symbol",
    date_column="date",
    close_column="close",
    volume_column="volume",
)

data_filtered

data_filtered["symbol"].unique()


# * 2.0 DATA CLEANING PIPELINE

#   - Preprocesses OHLCV data by standardizing columns, fixing inconsistencies, handling missing values, replacing zeros/negatives, removing duplicates, and adding flag columns to identify potential issues for feature engineering.

# Function Definition: https://github.com/quant-science/QSResearch/blob/d3467148a6524bf342b6dcb707a0f66b7a4e0b8d/qsresearch/preprocessors/preprocess_price_data.py#L16

data_preprocessed = preprocess_price_data(
    data_filtered,
    symbol_column="symbol",
    date_column="date",
    close_column="close",
    high_column="high",
    low_column="low",
    volume_column="volume",
    min_trading_days=252 * 2,
)

data_preprocessed

data_preprocessed.glimpse()


# * 3.0 ADDING FEATURES

#  - Custom Function: Adds the QS Momentum Factor to the preprocessed data, which is a measure of momentum based on the rate of change of prices over different periods.

def add_qsmom_features(
    data,
    roc_fast_period=21,
    roc_slow_period=252,
    returns_period=126,
    date_column="date",
    close_column="close",
    symbol_column="symbol",
    engine="polars",
):
    """
    Implements the QS Momentum Factor:

    (
        (prices[-21] - prices[-252]) / prices[-252] - (prices[-1] - prices[-21]) / prices[-21]
    ) / np.nanstd(returns, axis=0)
    """
    ret = data.copy()
    try:
        ret = ret.groupby(symbol_column).augment_qsmomentum(
            date_column=date_column,
            close_column=close_column,
            roc_fast_period=roc_fast_period,
            roc_slow_period=roc_slow_period,
            returns_period=returns_period,
            engine=engine,
        )
        logging.info("Added QS Momentum")
    except Exception as e:
        logging.error(f"QS Momentum failed: {e}")
        raise Exception(f"QS Momentum failed: {e}")
    return ret

data_features = add_qsmom_features(
    data_preprocessed,
    roc_fast_period=21,
    roc_slow_period=252,
    returns_period=126,
    date_column="date",
    close_column="close",
    symbol_column="symbol",
    engine="polars",
)

# Save the checkpoint file:
# data_features.to_parquet(PATH_DIR / "data" / "data_features.parquet")

data_features = pd.read_parquet(PATH_DIR / "data" / "data_features.parquet")

data_features.glimpse()

# * 4.0 REVIEWING THE DATA

# Get the latest date in the dataset

latest_date = data_features["date"].max()

data_max_date = data_features[data_features["date"] == latest_date] 

data_max_date.sort_values(by="close_qsmom_21_252_126", ascending=False)

# Review the top 10 symbols by QS Momentum

top_10_symbols = data_max_date.nlargest(10, "close_qsmom_21_252_126")['symbol'].tolist()

top_10_symbols

data_features[data_features["symbol"].isin(top_10_symbols)] \
    .filter_by_time("date", "2024-07-01", "2025-07-01") \
    .groupby("symbol") \
    .plot_timeseries(
        date_column="date",
        value_column="close",
        title="Top 10 Symbols by QS Momentum",
        smooth=True,
        facet_ncol=3,
    )
    
# CONCLUSIONS
# - We built a financial database using the QSConnect library.
# - We successfully implemented a universe selection and cleaning pipeline using the QSResearch library.
# - We added the QS Momentum Factor to the preprocessed data on an increased universe of stocks.

# NEXT STEPS
# - In the next clinic, we will use this data to build a Quant Lab with the QSResearch library.
# - We will implement a backtesting framework to test our trading strategies using the QSResearch library.
# - We will also explore the QSResearch library's capabilities for feature engineering and fundamental analysis.
# - We will use the QSResearch library to build a trading strategy based on the QS Momentum Factor.
# - And we will capture all of the results in MLFlow. 

# TRADING COMPETITION
# - Your task is to submit trades based on the QS Momentum Factor (or another custom strategy) using a larger universe of stocks via the QSConnect library.

