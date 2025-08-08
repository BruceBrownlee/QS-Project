# %%
# QUANT SCIENCE LLC 
# THE QUANT SCIENTIST PRO ALGORITHMIC TRADER 
# LEVEL 2 PROGRAM
# ****
# CLINIC #6: INTRO TO QSRESEARCH

# QSResearch is a powerful library designed for quantitative research and strategy development.
# It provides tools for data preprocessing, feature engineering, backtesting, and performance analysis.

# * Goal:
#   - Learn how to use the QSResearch library to implement a quantitative research project.

# * Prerequisites:
#   - Clinic #5:
#     - You have already set up the QSConnect environment and connected to the database.
#     - You have created a Zipline bundle from the price data called `qspro_historical_prices_fmp`.

# * INSTALLATION INSTRUCTIONS ----
# - Install the `qsresearch` package from GitHub:
#   ```
#   pip install git+https://github.com/quant-science/QSResearch.git
#   ```

# %%
# * LIBRARIES ----

import pandas as pd
import pytimetk as tk
import logging

from zipline.api import date_rules, time_rules

TODAY = "2025-07-31"

# %%
# * 1.0 RECAP:

# 1. We have already set up the QSConnect environment and connected to the database in the previous clinic and we have created a Zipline bundle from the price data called `qspro_historical_prices_fmp` (Clinic #5).
# 2. You learned how to backtest a strategy with Zipline in Clinic #2. (Ref. 02_momentum_omega.ipynb)

# 3. The QSResearch library builds on top of Zipline for much faster algorithmic trading strategy research and development: https://github.com/quant-science/QSResearch/tree/master/qsresearch

# %%
# * 2.0 QSRESEARCH ----

# * How it works:
# - QSResearch provides a structured way to define and run quantitative research projects.
# - It includes modules for data preprocessing, feature engineering, backtesting, and performance analysis.
# - And, most importantly, it provides an easy way to go from experiment design to backtesting and performance analysis that follows a 100% reproducible data science workflow.

# %%
# * Example of a typical algo trading development workflow:
# 1. Extract Zipline Data from Zipline Bundle:
#    - The `get_zipline_history` function retrieves historical price data from the Zipline bundle.
#    - This data can be used for testing your functions.

from qsresearch.utils.zipline import get_zipline_history

# Make sure to adjust your bundle name. Symbols must be in the bundle.
# Here we are using the `qspro_historical_prices_fmp` bundle created in Clinic #5.

df = get_zipline_history(
    bundle_name="qspro_demo_historical_prices_fmp",
    symbols=["AAPL", "AMZN","PLTR"], # This is a small universe for testing
    end_date=TODAY,
    bar_count=252*3,  # 3 years of daily data
    frequency="1d"
)

df

# %%
# 2. Select Universe of Stocks:
#    - The `universe_screener` allows you to filter and select a universe

from qsresearch.preprocessors import universe_screener

df_screened = universe_screener(
    df,
    volume_top_n=3,
)

df_screened

# %%
# 3. Preprocessing Price Data:
#    - The `preprocess_price_data` function cleans and prepares the price data for analysis.

from qsresearch.preprocessors import preprocess_price_data

df_preprocessed = preprocess_price_data(
    df_screened,
)

df_preprocessed.glimpse()

# %%
# 4. Feature Engineering:
#    - The `add_technical_indicators` function adds features to the dataset. More feature engineering functions can be found in the `qsresearch.features` module.

from qsresearch.features import add_technical_indicators

df_engineered = add_technical_indicators(
    df_preprocessed,
    compute_rolling_risk=False,  # Takes long to run
    compute_qs_momentum=True,
    
)

df_engineered.glimpse()

# %%
# 5. Algorithms
#    - The `use_factor_as_signal` function allows you to use a specific factor (column) as a trading signal.

from qsresearch.strategies.factor.algorithms import use_factor_as_signal

train_data = df_engineered[df_engineered["date"] < TODAY]

predict_data = df_engineered[df_engineered["date"] == TODAY]

predict_data.glimpse()

factor_signal = use_factor_as_signal(
    train_data=train_data,
    predict_data=predict_data,
    factor_column="close_fastqsmom_21_252_126",
)

factor_signal

predict_data["factor_signal"] = factor_signal

predict_data.glimpse()

# %%
# 6. Portfolio Construction:
#    - The `long_short_equal_weight_portfolio` function constructs a portfolio based on the factor signal.

from qsresearch.strategies.factor.portfolio_construction import long_short_equal_weight_portfolio

weights = long_short_equal_weight_portfolio(
    predictions=predict_data["factor_signal"],
    num_long_positions=2,
    long_threshold=0,
)

weights

# %%
# 7. Backtesting:
#   - The `run_backtest` function allows you to backtest the portfolio using historical data running this algorithm in a loop using Zipline.
#   - We develop a configuration dictionary that defines the backtest parameters and the algorithm to run.
#   - The CONFIG integrates our workflow steps into a single backtest configuration.

from qsresearch.strategies.factor import run_backtest

PREDICTOR_COLS = ["close_fastqsmom_21_252_126"]

CONFIG = {
    
    # MLFlow Tracking
    # Must include:
    # start_date: Start date of the backtest (pd.Timestamp).
    # end_date: End date of the backtest (pd.Timestamp).
    # capital_base: Initial capital for the backtest (float).
    # bundle_name: Name of the Zipline data bundle (str).
    
    # Optional keys:
    # calendar_name: Trading calendar name (default: 'NYSE').
    # custom_handle_data: Custom handle_data function (default: default_handle_data).
    # mlflow_tracking_uri:  MLflow tracking server URI (e.g., 'http://mlflow-server:5000').
    #                       An empty string, or a local file path, prefixed with file:/ 
    #                       Data is stored locally at the provided file (or ./mlruns if empty)
    #                       Can be an HTTP URI like https://my-tracking-server:5000 (or :8301)
    # mlflow_tracking_port: MLflow tracking server port (default: None) - don't use both port and URI.
    # mlflow_artifact_root: Artifact storage location (e.g., 's3://my-bucket/mlflow/artifacts').
    # mlflow_nested_run: Whether to create a nested run (default: False).
    # mlflow_log_metrics_frequency: Frequency for logging metrics (e.g., 'daily', default: None).
    # mlflow_artifact_subdir: Subdirectory for artifacts (e.g., 'momentum_backtest'). """
    
    "use_mlflow": True,
    "mlflow_tracking_uri": "/Users/brucebrownlee/dev/github/Resident/QS-Project/Clinic-06/mlruns",  # Local file storage for testing
    "mlflow_experiment_name": "Test Strategies",
    "mlflow_run_name": "Test Strategy 5",
    "mlflow_tags": {
        "strategy": "test", "portfolio": "equal_weight"
    },

    # BACKTEST PARAMETERS:
    "bundle_name": "qspro_demo_historical_prices_fmp",
    "start_date": pd.Timestamp("2025-05-01"), # NOTE - I'm making this small for testing
    "end_date": pd.Timestamp(TODAY),
    "capital_base": 1_000_000,    
    "benchmark_symbol": "SPY",  # Set to None to skip benchmark
    "window_length": 252 * 3,  # zipline bar count window for training and prediction
    "frequency": "1d",
    "predictor_cols": PREDICTOR_COLS,
    "calendar_name": "NYSE",  # Default calendar, can be changed
    "extra_init": None,  # Optional custom initialization function
    "custom_handle_data": None,  # Optional custom handle_data function
    
    # BACKTEST FUNCTIONS
    "rebalance_schedule": {
        "date_rule": date_rules.month_start(),
        "time_rule": time_rules.market_open(minutes=60),
    },
    "transaction_costs": {
        "slippage": {"spread": 0.01},
        "commission": {"cost": 0.005, "min_trade_cost": 0},
    },
    # # Add stop-loss settings
    # 'stop_loss': {
    #     'long_threshold': 0.10,  # 10% stop-loss
    #     'short_threshold': 0.10,  # 10% stop-loss
    #     'date_rule': date_rules.every_day(),
    #     'time_rule': time_rules.market_open(minutes=60),
    # },
    
    # Preprocessing steps applied sequentially
    "preprocess": [
        {
            "name": "screener",
            "func": universe_screener,  # Assuming this is from your module
            "params": {
                "lookback_days": 2 * 365,
                "volume_top_n": 10, # NOTE - I'M MAKING THIS VERY SMALL FOR TESTING
                "momentum_top_n": None,
                "percent_change_filter": False,
                "max_percent_change": 0.35,
                "volatility_filter": True,
                "max_volatility": 0.25,
                "min_avg_volume": 100_000,
                "min_avg_price": 4.0,
                "min_last_price": 5.0,
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
                "volume_column": "volume",
            },
        },
        {
            "name": "price_preprocessor",
            "func": preprocess_price_data,
            "params": {
                "min_trading_days": 252 * 2,
                "remove_low_trading_days": True,
                "remove_large_gaps": True,
                "remove_low_volume": True,
                "symbol_column": "symbol",
                "date_column": "date",
                "open_column": "open",
                "high_column": "high",
                "low_column": "low",
                "close_column": "close",
                "volume_column": "volume",
                "engine": "polars",
            },
        },
        {
            "name": "technical_indicators",
            "func": add_technical_indicators,
            "params": {
                "date_column": "date",
                "symbol_column": "symbol",
                "close_column": "close",
                "high_column": "high",
                "low_column": "low",
                "volume_column": "volume",
                "compute_rolling_risk": False,
                "compute_qs_momentum": True,  # Add QS Momentum Factor
            },
        }
    ],
    
    # Algorithm
    "algorithm": {
        "func": use_factor_as_signal,
        "params": {
            "factor_column": PREDICTOR_COLS[0],
        },
    },
    
    # Portfolio Construction:
    "portfolio_strategy": {
        "func": long_short_equal_weight_portfolio,
        "params": {
            "num_long_positions": 5, # NOTE - I'M MAKING THIS SMALL FOR TESTING
            "long_threshold": 1.00,
            # 'num_short_positions': 20,
            # 'short_threshold': -1.00,
        },
    },
}

performance_df = run_backtest(CONFIG)

performance_df.glimpse()

# %%
# * MLFLOW TRACKING ----

# Run in terminal:
#   mlflow server
# Use: mlflow server --port 8031 --backend-store-uri ~/dev/github/Resident/QS-Project/Clinic-06/mlruns 

# * NEXT STEPS:

# - Knowledge Check: Try adding a new function that adds a custom feature to the dataset.

# - Now you know how it works, we'll examine the QS Momentum Factor strategy.
# %%
performance_df

# %%