# QUANT SCIENCE LLC 
# THE QUANT SCIENTIST PRO ALGORITHMIC TRADER 
# LEVEL 2 PROGRAM
# ****
# CLINIC #6: MOMENTUM EDGE DISCOVERY WITH QSRESEARCH

# * Goal:
#   - Implement a momentum factor pipeline using the QSResearch library.
#   - This will help us discover and analyze momentum edges in financial data.
#   - The pipeline will include universe selection, data preprocessing, and backtesting.
#   - MLflow tracking will be integrated for experiment management.


# * LIBRARIES ----

# Data Analysis
import logging
import pandas as pd
import pytimetk as tk

# Zipline Helpers
from zipline.api import date_rules, time_rules

# QSResearch - Preprocessing
from qsresearch.preprocessors import preprocess_price_data, universe_screener

# QSResearch - Factor Strategies
from qsresearch.strategies.factor import run_backtest
from qsresearch.strategies.factor.algorithms import use_factor_as_signal
from qsresearch.strategies.factor.portfolio_construction import long_short_equal_weight_portfolio

# QS Research - Performance Analysis
from qsresearch.portfolio_analysis.returns import create_full_returns_tearsheet_from_zipline

from qsresearch.portfolio_analysis.factor import create_full_alpha_factor_tearsheet_from_zipline

# * ENVIRONMENT VARIABLES ----

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants

PREDICTOR_COLS = [
    "close_qsmom_21_252_126",
]

# * CUSTOM FUNCTIONS ----

# Data Printer Function
def data_printer(data, label="Processed Data"):
    print(f"\n{label}:\n")
    tk.glimpse(data.tail())
    return data

# Add QS Momentum Feature 
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


# * 1.0 TRADING STRATEGY CONFIGURATION ----

# Run Backtest Using Configurations:
# https://github.com/quant-science/QSResearch/blob/master/qsresearch/strategies/factor/run_backtest.py

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
    "mlflow_experiment_name": "Momentum Factor Strategy",
    "mlflow_run_name": "qsmom equal weight: long only, volume_top_n=500, no stop loss, 1.5 year",
    "mlflow_tags": {"strategy": "simple_momentum", "portfolio": "equal_weight"},
    
    # BACKTEST PARAMETERS:
    "start_date": pd.Timestamp("2024-01-01"),
    "end_date": pd.Timestamp("2025-07-01"),
    "capital_base": 1_000_000,
    "bundle_name": "qspro_historical_prices_fmp",
    "benchmark_symbol": "SPY",  # Set to None to skip benchmark
    "window_length": 252 * 3,  # zipline bar count window for training and prediction
    "frequency": "1d",
    "predictor_cols": PREDICTOR_COLS,
    "factor_signal_sort_descending": True,  # Sort factor signals in descending order
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
            "name": "data_printer_raw",
            "func": data_printer,
            "params": {"label": "Raw Data"},            
        },
        {
            "name": "screener",
            "func": universe_screener,  # Assuming this is from your module
            "params": {
                "lookback_days": 2 * 365,
                "volume_top_n": 500,
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
            "name": "momentum_factor",
            "func": add_qsmom_features,
            "params": {
                "roc_fast_period": 21,
                "roc_slow_period": 252,
                "returns_period": 126,
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
                "engine": "polars",
            },
        },
        {
            "name": "data_printer_final",
            "func": data_printer,  # Optional function to print data post processing
            "params": {"label": "Processed Data with QS Momentum"},
        },
    ],
    # Algorithms
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
            "num_long_positions": 20,
            "long_threshold": 1.00,
            # 'num_short_positions': 20,
            # 'short_threshold': -1.00,
        },
    },
}

results = run_backtest(CONFIG)

# * 2.0 MLFLOW TRACKING ----

# 1. Open Terminal and run:
#    mflow server
# 2. Open Browser and go to:
#    http://localhost:5000

# * 3.0 HOW I USE MLFLOW FOR ORGANIZING TRADING STRATEGIES:

# 1. MLFlow Directory Folder: 
#   MLFlow stores all its data in a folder named `mlruns` in the current working directory. This folder contains subfolders for each experiment, and within each experiment, there are subfolders for each run.

# 2. MLFlow UI: 
#   The MLFlow UI provides a web interface to visualize and compare runs, view metrics, parameters, and artifacts. You can access it by running `mlflow ui` in the terminal and navigating to `http://localhost:5000` in your browser.

# 3. Experiments vs Runs: 
#   An experiment is a logical grouping of runs, while a run is a single execution of your code. You can think of an experiment as a folder that contains multiple runs.

# 4. Searching Runs:
#    - Search Fields: params.start_date = '2024-01-01 00:00:00' AND metrics.portfolio_daily_sharpe >= 0.4
#    - Charts: You can create interactive charts to visualize metrics across runs. By clicking on datapoints, you can filter runs to find the best performing trading strategies.
#    - Metrics & Params: You can compare runs based on metrics like Sharpe Ratio, Total Return, etc. This helps in identifying the best performing strategies in Table format.
#    - Groups: You can group runs based on parameters or tags to analyze performance across different configurations.
#    - Hide and Unhide Strategies: You can hide or unhide specific runs to focus on the most relevant strategies.


# * 4.0 HOW I ACCESS TRADING STRATEGY RESULTS:

# 1. Run Overview:
#    - Metadata: Each run has metadata such as start date, end date, and duration.
#    - Metrics: Key metrics like total return, Sharpe ratio, and max drawdown
#    - Parameters: Parameters used for the run FROM CONFIG, such as preprocessor parameters, start and end dates, and portfolio construction parameters.

# 2. Artifacts:
#    - Artifacts are files generated during a run
#    - We collect:
#      - Performance reports: alphalens, pyfolio
#      - Backtest results: zipline performance.pkl file
#      - Config: A config.pkl file containing the configuration used for the run
#      - portfolio_stats.pkl: A CSV file containing portfolio statistics

# * 5.0 GETTING THE STRATEGY ZIPLINE PERFORMACE DATA:

# * GET THIS PATH FROM THE MLFlow UI > Artifacts > performance.pkl:
PATH_TO_PERFORMANCE = "file:///Users/mdancho/Desktop/course_code/QS02-Quant_Scientist_Algo_Trading_System/mlruns/331281106765637158/f5c0461cd4244801a5554ea4af38f721/artifacts/performance.pkl"


# Load the performance data (this is the *exact* same zipline output from Clinic 2)
performance_df = pd.read_pickle(PATH_TO_PERFORMANCE)
performance_df

performance_df.glimpse()

# * Returns Analysis ----

create_full_returns_tearsheet_from_zipline(
    zipline_results=performance_df,
    # engine='pyfolio',
)

# * Alpha Factor Analysis ----

create_full_alpha_factor_tearsheet_from_zipline(
    zipline_results=performance_df,
    periods=(5, 10, 21, 30, 42, 63),
    quantiles=20,
)

# * 6.0 PLACE TRADES WITH OMEGA APP ----

# Omega 
import omega
from omega import start_loop
from omega import MarketOrder, Stock
from omega.utils.zipline_utils import omega_trades_from_zipline

# Calling `start_loop` is only required when using Omega in a Jupyter Notebook.
start_loop()

# Instantiate the Omega app
app = omega.Omega()

print(app.is_connected())

# Current positions in IBKR
positions = app.positions_as_symbols()
positions

# Backtest positions from the performance DataFrame
bt_positions = [d["sid"].symbol for d in performance_df.positions.iloc[-1]]
bt_positions

# Calculate positions to liquidate
divest = list(set(positions) - set(bt_positions))
divest

# Divest from positions not in the backtest
for sym in divest:
    contract = Stock(sym, "SMART", "USD")
    
    try:
        logging.info(f"Divesting {sym}...")
        app.order_target_percent(
            contract=contract, order_type=MarketOrder, target=0.0
        )
    except Exception as e:
        logging.error(f"Failed to divest {sym}: {e}")
        continue

# Invest in backtest positions
weights = [1.0 / len(bt_positions)] * len(bt_positions)  # Equal weight for each position

for i, sym in enumerate(bt_positions):
    contract = Stock(sym, "SMART", "USD")
    app.order_target_percent(
        contract=contract, order_type=MarketOrder, target=weights[i]
    )

# Disconnect from the Omega app
app.disconnect()

# * 7.0 NEXT STEPS ----

# 1. We validated the QS Momentum factor for 1.5 years at 21 day rebalance frequency.
# 2. Increase the backtest period to **20 years** to validate the factor over a longer time horizon.
# 3. Then start trading in paper trading it if the factor is profitable.


# * COMING IN CLINIC #7:

# 1. We will implement a **multi-factor model** using the QSResearch library WITH MACHINE LEARNING.
# 2. We will use the QSConnect library to fetch Fundamental data and build a Momentum + Fundamental Factor Strategy.
# 3. We will leverage FMP Fundamental data to enhance our trading strategies.
