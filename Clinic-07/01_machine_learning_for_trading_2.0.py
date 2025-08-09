# QUANT SCIENCE LLC 
# THE QUANT SCIENTIST PRO ALGORITHMIC TRADER 
# LEVEL 2 PROGRAM
# ****
# CLINIC #7: MACHINE LEARNING FOR TRADING 2.0

# * Goal:
#   - Learn how to implement a machine learning-based trading strategy using the QSResearch library.
#   - Focus on integrating XGBoost for predictive modeling and advanced portfolio construction.

# * Prerequisites:
#   - Clinic #6: Familiarity with QSResearch library and its components for quantitative research.
#   - Clinic #5: QSConnect environment setup and Zipline bundle creation (`qspro_historical_prices_fmp`).
#   - Bonus Level 1 Machine Learning Clinic: Basic understanding of machine learning concepts and algorithmic trading workflows covered in the BONUS Level 1 Clinic.

# * INSTALLATION INSTRUCTIONS ----
# - Ensure the `qsresearch` package is installed:
#   ```
#   pip install git+https://github.com/quant-science/QSResearch.git
#   ```
# - Install additional dependencies for machine learning:
#   ```
#   pip install xgboost ffn pyfolio alphalens
#   ```

# * LIBRARIES ----

# Data Analysis
import pandas as pd
import numpy as np
import pytimetk as tk

from zipline.api import date_rules, time_rules

# QSResearch
from qsresearch.utils.zipline import get_zipline_history
from qsresearch.preprocessors import universe_screener, preprocess_price_data
from qsresearch.features import add_forward_returns
from qsresearch.strategies.factor.algorithms import train_and_predict_xgb_regressor, train_and_predict_h2o_automl
from qsresearch.strategies.factor.portfolio_construction import long_short_equal_weight_portfolio
from qsresearch.strategies.factor import run_backtest
from qsresearch.portfolio_analysis.factor import create_full_alpha_factor_tearsheet_from_zipline

# Used to import Clinic 07 custom functions
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))

# Clinic 07 Custom Functions
from clinic07.custom_functions import custom_indicator_function, data_printer

# Constants
BUNDLE_NAME = "qspro_historical_prices_fmp"  
TODAY = "2025-07-01"
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
FORWARD_PERIOD = 21
TARGET_COLUMN = f"Return_fwd_{FORWARD_PERIOD}"

PREDICTOR_COLS = [
    "close_macd_histogram_50_200_30",
    "close_qsmom_21_252_126",
    "close_roc_0_63",
    "close_roc_0_252",
    "close_rsi_63",
    "close_fip_momentum_252",
    "close_volatility_annualized_252",
]

# * 1.0 RECAP:

# 1. You have set up the QSConnect environment and created a Zipline bundle (`qspro_historical_prices_fmp`) in Clinic #5.
# 2. You learned how to use QSResearch for factor-based trading strategies in Clinic #6.
# 3. This clinic extends the workflow to incorporate machine learning models, specifically XGBoost, for predicting forward returns.

# * 2.0 MACHINE LEARNING FOR TRADING ----

# * How it works:
# - This clinic introduces a machine learning-based trading strategy using XGBoost to predict forward returns.
# - The workflow includes data preprocessing, feature engineering with technical indicators, training an XGBoost model, and backtesting the strategy.


# * Workflow:

# 1. Extract Zipline Data:
#    - Retrieve historical price data from the Zipline bundle for a universe of stocks.

df = get_zipline_history(
    bundle_name=BUNDLE_NAME,
    symbols=TEST_SYMBOLS,
    end_date=TODAY,
    bar_count=252*5,  # 5 years of daily data
    frequency="1d"
)

df.glimpse()

df.tail()

# 2. Universe Screening:
#    - Filter the stock universe based on volume and other criteria.

df_screened = universe_screener(df, volume_top_n=3)

df_screened

df_screened.glimpse()

# 3. Preprocess Price Data:
#    - Clean and prepare the price data for analysis.

df_preprocessed = preprocess_price_data(df_screened)

df_preprocessed.glimpse()

# 4. Feature Engineering:
#    - Define a custom function to add technical indicators for machine learning.
#    - This function includes MACD, QS Momentum, ROC, RSI, FIP Momentum, and Volatility from the `pytimetk` library.
#    - Reference: https://business-science.github.io/pytimetk/reference/#finance-module-momentum-indicators

# MATT'S METHOD FOR ADDING TECHNICAL INDICATORS OR GOOD FEATURES:
# - Use previous factors based on your backtests
# - Add features that other people have used successfully via:
#.  - Papers
#.  - Books
#.  - etc

df_engineered = custom_indicator_function(
    df_preprocessed,
    symbol_column="symbol",
    date_column="date",
    close_column="close",
    engine="polars",
)

df_engineered.glimpse()

# 5. Add Forward Returns:
#    - Compute forward returns for the target variable.

df_with_returns = add_forward_returns(
    df_engineered,
    symbol_column="symbol",
    date_column="date",
    close_column="close",
    forward_periods=FORWARD_PERIOD,
    engine="polars",
)

df_with_returns.glimpse()

# 6. Train and Predict with XGBoost:
#    - Train an XGBoost model to predict forward returns.
#    - Reference: https://github.com/quant-science/QSResearch/blob/master/qsresearch/strategies/factor/algorithms.py

train_data = df_with_returns[df_with_returns["date"] < TODAY].dropna()
predict_data = df_with_returns[df_with_returns["date"] == TODAY]

factor_signal = train_and_predict_xgb_regressor(
    train_data=train_data,
    predict_data=predict_data,
    predictor_cols=PREDICTOR_COLS,
    target_col=TARGET_COLUMN,
    model_params={
        "random_state": 123,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    },
)

predict_data["factor_signal"] = factor_signal

predict_data

# 7. Portfolio Construction:
#    - Construct a portfolio based on the predicted signals.

weights = long_short_equal_weight_portfolio(
    predictions=predict_data["factor_signal"],
    num_long_positions=2,
    long_threshold=0.01,
    num_short_positions=1,
    short_threshold=-0.01,
)

weights

# 8. Backtesting:
#    - Run a backtest using the configured ML strategy.

CONFIG = {
    # MLFlow Tracking
    "use_mlflow": True,
    "mlflow_experiment_name": "Test Strategies",
    "mlflow_run_name": "Cohort 10 - Test Strategy XGB Regressor Clinic 7",
    "mlflow_tags": {"strategy": "test", "portfolio": "equal_weight"},
    
    # BACKTEST PARAMETERS:
    "bundle_name": BUNDLE_NAME,
    "start_date": pd.Timestamp("2025-05-01"),
    "end_date": pd.Timestamp(TODAY),
    "capital_base": 1_000_000,
    "benchmark_symbol": "SPY",
    "window_length": 252 * 5,
    "frequency": "1d",
    "target_col": TARGET_COLUMN,
    "predictor_cols": PREDICTOR_COLS,
    "factor_signal_sort_descending": True,
    "calendar_name": "NYSE",
    "extra_init": None,
    "custom_handle_data": None,
    
    # BACKTEST FUNCTIONS
    "rebalance_schedule": {
        "date_rule": date_rules.month_start(),
        "time_rule": time_rules.market_open(minutes=60),
    },
    "transaction_costs": {
        "slippage": {"spread": 0.01},
        "commission": {"cost": 0.005, "min_trade_cost": 0},
    },
    
    # Preprocessing steps
    "preprocess": [
        {
            "name": "screener",
            "func": universe_screener,
            "params": {
                "lookback_days": 2 * 365,
                "volume_top_n": 10,
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
            "func": custom_indicator_function,
            "params": {
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
                "engine": "polars",
            },
        },
        {
            "name": "forward_returns",
            "func": add_forward_returns,
            "params": {
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
                "forward_periods": FORWARD_PERIOD,
                "engine": "polars",
            },
        },
    ],
    
    # Algorithm
    "algorithm": {
        "func": train_and_predict_xgb_regressor,
        "params": {
            "predictor_cols": PREDICTOR_COLS,
            "target_col": TARGET_COLUMN,
            "model_params": {
                "random_state": 123,
                "objective": "reg:squarederror",
                "n_jobs": -1,
            },
        },
    },
    
    # Portfolio Construction
    "portfolio_strategy": {
        "func": long_short_equal_weight_portfolio,
        "params": {
            "num_long_positions": 5,
            "long_threshold": 0.01,
            "num_short_positions": 0,
            "short_threshold": -0.01,
        },
    },
}

performance_df = run_backtest(CONFIG)

performance_df.glimpse()


# * MLFLOW TRACKING ----

# Run in terminal to start MLflow server:
#   mlflow server

# * 3.0 COMPLETE MACHINE LEARNING FACTOR TRADING STRATEGY CONFIGURATION ----

# Complete configuration for a machine learning trading strategy using XGBoost
CONFIG = {
    # MLFlow Tracking
    "use_mlflow": True,
    "mlflow_experiment_name": "Momentum ML Strategies",
    "mlflow_run_name": "XGB Regressor equal weight: reduce features, long-only, momentum_top_n=None, volume_top_n=500, 1.5 year",
    "mlflow_tags": {"strategy": "xgb-regressor", "portfolio": "equal_weight"},
    
    # BACKTEST PARAMETERS:
    "bundle_name": BUNDLE_NAME,
    "start_date": pd.Timestamp("2024-01-01"),
    "end_date": pd.Timestamp(TODAY),
    "capital_base": 1_000_000,
    "benchmark_symbol": "SPY",  
    "window_length": 252 * 5,  
    "frequency": "1d",
    "target_col": TARGET_COLUMN,  
    "predictor_cols": PREDICTOR_COLS,
    "factor_signal_sort_descending": True,
    "calendar_name": "NYSE",  
    "extra_init": None,  
    "custom_handle_data": None,  
    
    # BACKTEST FUNCTIONS
    "rebalance_schedule": {
        "date_rule": date_rules.month_start(),
        "time_rule": time_rules.market_open(minutes=60),
    },
    "transaction_costs": {
        "slippage": {"spread": 0.01},
        "commission": {"cost": 0.005, "min_trade_cost": 0},
    },
    
    # Preprocessing steps applied sequentially
    "preprocess": [
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
            "name": "technical_indicators",
            "func": custom_indicator_function,
            "params": {
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
                "engine": "polars",
            },
        },
        # ADD MORE CUSTOM INDICATOR FUNCTIONS HERE IF DESIRED
        {
            "name": "forward_returns",
            "func": add_forward_returns,  # Custom function to add forward returns
            "params": {
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
                "forward_periods": FORWARD_PERIOD,
                "engine": "polars",
            },
        },
        {
            "name": "data_printer",
            "func": data_printer,  # Optional function to print data post processing
            "params": {},
        },
    ],
    # Algorithms
    "algorithm": {
        "func": train_and_predict_xgb_regressor,
        "params": {
            "predictor_cols": PREDICTOR_COLS,
            "target_col": TARGET_COLUMN,
            "model_params": {
                "random_state": 123,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                # "learning_rate": 0.1,
            },
        },
    },
    # Portfolio Construction:
    "portfolio_strategy": {
        "func": long_short_equal_weight_portfolio,
        "params": {
            "num_long_positions": 20,
            "long_threshold": 0.01,
            "num_short_positions": 0,
            "short_threshold": -0.01,
        },
    },
}

performance_df = run_backtest(CONFIG)

# * GETTING THE PERFORMANCE RESULTS ----

# Change the path to your artifact performance.pkl file
PATH_TO_PERFORMANCE = "file:///Users/mdancho/Desktop/course_code/QS02-Quant_Scientist_Algo_Trading_System/mlruns/864121016415026816/f04e7a07ac77433eb1b31d33372bfdfb/artifacts/performance.pkl"

performance_df = pd.read_pickle(PATH_TO_PERFORMANCE)

create_full_alpha_factor_tearsheet_from_zipline(
    zipline_results=performance_df,
    periods=(5, 10, 21, 30, 42, 63),
    quantiles=10,
)

# * ML STRATEGY IMPROVEMENT AREAS:

# 1. Universe Screening:

# - Limit the universe to stocks you would actually trade.
# - Use top_n_volume and top_n_momentum to focus on the most liquid and trending stocks.

# 2. Feature Selection:

# - Review the predictor columns in `PREDICTOR_COLS` to ensure they are relevant and not overly correlated.
# - Consider using feature importance from the XGBoost model to refine the feature set.

# 3. Hyperparameter Tuning:

#  - Option 1: Manual Tuning
#    - Adjust XGBoost parameters in the `model_params` dictionary within the `algorithm` section of the configuration.
#    - Experiment with parameters like `n_estimators`, `max_depth`, `learning_rate`, and `subsample` to see their impact on model performance.

#  - Option 2: Automated Tuning
#    - Integrate libraries like Optuna or Hyperopt for automated hyperparameter optimization.
#    - This would involve setting up an optimization loop that evaluates different parameter combinations based on backtest performance.

#  - Option 3: H2O AutoML
#    - Use H2O's AutoML capabilities to automatically train and tune multiple models,

# pip install h2o

import h2o

h2o.init(max_mem_size="4G")

factor_signal = train_and_predict_h2o_automl(
    train_data=train_data,
    predict_data=predict_data,
    predictor_cols=PREDICTOR_COLS,
    target_col=TARGET_COLUMN,
    model_params={
        "max_runtime_secs": 3600,
        "seed": 123,
        "exclude_algos": ["DeepLearning"],
        "nfolds": 5,
    },
)

# * NEXT STEPS:

# - Knowledge Check: Experiment with different predictor columns or model parameters in the XGBoost configuration.

# - Explore advanced machine learning like H2O AutoML for automated model training and hyperparameter tuning.





