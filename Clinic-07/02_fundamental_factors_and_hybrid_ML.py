# QUANT SCIENCE LLC 
# THE QUANT SCIENTIST PRO ALGORITHMIC TRADER 
# LEVEL 2 PROGRAM
# ****
# CLINIC #7 PART 2: FUNDAMENTAL STRATEGIES FOR MISPRICED MOMENTUM STOCKS

# * Goal:
#   - Develop strategies to identify mispriced momentum stocks using fundamental data.
#   - Focus on three approaches:
#     1. Rule of 40 Score: Balances revenue growth and profitability for growth stocks.
#     2. Free Cash Flow (FCF) Growth + Momentum: Targets cash-generative stocks with price momentum.
#     3. Peter Lynch PEG + Quality: Identifies undervalued growth stocks with high quality and momentum.
#   - Combine these with price-based features for a hybrid machine learning strategy.

# * FINANCIAL DATA REQUIREMENTS:
# - We use the Financial Modeling Prep (FMP) API to access financial data. 
# - We have a partnership with FMP for 30% OFF. 
#   DISCOUNT LINK: https://site.financialmodelingprep.com/developer/docs/pricing?couponCode=quantscience

# * PREREQUISITES:
# - CLINIC #5: Set up database with historical fundamentals from FMP.
# - CLINIC #6: Set up Zipline bundle with historical price data named "qspro_historical_prices_fmp".
# - CLINIC #7 PART 1: Familiarity with QSResearch and ML workflows (e.g., XGBoost).

# * INSTALLATION INSTRUCTIONS ----
# - Ensure the `qsresearch` package is installed:
#   ```
#   pip install git+https://github.com/quant-science/QSResearch.git
#   ```
# - Install additional dependencies:
#   ```
#   pip install xgboost ffn pyfolio alphalens
#   ```

# * LIBRARIES ----

import logging
import os
from pathlib import Path

import pandas as pd
import pytimetk as tk
import numpy as np
import polars as pl

from qsconnect import Client
from zipline.api import date_rules, time_rules

from qsresearch.utils.zipline import get_zipline_history
from qsresearch.preprocessors import preprocess_price_data, universe_screener
from qsresearch.features import add_forward_returns
from qsresearch.strategies.factor import run_backtest
from qsresearch.strategies.factor.algorithms import train_and_predict_xgb_regressor
from qsresearch.strategies.factor.portfolio_construction import long_short_equal_weight_portfolio
from qsresearch.portfolio_analysis.factor import create_full_alpha_factor_tearsheet_from_zipline

# Used to import Clinic 07 custom functions
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))

# Clinic 07 Custom Functions
from clinic07.custom_functions import (
    custom_indicator_function,
    data_printer,
    add_fundamentals_from_database,
    add_fcf_growth_from_database,
    add_eps_growth_from_database,
    add_rule_of_40_score,
    add_fcf_growth_momentum,
    add_lynch_multibagger_score,
    rule_of_40_screener,    
)

# Constants
BUNDLE_NAME = "qspro_historical_prices_fmp"
TODAY = "2025-07-01"
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
FORWARD_PERIOD = 21
TARGET_COLUMN = f"Return_fwd_{FORWARD_PERIOD}"

HYBRID_PREDICTOR_COLS = [
    
    # Technical Features
    "close_macd_histogram_50_200_30",
    "close_qsmom_21_252_126",
    "close_roc_0_63",
    "close_roc_0_252",
    "close_rsi_63",
    "close_fip_momentum_252",
    "close_volatility_annualized_252",

    # NEW Fundamental Features
    "returnOnEquity",
    "debtEquityRatio",
    "revenueGrowth",
    "netProfitMargin",
    "rule_of_40_momentum",
    "fcf_growth_momentum_score",
    "lynch_multibagger_score"
]

# * DATABASE CONNECTION ----

# Database Connection
os.environ["QSCONNECT_ROOT"] = str(Path.cwd() / "data" / "qsconnect")
client = Client()
conn = client.connect_to_database()

# Reminder - We can get a glimpse of all database tables which can be useful in using AI to develop fundamental factors.
client.glimpse_all_database_tables()


# * 1.0 WORKFLOW ----

# * 1. Extract Zipline Data
df_prices = get_zipline_history(
    bundle_name=BUNDLE_NAME,
    symbols=TEST_SYMBOLS,
    end_date=TODAY,
    bar_count=252*5,  # 5 years of daily data
    frequency="1d"
)

df_prices

# * 2. Preprocess Price Data
df_preprocessed = preprocess_price_data(df_prices)
df_preprocessed.glimpse()

# * 3. Add Technical Indicators
df_technicals = custom_indicator_function(df_preprocessed)
df_technicals.glimpse()

# * 4. Add Fundamental Data

# * 4.1 Ratio Data

fundamental_cols = [
    "returnOnEquity",
    "debtEquityRatio",
    "netProfitMargin",
    "freeCashFlowPerShare"
]

df_with_fundamentals = add_fundamentals_from_database(
    df_technicals,
    conn=conn,
    table="bulk_ratios_annual_fmp",
    columns=fundamental_cols,
    left_on="date",
    right_on="date"
)

df_with_fundamentals.glimpse()

# * 4.2 Key Metrics Data

df_with_fundamentals = add_fundamentals_from_database(
    df_with_fundamentals,
    conn=conn,
    table="bulk_key_metrics_annual_fmp",
    columns=["peRatio"],
    left_on="date",
    right_on="date"
)

df_with_fundamentals.glimpse()

# * 4.3 Cash Flow Growth Data

df_with_fundamentals = add_fcf_growth_from_database(
    df_with_fundamentals,
    conn=conn,
    table="bulk_cash_flow_statement_growth_annual_fmp",
    columns=["growthFreeCashFlow"]
)

df_with_fundamentals.glimpse()

# * 4.4 Financial Growth Data

df_with_fundamentals = add_eps_growth_from_database(
    df_with_fundamentals,
    conn=conn,
    table="bulk_financial_growth_annual_fmp",
    columns=["EPSGrowth", "revenueGrowth"]
)

df_with_fundamentals.glimpse()

# * 5. Add Fundamental Factors

# * 5.1 Rule of 40 Score
# - Combines revenue growth and profitability for growth stocks.

df_with_factors = add_rule_of_40_score(df_with_fundamentals)

df_with_factors.glimpse()

df_with_factors.tail().glimpse()

# * 5.2 Free Cash Flow (FCF) Growth + Momentum
# - Targets cash-generative stocks with price momentum.

df_with_factors = add_fcf_growth_momentum(df_with_factors)

df_with_factors.glimpse()

df_with_factors.tail().glimpse()

# * 5.3 Peter Lynch Multibagger Score
# - Identifies undervalued growth stocks with high quality and momentum.

df_with_factors = add_lynch_multibagger_score(df_with_factors)

df_with_factors.tail().glimpse()

# * 6. Screen Universe with Rule of 40
#   - Optional: You can screen the universe based on the Rule of 40 score.
#   - This step filters stocks that meet a minimum Rule of 40 score threshold.

df_screened = rule_of_40_screener(df_with_factors, min_score=0.40)

df_screened.glimpse()

df_screened.tail().glimpse()

# * 7. Add Forward Returns

df_hybrid = add_forward_returns(
    df_screened,
    symbol_column="symbol",
    date_column="date",
    close_column="close",
    forward_periods=FORWARD_PERIOD,
    engine="polars"
)

df_hybrid.glimpse()

# * 8. Train and Predict with XGBoost

train_data = df_hybrid[df_hybrid["date"] < TODAY].dropna()
predict_data = df_hybrid[df_hybrid["date"] == TODAY]

factor_signal = train_and_predict_xgb_regressor(
    train_data=train_data,
    predict_data=predict_data,
    predictor_cols=HYBRID_PREDICTOR_COLS,
    target_col=TARGET_COLUMN,
    model_params={
        "random_state": 123,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    },
)

predict_data["factor_signal"] = factor_signal
predict_data

# * 9. Portfolio Construction
weights = long_short_equal_weight_portfolio(
    predictions=predict_data["factor_signal"],
    num_long_positions=2,
    long_threshold=0.01,
    num_short_positions=0,
    short_threshold=-0.01,
)

weights

# * 10. Time Series Validation via Zipline Backtest

CONFIG = {
    # MLFlow Tracking
    "use_mlflow": True,
    "mlflow_experiment_name": "Momentum ML Strategies",
    "mlflow_run_name": "XGB Rule of 40 + FCF + Lynch",
    "mlflow_tags": {"strategy": "mispriced_momentum", "portfolio": "equal_weight"},
    
    # BACKTEST PARAMETERS:
    "bundle_name": BUNDLE_NAME,
    "start_date": pd.Timestamp("2024-01-01"),
    "end_date": pd.Timestamp(TODAY),
    "capital_base": 1_000_000,
    "benchmark_symbol": "SPY",
    "window_length": 252 * 5,
    "frequency": "1d",
    "target_col": TARGET_COLUMN,
    "predictor_cols": HYBRID_PREDICTOR_COLS,
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
                "volume_top_n": 1000,
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
            "name": "fundamentals_ratios",
            "func": add_fundamentals_from_database,
            "params": {
                # "conn": conn,
                "table": "bulk_ratios_annual_fmp",
                "columns": fundamental_cols,
                "left_on": "date",
                "right_on": "date",
            },
        },
        {
            "name": "fundamentals_key_metrics",
            "func": add_fundamentals_from_database,
            "params": {
                # "conn": conn,
                "table": "bulk_key_metrics_annual_fmp",
                "columns": ["peRatio"],
                "left_on": "date",
                "right_on": "date",
            },
        },
        {
            "name": "fcf_growth",
            "func": add_fcf_growth_from_database,
            "params": {
                # "conn": conn,
                "table": "bulk_cash_flow_statement_growth_annual_fmp",
                "columns": ["growthFreeCashFlow"],
            },
        },
        {
            "name": "eps_growth",
            "func": add_eps_growth_from_database,
            "params": {
                # "conn": conn,
                "table": "bulk_financial_growth_annual_fmp",
                "columns": ["EPSGrowth", "revenueGrowth"],
            },
        },
        {
            "name": "rule_of_40",
            "func": add_rule_of_40_score,
            "params": {
                "symbol_column": "symbol",
                "date_column": "date",
                "growth_col": "revenueGrowth",
                "margin_col": "netProfitMargin",
                "momentum_col": "close_roc_0_252",
            },
        },
        {
            "name": "rule_of_40_screener",
            "func": rule_of_40_screener,
            "params": {
                "min_score": 0.40,
                "symbol_column": "symbol",
                "date_column": "date",
            },
        },
        {
            "name": "fcf_momentum",
            "func": add_fcf_growth_momentum,
            "params": {
                "symbol_column": "symbol",
                "date_column": "date",
                "fcf_growth_col": "growthFreeCashFlow",
                "price_col": "close",
                "fcf_per_share_col": "freeCashFlowPerShare",
                "momentum_col": "close_roc_0_252",
            },
        },
        {
            "name": "lynch_multibagger",
            "func": add_lynch_multibagger_score,
            "params": {
                "symbol_column": "symbol",
                "date_column": "date",
                "pe_col": "peRatio",
                "eps_growth_col": "EPSGrowth",
                "roe_col": "returnOnEquity",
                "debt_col": "debtEquityRatio",
                "momentum_col": "close_roc_0_252",
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
        {
            "name": "data_printer",
            "func": data_printer,  # Optional function to print data post processing
            "params": {},
        },
    ],
    
    # Algorithm
    "algorithm": {
        "func": train_and_predict_xgb_regressor,
        "params": {
            "predictor_cols": HYBRID_PREDICTOR_COLS,
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
            "num_long_positions": 20,
            "long_threshold": 0.01,
            "num_short_positions": 0,
            "short_threshold": -0.01,
        },
    },
}

performance_df = run_backtest(CONFIG)
performance_df.glimpse()

# * GETTING THE PERFORMANCE RESULTS ----

# Change the path to your artifact performance.pkl file
PATH_TO_PERFORMANCE = "file:///Users/mdancho/Desktop/course_code/QS02-Quant_Scientist_Algo_Trading_System/mlruns/864121016415026816/d5345be14e954ac4a07a32e4c9b9d94e/artifacts/performance.pkl"

performance_df = pd.read_pickle(PATH_TO_PERFORMANCE)

create_full_alpha_factor_tearsheet_from_zipline(
    zipline_results=performance_df,
    periods=(5, 10, 21, 30, 42, 63),
    quantiles=12,
)

# * MLFLOW TRACKING ----
# Run in terminal to start MLflow server:
#   mlflow server

# * NEXT STEPS:
# - Knowledge Check: Test alternative thresholds for Rule of 40 (e.g., min_score=0.50) or add criteria like low PEG (<1).

# - Compare performance vs. pure momentum or Buffett score strategies in MLflow.

# - Explore other fundamental tables (e.g., bulk_key_metrics_annual_fmp) for additional multibagger signals like ROIC or operating cash flow growth.

