# FEATURE ENGINEERING FUNCTIONS

import pandas as pd
import numpy as np
import pytimetk as tk
import polars as pl

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Custom data printer function for debugging
def data_printer(data):
    tk.glimpse(data.tail())
    return data

# Create features
def custom_indicator_function(
    data,
    symbol_column="symbol",
    date_column="date",
    close_column="close",
    engine="polars",
):
    ret = data.copy()

    # MACD
    for fast, slow, signal in [(50, 200, 30)]:
        try:
            ret = ret.groupby(symbol_column).augment_macd(
                date_column=date_column,
                close_column=close_column,
                fast_period=fast,
                slow_period=slow,
                signal_period=signal,
                engine=engine,
            )
            logger.info(f"Added MACD with periods {fast}_{slow}_{signal}")
        except Exception as e:
            logger.error(f"Failed to add MACD {fast}_{slow}_{signal}: {e}")

    # QS Momentum
    try:
        ret = ret.groupby(symbol_column).augment_qsmomentum(
            date_column=date_column,
            close_column=close_column,
            roc_fast_period=21,
            roc_slow_period=252,
            returns_period=126,
            engine=engine,
        )
        logger.info("Added QS Momentum")
    except Exception as e:
        logger.error(f"QS Momentum failed: {e}")
        ret[f"{close_column}_qsmom_21_252_126"] = np.nan

    # ROC Features
    try:
        ret = ret.groupby(symbol_column).augment_roc(
            date_column=date_column,
            close_column=close_column,
            periods=[63, 252],
            engine=engine,
        )
        logger.info("Added ROC features")
    except Exception as e:
        logger.error(f"Failed to add ROC features: {e}")

    # RSI
    try:
        ret = ret.groupby(symbol_column).augment_rsi(
            date_column=date_column,
            close_column=close_column,
            periods=[63],
            engine=engine,
        )
        logger.info("Added RSI")
    except Exception as e:
        logger.error(f"Failed to add RSI: {e}")

    # FIP Momentum
    try:
        ret = ret.groupby(symbol_column).augment_fip_momentum(
            date_column=date_column,
            close_column=close_column,
            window=[252],
            engine=engine,
        )
        logger.info("Added FIP Momentum")
    except Exception as e:
        logger.error(f"Failed to add FIP Momentum: {e}")

    # Volatility
    try:
        ret = ret.groupby(symbol_column).augment_rolling_risk_metrics(
            date_column=date_column,
            close_column=close_column,
            window=[252],
            engine=engine,
            metrics=["volatility_annualized"],
        )
        logger.info("Added Rolling Risk Metrics")
    except Exception as e:
        logger.error(f"Failed to add Rolling Risk Metrics: {e}")

    return ret


def add_rule_of_40_score(
    df,
    symbol_column="symbol",
    date_column="date",
    growth_col="revenueGrowth",
    margin_col="netProfitMargin",
    momentum_col="close_roc_0_252"
):
    logger.info("Computing Rule of 40 Score")
    
    df_pl = pl.from_pandas(df)

    df_pl = df_pl.with_columns(
        (pl.col(growth_col) + pl.col(margin_col)).alias("rule_of_40_score")
    ).with_columns(
        (pl.col("rule_of_40_score") * pl.col(momentum_col)).alias("rule_of_40_momentum")
    )

    return df_pl.to_pandas()

def add_fcf_growth_momentum(
    df,
    symbol_column="symbol",
    date_column="date",
    fcf_growth_col="growthFreeCashFlow",
    price_col="close",
    fcf_per_share_col="freeCashFlowPerShare",
    momentum_col="close_roc_0_252"
):
    logger.info("Computing FCF Growth + Momentum Score")
    
    df_pl = pl.from_pandas(df)

    df_pl = df_pl.with_columns(
        (pl.col(fcf_per_share_col) / pl.col(price_col)).alias("fcf_yield")
    ).with_columns(
        (pl.col(fcf_growth_col) * pl.col("fcf_yield") * pl.col(momentum_col)).alias("fcf_growth_momentum_score")
    )

    return df_pl.to_pandas()

def add_lynch_multibagger_score(
    df,
    symbol_column="symbol",
    date_column="date",
    pe_col="peRatio",
    eps_growth_col="EPSGrowth",
    roe_col="returnOnEquity",
    debt_col="debtEquityRatio",
    momentum_col="close_roc_0_252"
):
    logger.info("Computing Peter Lynch Multibagger Score")
    
    df_pl = pl.from_pandas(df)

    df_pl = df_pl.with_columns(
        (pl.col(pe_col) / (pl.col(eps_growth_col) * 100)).alias("peg_ratio")
    ).with_columns(
        ((1 / pl.col("peg_ratio")) * (pl.col(roe_col) - pl.col(debt_col)) * pl.col(momentum_col)).alias("lynch_multibagger_score")
    )

    return df_pl.to_pandas()

def rule_of_40_screener(
    df,
    min_score=0.40,
    symbol_column="symbol",
    date_column="date"
):
    logger.info(f"Filtering universe with Rule of 40 Score > {min_score}")
    
    df_pl = pl.from_pandas(df) 

    latest_date = df_pl[date_column].max()
    df_latest = df_pl.filter(pl.col(date_column) == latest_date)
    filtered_symbols = df_latest.filter(
        pl.col("rule_of_40_score") > min_score
    )[symbol_column].unique()

    ret = df_pl.filter(pl.col(symbol_column).is_in(filtered_symbols))

    logger.info(f"Filtered to {len(filtered_symbols)} symbols")
    return ret.to_pandas()

def add_fundamentals_from_database(
    df_price,
    conn=None,
    table="bulk_ratios_annual_fmp",
    columns=None,
    left_on="date",
    right_on="date",
):
    logger.info(f"Merging fundamentals from {table} for provided symbols")
    if conn is None:
        from qsconnect import Client
        client = Client()
        conn = client.connect_to_database()

    if not isinstance(df_price, pl.DataFrame):
        df_price = pl.from_pandas(df_price)
        
    if "ratios" in table or "growth" in table or "key_metrics" in table:
        right_on = "date"

    # Retrieve only symbols in df_price
    symbols_list = df_price["symbol"].unique().to_list()
    date_min, date_max = df_price[left_on].min(), df_price[left_on].max()

    # Retrieve all columns if none specified
    if columns is None:
        columns_query = f"DESCRIBE {table}"
        columns_df = conn.execute(columns_query).fetchdf()
        exclude_cols = {"symbol", "date", "fillingDate", "reportedCurrency", "period"}
        columns = [col for col in columns_df["column_name"] if col not in exclude_cols]
        logger.info(f"Loaded columns from {table}: {columns}")

    selected_cols = ", ".join(["symbol", f"{right_on} AS join_date"] + columns)

    query = f"""
        SELECT {selected_cols}
        FROM {table}
        WHERE symbol IN ({', '.join(repr(s) for s in symbols_list)})
        AND {right_on} BETWEEN '{date_min}' AND '{date_max}'
    """

    fundamentals_df = conn.execute(query).pl()

    df_price = df_price.with_columns(pl.col(left_on).cast(pl.Date))
    fundamentals_df = fundamentals_df.with_columns(pl.col("join_date").cast(pl.Date))

    df_price = df_price.sort(by=["symbol", left_on])
    fundamentals_df = fundamentals_df.sort(by=["symbol", "join_date"])

    joined = df_price.join_asof(
        fundamentals_df,
        left_on=left_on,
        right_on="join_date",
        by="symbol",
        strategy="backward",
    )

    filled = joined.with_columns(
        [pl.col(col).forward_fill().over("symbol") for col in columns]
    )
    
    # Drop join_date column
    filled = filled.drop("join_date")

    logger.info("Successfully merged fundamentals")
    return filled.to_pandas()

def add_fcf_growth_from_database(
    df_price,
    conn=None,
    table="bulk_cash_flow_statement_growth_annual_fmp",
    columns=["growthFreeCashFlow"],
    left_on="date",
    right_on="date",
):
    logger.info(f"Merging FCF growth from {table} for provided symbols")
    if conn is None:
        from qsconnect import Client
        client = Client()
        conn = client.connect_to_database()

    if not isinstance(df_price, pl.DataFrame):
        df_price = pl.from_pandas(df_price)
    
    if "ratios" in table or "growth" in table or "key_metrics" in table:
        right_on = "date"

    symbols_list = df_price["symbol"].unique().to_list()
    date_min, date_max = df_price[left_on].min(), df_price[left_on].max()

    selected_cols = ", ".join(["symbol", f"{right_on} AS join_date"] + columns)

    query = f"""
        SELECT {selected_cols}
        FROM {table}
        WHERE symbol IN ({', '.join(repr(s) for s in symbols_list)})
        AND {right_on} BETWEEN '{date_min}' AND '{date_max}'
    """

    fcf_growth_df = conn.execute(query).pl()

    df_price = df_price.with_columns(pl.col(left_on).cast(pl.Date))
    fcf_growth_df = fcf_growth_df.with_columns(pl.col("join_date").cast(pl.Date))

    df_price = df_price.sort(by=["symbol", left_on])
    fcf_growth_df = fcf_growth_df.sort(by=["symbol", "join_date"])

    joined = df_price.join_asof(
        fcf_growth_df,
        left_on=left_on,
        right_on="join_date",
        by="symbol",
        strategy="backward",
    )

    filled = joined.with_columns(
        [pl.col(col).forward_fill().over("symbol") for col in columns]
    )
    
    # Drop join_date column
    filled = filled.drop("join_date")

    logger.info("Successfully merged FCF growth data")
    return filled.to_pandas()

def add_eps_growth_from_database(
    df_price,
    conn=None,
    table="bulk_financial_growth_annual_fmp",
    columns=["EPSGrowth", "revenueGrowth"],
    left_on="date",
    right_on="date",
):
    logger.info(f"Merging EPS and revenue growth from {table} for provided symbols")
    if conn is None:
        from qsconnect import Client
        client = Client()
        conn = client.connect_to_database()

    if not isinstance(df_price, pl.DataFrame):
        df_price = pl.from_pandas(df_price)

    symbols_list = df_price["symbol"].unique().to_list()
    date_min, date_max = df_price[left_on].min(), df_price[left_on].max()

    selected_cols = ", ".join(["symbol", f"{right_on} AS join_date"] + columns)

    query = f"""
        SELECT {selected_cols}
        FROM {table}
        WHERE symbol IN ({', '.join(repr(s) for s in symbols_list)})
        AND {right_on} BETWEEN '{date_min}' AND '{date_max}'
    """

    eps_growth_df = conn.execute(query).pl()

    df_price = df_price.with_columns(pl.col(left_on).cast(pl.Date))
    eps_growth_df = eps_growth_df.with_columns(pl.col("join_date").cast(pl.Date))

    df_price = df_price.sort(by=["symbol", left_on])
    eps_growth_df = eps_growth_df.sort(by=["symbol", "join_date"])

    joined = df_price.join_asof(
        eps_growth_df,
        left_on=left_on,
        right_on="join_date",
        by="symbol",
        strategy="backward",
    )

    filled = joined.with_columns(
        [pl.col(col).forward_fill().over("symbol") for col in columns]
    )
    
    # Drop join_date column
    filled = filled.drop("join_date")

    logger.info("Successfully merged EPS and revenue growth data")
    return filled.to_pandas()
