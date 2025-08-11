# QSResearch Trading Strategy Development Guide

## How This Quant Trading System Works

### System Architecture Overview

```
Data Source → QSResearch Pipeline → Backtest → MLflow Tracking → Live Trading
     ↓              ↓                 ↓           ↓              ↓
  Zipline        Preprocessing    Performance   Experiment    Omega/IBKR
  Bundle         + Features        Analysis     Management    Integration
```

### Core Components

#### A. Data Layer (Zipline Bundle)
- **Purpose**: Historical price data storage
- **Your Bundle**: `qspro_demo_historical_prices_fmp`
- **Format**: OHLCV data optimized for backtesting
- **Location**: Created in Clinic #5

#### B. QSResearch Pipeline
- **Preprocessing**: Universe screening, data cleaning
- **Feature Engineering**: Technical indicators, custom factors
- **Algorithm**: Signal generation logic
- **Portfolio Construction**: Position sizing and allocation

#### C. MLflow Tracking
- **Experiments**: Group related strategy tests
- **Runs**: Individual backtest executions
- **Artifacts**: Performance reports, config files
- **Metrics**: Sharpe ratio, returns, drawdowns

#### D. Live Trading Integration
- **Omega**: Interface to Interactive Brokers
- **Position Management**: Automatic rebalancing
- **Risk Controls**: Stop-losses, position limits

## What You Need to Install Your Own Trading Strategy

### 1. Required Strategy Components

#### A. Factor/Signal Function
```python
def your_custom_signal(data, **params):
    """
    Your signal generation logic
    Returns: DataFrame with signal column
    """
    # Your custom logic here
    return data_with_signals
```

#### B. Portfolio Construction Function
```python
def your_portfolio_strategy(predictions, **params):
    """
    Convert signals to portfolio weights
    Returns: Dictionary of {symbol: weight}
    """
    # Your position sizing logic
    return weights
```

#### C. Configuration Dictionary
```python
YOUR_CONFIG = {
    # MLflow settings
    "mlflow_experiment_name": "Your Strategy Name",
    
    # Your preprocessing pipeline
    "preprocess": [
        {"name": "your_feature", "func": your_custom_signal, "params": {...}}
    ],
    
    # Your algorithm
    "algorithm": {"func": your_signal_func, "params": {...}},
    
    # Your portfolio construction
    "portfolio_strategy": {"func": your_portfolio_func, "params": {...}}
}
```

### 2. Step-by-Step Strategy Installation Process

#### Step 1: Create Your Feature Engineering Function
```python
def add_your_features(data, **params):
    """Add your custom technical indicators or factors"""
    # Example: Moving average crossover
    data['ma_short'] = data.groupby('symbol')['close'].rolling(20).mean()
    data['ma_long'] = data.groupby('symbol')['close'].rolling(50).mean()
    data['ma_signal'] = (data['ma_short'] > data['ma_long']).astype(int)
    return data
```

#### Step 2: Create Your Signal Generation Algorithm
```python
def your_trading_algorithm(train_data, predict_data, factor_column, **params):
    """Generate trading signals based on your logic"""
    # Your strategy logic here
    signals = predict_data[factor_column].rank(pct=True)  # Example ranking
    return signals
```

#### Step 3: Create Your Portfolio Construction
```python
def your_portfolio_construction(predictions, num_positions=10, **params):
    """Convert signals to portfolio weights"""
    # Select top N positions
    top_stocks = predictions.nlargest(num_positions)
    weights = {symbol: 1.0/num_positions for symbol in top_stocks.index}
    return weights
```

#### Step 4: Build Your Configuration
```python
YOUR_STRATEGY_CONFIG = {
    "mlflow_experiment_name": "Your Strategy Name",
    "mlflow_run_name": "Your Strategy Test 1",
    
    "preprocess": [
        {"name": "your_features", "func": add_your_features, "params": {...}},
        # Include standard preprocessing too
        {"name": "screener", "func": universe_screener, "params": {...}},
    ],
    
    "algorithm": {
        "func": your_trading_algorithm,
        "params": {"factor_column": "your_signal_column"}
    },
    
    "portfolio_strategy": {
        "func": your_portfolio_construction,
        "params": {"num_positions": 15}
    }
}
```

#### Step 5: Test Your Strategy
```python
results = run_backtest(YOUR_STRATEGY_CONFIG)
```

### 3. Key Files You'll Need to Modify/Create

1. **Strategy File**: `your_strategy_name.py` or `.ipynb`
2. **Custom Functions**: Your feature engineering and signal generation
3. **Configuration**: Your strategy's CONFIG dictionary
4. **Testing Script**: Backtest execution and analysis

### 4. Integration Points with Existing System

#### Preprocessing Pipeline
- **Add your functions** to the `preprocess` list
- **Maintain data format** (symbol, date, OHLCV columns)
- **Use existing functions** (screener, price_preprocessor) as needed

#### MLflow Integration
- **Choose unique experiment names** to avoid conflicts
- **Use consistent tagging** for easy organization
- **Leverage existing artifacts** (performance reports)

#### Live Trading
- **Use same Omega integration** for live execution
- **Implement risk controls** appropriate for your strategy
- **Test thoroughly** in paper trading first

### 5. Best Practices for Strategy Development

1. **Start Simple**: Begin with basic moving averages or momentum
2. **Test Incrementally**: Add complexity gradually
3. **Use MLflow**: Track all experiments systematically
4. **Validate Thoroughly**: Long-term backtests before live trading
5. **Risk Management**: Always include stop-losses and position limits

## Example: Simple Moving Average Crossover Strategy

### Complete Implementation

```python
# File: ma_crossover_strategy.py

import pandas as pd
import numpy as np
from qsresearch.strategies.factor import run_backtest
from qsresearch.preprocessors import universe_screener, preprocess_price_data
from zipline.api import date_rules, time_rules

def add_ma_crossover_features(data, fast_period=20, slow_period=50, **params):
    """Add moving average crossover signals"""
    data = data.copy()
    
    # Calculate moving averages
    data['ma_fast'] = data.groupby('symbol')['close'].rolling(fast_period).mean().reset_index(0, drop=True)
    data['ma_slow'] = data.groupby('symbol')['close'].rolling(slow_period).mean().reset_index(0, drop=True)
    
    # Generate crossover signal
    data['ma_crossover_signal'] = (data['ma_fast'] > data['ma_slow']).astype(int)
    
    # Signal strength (how far apart the MAs are)
    data['ma_spread'] = (data['ma_fast'] - data['ma_slow']) / data['ma_slow']
    
    return data

def ma_crossover_algorithm(train_data, predict_data, factor_column='ma_crossover_signal', **params):
    """Generate signals based on MA crossover"""
    # Simple binary signal: 1 for buy, 0 for no position
    signals = predict_data[factor_column]
    return signals

def equal_weight_long_only_portfolio(predictions, num_positions=10, **params):
    """Equal weight portfolio of top signals"""
    # Filter for buy signals (signal = 1)
    buy_signals = predictions[predictions > 0]
    
    if len(buy_signals) == 0:
        return {}
    
    # Take top N positions
    selected = buy_signals.head(min(num_positions, len(buy_signals)))
    
    # Equal weight
    weight = 1.0 / len(selected)
    weights = {symbol: weight for symbol in selected.index}
    
    return weights

# Strategy Configuration
MA_CROSSOVER_CONFIG = {
    # MLflow Tracking
    "use_mlflow": True,
    "mlflow_tracking_uri": "/Users/brucebrownlee/dev/github/Resident/QS-Project/Clinic-06/mlruns",
    "mlflow_experiment_name": "Moving Average Strategies",
    "mlflow_run_name": "MA Crossover 20/50 Equal Weight",
    "mlflow_tags": {"strategy": "ma_crossover", "portfolio": "equal_weight"},
    
    # Backtest Parameters
    "start_date": pd.Timestamp("2024-01-01"),
    "end_date": pd.Timestamp("2025-07-01"),
    "capital_base": 1_000_000,
    "bundle_name": "qspro_demo_historical_prices_fmp",
    "benchmark_symbol": "SPY",
    "window_length": 252 * 2,
    "frequency": "1d",
    "predictor_cols": ["ma_crossover_signal"],
    
    # Rebalancing
    "rebalance_schedule": {
        "date_rule": date_rules.week_start(),
        "time_rule": time_rules.market_open(minutes=60),
    },
    
    # Transaction Costs
    "transaction_costs": {
        "slippage": {"spread": 0.01},
        "commission": {"cost": 0.005, "min_trade_cost": 0},
    },
    
    # Preprocessing Pipeline
    "preprocess": [
        {
            "name": "screener",
            "func": universe_screener,
            "params": {
                "volume_top_n": 200,
                "min_avg_volume": 1_000_000,
                "min_avg_price": 10.0,
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
                "min_trading_days": 252,
                "remove_low_trading_days": True,
                "remove_large_gaps": True,
                "remove_low_volume": True,
                "symbol_column": "symbol",
                "date_column": "date",
                "engine": "polars",
            },
        },
        {
            "name": "ma_features",
            "func": add_ma_crossover_features,
            "params": {
                "fast_period": 20,
                "slow_period": 50,
            },
        },
    ],
    
    # Algorithm
    "algorithm": {
        "func": ma_crossover_algorithm,
        "params": {
            "factor_column": "ma_crossover_signal",
        },
    },
    
    # Portfolio Construction
    "portfolio_strategy": {
        "func": equal_weight_long_only_portfolio,
        "params": {
            "num_positions": 15,
        },
    },
}

# Run the backtest
if __name__ == "__main__":
    results = run_backtest(MA_CROSSOVER_CONFIG)
    print("Backtest completed successfully!")
```

## How to Use This Guide

1. **Read through the architecture** to understand the system
2. **Use the example strategy** as a template for your own strategies
3. **Modify the functions** to implement your trading logic
4. **Test incrementally** with small changes
5. **Use MLflow** to track and compare different versions
6. **Refer back to this guide** when creating new strategies

## Next Steps

1. **Study the momentum strategy** in `02_qsmomentum_factor_research.ipynb`
2. **Create your first custom strategy** using the template above
3. **Test with paper trading** before going live
4. **Build a library** of your proven strategies

---

*Created: August 10, 2025*  
*For: QS-Project Clinic #6*  
*Purpose: Reference guide for custom strategy development*
