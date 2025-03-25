import pandas as pd 
import numpy as np 
import os
import useful_function as uf
from bayes_opt import BayesianOptimization

df = pd.read_csv('data_cleaned.csv', parse_dates=['Date'], index_col='Date')

df_return= df.pct_change()

# Define moving avg 
def signal_ewm(df_return, short_window, long_window):
    short_ema = df_return.ewm(span=short_window, adjust=False).mean()
    long_ema = df_return.ewm(span=long_window, adjust=False).mean() 

        # Create empty positions Series
    signals = pd.Series(0, index=df_return.index)

    bullish = short_ema > long_ema
    bearish = short_ema < long_ema

    signals[bullish] = 1
    signals[bearish] = -1

    return signals

def backtest_strategy(df, signals):
    daily_returns = df.pct_change().fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * daily_returns  # Next-day execution

    return uf.get_sr(strategy_returns)

def multi_asset_performance(short_window, long_window): 
    sharpe_values = []

    for col in df_return.columns:
        # Generate trading signals
        signals = signal_ewm(
            df_return[col],
            short_window=int(round(short_window)),
            long_window=int(round(long_window))
        )
        # Compute Sharpe ratio for the asset
        sharpe = backtest_strategy(df_return[col], signals)
        sharpe_values.append(sharpe)

    return np.mean(sharpe_values)

# Define parameter search space
pbounds = {
    'short_window': (20,50),
    'long_window': (150,300),
 # Minimum volatility threshold
}

# Bayesian Optimization
optimizer = BayesianOptimization(
    f=multi_asset_performance,  # Function to optimize
    pbounds=pbounds,
    random_state=42
)

# Run optimization (5 random points, then 150 iterations)
optimizer.maximize(init_points=10, n_iter=150)

# Best found parameters
print("Best found parameters:", optimizer.max)