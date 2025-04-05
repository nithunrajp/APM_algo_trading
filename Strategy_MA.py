import pandas as pd 
import numpy as np 
import os
import useful_function as uf
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import useful_function as uf

df = pd.read_csv('DF_data_cleaned.csv', parse_dates=['Date'], index_col='Date')

returns= df.pct_change().dropna()

### Preliminary plot 

#### date and cumulative sum 

x_data_date = returns.index
y_data_return = returns.sum(axis=1).cumsum() 

#### show it 

plt.plot(x_data_date, y_data_return) 

plt.xlabel("Date")
plt.ylabel("Cumulative return")

plt.show()

def ema_signal(prices, short_window, long_window):
    # Compute short & long EMA
    short_ema = prices.shift(1).ewm(span=short_window, adjust=False).mean()
    long_ema = prices.shift(1).ewm(span=long_window, adjust=False).mean()

    # Generate raw signals
    
    signals = pd.Series(0, index=prices.index)
    signals[short_ema > long_ema] = 1
    signals[short_ema < long_ema] = -1

    return signals.fillna(0)

# create positions df 

positions = pd.DataFrame(index=returns.index, columns=returns.columns)

# Generate signals for each futures contract
for col in returns.columns:
    positions[col] = ema_signal(df[col], short_window=5, long_window=300)


pnl = (positions * returns).sum(axis=1)
pnl

def optimize_ema(short_window, long_window):
    # Make sure short_window is less than long_window
    if short_window >= long_window:
        return -1e6  # big penalty

    short_window = int(short_window)
    long_window = int(long_window)

    positions = pd.DataFrame(index=returns.index, columns=returns.columns)

    for col in returns.columns:
        signals = ema_signal(df[col], short_window, long_window)
        positions[col] = signals

    pnl = (positions * returns).sum(axis=1)
    sharpe = uf.get_sr(pnl)

    return sharpe


optimizer = BayesianOptimization(
    f=optimize_ema,
    pbounds={
        'short_window': (5, 50),
        'long_window': (100, 500),
    },
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=40)
