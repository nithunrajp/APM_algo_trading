import pandas as pd 
import numpy as np 
import os
import useful_function as uf
from bayes_opt import BayesianOptimization

df = pd.read_csv('data_cleaned.csv', parse_dates=['Date'], index_col='Date')

df_return= df.pct_change()

