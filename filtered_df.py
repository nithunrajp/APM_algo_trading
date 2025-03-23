import pandas as pd 

df = pd.read_csv("stock_data.csv")

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

date_from = pd.to_datetime('1980-17-03', format='%Y-%d-%m')

df = df.loc[date_from:].dropna(axis=1)

df.to_csv("data_cleaned.csv")