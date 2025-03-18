import pandas as pd 
import numpy as np 
import os

# Define the directory containing CSV files
csv_directory = "sp500/csv"

# List all CSV files in the directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith(".csv")]

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through CSV files
for file in csv_files:
    file_path = os.path.join(csv_directory, file)
    
    # Read CSV into DataFrame
    df = pd.read_csv(file_path, usecols=["Date", "Adjusted Close"])
    
    # Rename "Adjusted Close" column to "filename_adjclose"
    stock_name = os.path.splitext(file)[0]  # Extract filename without .csv
    df.rename(columns={"Adjusted Close": f"{stock_name}_adjclose"}, inplace=True)
    
    # Append to list
    dfs.append(df)

# Merge all DataFrames on "Date"
merged_df = dfs[0]

# Define moving avg 

ma_df_st = merged_df['A_adjclose'].ewm(span=20, adjust=False).mean()
ma_df_lt = merged_df['A_adjclose'].ewm(span=100, adjust=False).mean() 

if ma_df_st > ma_df_lt:
    print("buy")
else: 
    print("sell")
