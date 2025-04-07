import pandas as pd

def calculate_moving_average(data, window=30):
    """
    Calculate the moving average (MA) for each stock's PE ratio.
    """
    # Apply rolling mean for each stock
    data_ma = data.set_index('Dates').rolling(window=window, min_periods=1).mean().reset_index()
    return data_ma

def generate_trading_signals(data, ma_data):
    """
    Generate trading signals (Buy/Sell) based on PE ratio and its moving average.
    """
    signals = pd.DataFrame()
    for stock in data.columns[1:]:  # Skip the 'Dates' column
        stock_data = data[['Dates', stock]].copy()
        stock_ma = ma_data[['Dates', stock]].copy()
        
        stock_data['MA'] = stock_ma[stock]  # Add moving average
        stock_data['Signal'] = 'Hold'  # Default signal
        
        # Generate signals
        stock_data.loc[stock_data[stock] > stock_data['MA'], 'Signal'] = 'Sell'
        stock_data.loc[stock_data[stock] < stock_data['MA'], 'Signal'] = 'Buy'
        
        # Append stock name for identification
        stock_data['Stock'] = stock
        signals = pd.concat([signals, stock_data], ignore_index=True)
    
    return signals

def simulate_trading(signals, price_data):
    """
    Simulate trading based on signals and calculate profits/losses.
    """
    results = []
    for stock in signals['Stock'].unique():
        stock_signals = signals[signals['Stock'] == stock]
        stock_prices = price_data[['Dates', stock]].copy()
        
        stock_signals = pd.merge(stock_signals, stock_prices, on='Dates', how='left')
        stock_signals.rename(columns={stock: 'Price'}, inplace=True)
        
        # Simulate trading
        profit = 0
        position = None
        entry_price = 0
        
        for _, row in stock_signals.iterrows():
            if row['Signal'] == 'Buy' and position is None:
                position = 'Long'
                entry_price = row['Price']
            elif row['Signal'] == 'Sell' and position == 'Long':
                profit += row['Price'] - entry_price
                position = None
        
        results.append({'Stock': stock, 'Profit': profit})
    
    return pd.DataFrame(results)

def main():
    # Load PE ratio data from Excel
    excel_file = "PE RATIO.xlsx"  # Path to the Excel file
    pe_data = pd.read_excel(excel_file, sheet_name="PE_ratio_hist")  # Load the specified sheet
    
    # Ensure the 'Dates' column is in datetime format
    pe_data['Dates'] = pd.to_datetime(pe_data['Dates'], format="%d%m%Y")
    
    # Calculate moving average for PE ratios
    pe_ma = calculate_moving_average(pe_data, window=30)
    
    # Generate trading signals
    signals = generate_trading_signals(pe_data, pe_ma)
    
    # Load price data (assume it has the same structure as PE data)
    price_data = pd.read_excel(excel_file, sheet_name="Price_hist")  # Load price data
    
    # Simulate trading and calculate profits/losses
    results = simulate_trading(signals, price_data)
    
    print("Trading Results:")
    print(results)
    
    # Save results to a CSV file
    results.to_csv("trading_results.csv", index=False)

if __name__ == "__main__":
    main()