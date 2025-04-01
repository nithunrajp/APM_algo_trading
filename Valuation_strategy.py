import pandas as pd

def calculate_valuation_metrics(data):
    """
    Calculate valuation metrics such as PE ratio, book-to-market ratio, and price ratios.
    """
    data['PE_ratio'] = data['Price'] / data['Earnings']
    data['Book_to_Market'] = data['Book_Value'] / data['Price']
    data['Price_Ratio'] = data['Price'] / data['Adjusted_Close']
    return data

def select_stocks(data, top_n=10):
    """
    Select top N stocks based on valuation metrics.
    """
    data['Score'] = data['PE_ratio'] + data['Book_to_Market'] + data['Price_Ratio']
    selected_stocks = data.nsmallest(top_n, 'Score')  # Lower score is better
    return selected_stocks

def main():
    # Load stock data
    stock_data = pd.read_csv("stock_data.csv")
    
    # Example additional data for valuation metrics
    valuation_data = pd.read_csv("valuation_data.csv")  # Contains Price, Earnings, Book_Value columns
    
    # Merge stock data with valuation data
    merged_data = pd.merge(stock_data, valuation_data, on="Date", how="inner")
    
    # Calculate valuation metrics
    merged_data = calculate_valuation_metrics(merged_data)
    
    # Select top stocks
    top_stocks = select_stocks(merged_data)
    
    print("Top Selected Stocks:")
    print(top_stocks)
    
    # Save selected stocks to a CSV file
    top_stocks.to_csv("selected_stocks.csv", index=False)

if __name__ == "__main__":
    main()