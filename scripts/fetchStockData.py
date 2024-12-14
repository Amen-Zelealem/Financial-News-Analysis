import pandas as pd
import pynance as pn

def fetch_historical_data(stocks, start_date="2022-01-01", end_date="2024-01-01"):
    """
    Fetch historical market data for a list of stocks.

    Parameters:
    stocks (tuple): A tuple of stock symbols.
    start_date (str): Start date for fetching the data in YYYY-MM-DD format.
    end_date (str): End date for fetching the data in YYYY-MM-DD format.

    Returns:
    pd.DataFrame: A DataFrame containing historical data for all stocks.
    """
    yfinance_data = []  # Initialize an empty list to store DataFrames

    for stock in stocks:
        # Fetch data for each stock
        data = pn.data.get(stock, start=start_date, end=end_date)
        yf_df = pd.DataFrame(data)  # Convert to a DataFrame
        yf_df['stock'] = stock  # Add the stock symbol to the DataFrame
        yfinance_data.append(yf_df)  # Append the DataFrame to the list

    # Concatenate all DataFrames in the list into a single DataFrame
    yfinance_df = pd.concat(yfinance_data)
    return yfinance_df