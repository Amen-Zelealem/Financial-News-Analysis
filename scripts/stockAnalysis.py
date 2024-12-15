import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load stock data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file containing stock data.

    Returns:
    - pd.DataFrame: Stock data with 'Date' as the index.
    """
    # Read CSV file into a DataFrame, parsing 'Date' column as index
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df

def plot_stock_data(stock, df):
    """
    Plot stock price with moving averages (20-day SMA and EMA).

    Parameters:
    - stock (str): The stock symbol to filter and plot data for.
    - df (pd.DataFrame): DataFrame containing stock data.

    Returns:
    - matplotlib.figure.Figure: The figure containing the stock price plot.
    """
    # Create a new figure and axis for plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Filter the DataFrame for the specified stock
    stock_data = df[df['stock'] == stock]
    
    # Plot the closing price of the stock
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    
    # Plot the 20-day Simple Moving Average (SMA)
    ax.plot(stock_data.index, stock_data['SMA_20'], label='20-Day SMA', color='orange')
    
    # Plot the 20-day Exponential Moving Average (EMA)
    ax.plot(stock_data.index, stock_data['EMA_20'], label='20-Day EMA', color='green')
    
    # Set plot title and labels
    ax.set_title(f'Stock Price for {stock} with SMA and EMA')
    ax.set_xlabel('Date')  # X-axis represents dates
    ax.set_ylabel('Price')  # Y-axis represents stock price
    
    # Add a legend to differentiate between lines
    ax.legend()
    
    # Add a grid for better visualization
    ax.grid()
    return fig

def plot_rsi(stock, df):
    """
    Plot the Relative Strength Index (RSI) of the stock.

    Parameters:
    - stock (str): The stock symbol to filter and plot data for.
    - df (pd.DataFrame): DataFrame containing stock data.

    Returns:
    - matplotlib.figure.Figure: The figure containing the RSI plot.
    """
    # Create a new figure and axis for the RSI plot
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Filter the DataFrame for the specified stock
    stock_data = df[df['stock'] == stock]
    
    # Plot the 14-day RSI of the stock
    ax.plot(stock_data.index, stock_data['RSI_14'], label='14-Day RSI', color='purple')
    
    # Add horizontal lines at 70 (overbought) and 30 (oversold) levels
    ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    
    # Set plot title and labels
    ax.set_title(f'RSI for {stock}')
    ax.set_xlabel('Date')  # X-axis represents dates
    ax.set_ylabel('RSI')  # Y-axis represents the RSI value
    
    # Add a legend to describe the lines
    ax.legend()
    
    # Add a grid for better visualization
    ax.grid()
    return fig

def plot_macd(stock, df):
    """
    Plot the Moving Average Convergence Divergence (MACD) of the stock.

    Parameters:
    - stock (str): The stock symbol to filter and plot data for.
    - df (pd.DataFrame): DataFrame containing stock data.

    Returns:
    - matplotlib.figure.Figure: The figure containing the MACD plot.
    """
    # Create a new figure and axis for the MACD plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Filter the DataFrame for the specified stock
    stock_data = df[df['stock'] == stock]
    
    # Plot the MACD line
    ax.plot(stock_data.index, stock_data['MACD'], label='MACD Line', color='black')
    
    # Plot the MACD signal line
    ax.plot(stock_data.index, stock_data['MACD_Signal'], label='Signal Line', color='red')
    
    # Plot the MACD histogram as a bar chart
    ax.bar(stock_data.index, stock_data['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.5)
    
    # Set plot title and labels
    ax.set_title(f'MACD for {stock}')
    ax.set_xlabel('Date')  # X-axis represents dates
    ax.set_ylabel('MACD')  # Y-axis represents the MACD value
    
    # Add a legend to describe the lines and histogram
    ax.legend()
    
    # Add a grid for better visualization
    ax.grid()
    return fig
