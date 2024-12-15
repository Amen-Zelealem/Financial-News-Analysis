import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import statsmodels.api as sm

def analyze_annual_trends(data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze yearly publication counts.

    Parameters:
    - data (pd.DataFrame): Data with a 'date' column.

    Returns:
    - pd.DataFrame: Yearly article counts.
    """
    # Remove timezone information from 'date' before grouping
    data['date'] = data['date'].dt.tz_localize(None)
    # Group by year and count articles
    annual_counts = data.groupby(data['date'].dt.to_period('Y')).size().reset_index(name='no_of_articles')
    annual_counts['date'] = annual_counts['date'].dt.to_timestamp()
    return annual_counts

def analyze_quarterly_trends(data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze quarterly publication counts.

    Parameters:
    - data (pd.DataFrame): Data with a 'date' column.

    Returns:
    - pd.DataFrame: Quarterly article counts.
    """
    # Remove timezone information from 'date' before grouping
    data['date'] = data['date'].dt.tz_localize(None)
    # Group by quarter and count articles
    quarterly_counts = data.groupby(data['date'].dt.to_period('Q')).size().reset_index(name='no_of_articles')
    quarterly_counts['date'] = quarterly_counts['date'].dt.to_timestamp()
    return quarterly_counts

def plot_long_term_trends(annual_counts: pd.DataFrame, quarterly_counts: pd.DataFrame) -> None:
    """
    Plot annual and quarterly trends.

    Parameters:
    - annual_counts (pd.DataFrame): Yearly article counts.
    - quarterly_counts (pd.DataFrame): Quarterly article counts.
    """
    # Plot yearly trends
    plt.figure(figsize=(14, 7))
    plt.plot(annual_counts['date'], annual_counts['no_of_articles'], marker='o', linestyle='-', color='blue')
    plt.title('Annual Article Trends')
    plt.xlabel('Year')
    plt.ylabel('Articles')
    plt.grid(True)
    plt.show()

    # Plot quarterly trends
    plt.figure(figsize=(14, 7))
    plt.plot(quarterly_counts['date'], quarterly_counts['no_of_articles'], marker='o', linestyle='-', color='purple')
    plt.title('Quarterly Article Trends')
    plt.xlabel('Quarter')
    plt.ylabel('Articles')
    plt.grid(True)
    plt.show()

def decompose_time_series(data: pd.DataFrame, frequency: int) -> None:
    """
    Decompose time series into trend, seasonality, and residuals.

    Parameters:
    - data (pd.DataFrame): Data with a 'date' column.
    - frequency (int): Period for decomposition.

    Returns:
    - None: Displays decomposition plots.
    """
    # Set 'date' as index and decompose
    data.set_index('date', inplace=True)
    decomposed = sm.tsa.seasonal_decompose(data['no_of_articles'], model='additive', period=frequency)
    return decomposed