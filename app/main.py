import os
import sys
import pandas as pd
import streamlit as st

# Add the 'scripts' directory to the system path for importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

# Import functions from custom scripts
from stockAnalysis import load_data, plot_stock_data, plot_rsi, plot_macd
from sentimentAnalysis import SentimentAnalyzer as sa  # For sentiment analysis


# Helper function to construct file paths
def get_file_path(folder_name, file_name):
    """Constructs an absolute path for a file in a given folder."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, folder_name, file_name)


# Main Streamlit App
def main():
    # === App Title ===
    st.title("📊 Sentiment Analysis of Stock Data")
    st.write("Analyze stock data with various indicators and sentiment scores.")

    # === Data Loading ===
    # Define data file paths
    stock_data_path = get_file_path("data", "stock_data.csv")
    daily_sentiment_path = get_file_path("data", "daily_sentiment.csv")

    # Load datasets
    try:
        df = load_data(stock_data_path)
        daily_sentiment = pd.read_csv(daily_sentiment_path)
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return

    # === Sidebar Options ===
    st.sidebar.header("🔧 Configuration")
    stocks = df["stock"].unique()
    selected_stock = st.sidebar.selectbox("**Select Stock**", stocks)
    indicator = st.sidebar.selectbox(
        "**Select Indicator**",
        ["Moving Averages", "RSI", "MACD", "Daily Sentiment"],
    )

    # === Visualization Logic ===
    st.subheader(f"Selected Stock: **{selected_stock}**")
    st.write(f"Displaying: **{indicator}** Indicator")

    # Generate the appropriate plot based on the selected indicator
    try:
        if indicator == "Moving Averages":
            fig = plot_stock_data(selected_stock, df)
        elif indicator == "RSI":
            fig = plot_rsi(selected_stock, df)
        elif indicator == "MACD":
            fig = plot_macd(selected_stock, df)
        elif indicator == "Daily Sentiment":
            fig = sa.plot_sentiment(daily_sentiment, selected_stock)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred while generating the plot: {e}")

    # === Footer ===
    st.markdown(
        """
        <hr style="border:1px solid #ccc;">
        <p style="text-align:center; font-size:14px; color:grey;">
        &copy; 2024 Developed by <strong>Amen Zelealem</strong>
        </p>
        """,
        unsafe_allow_html=True,
    )


# Run the app
if __name__ == "__main__":
    main()
