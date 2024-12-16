import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Tuple, List, Dict

# Ensure you have the required nltk resources for tokenization and sentiment analysis
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    
    @staticmethod
    def analyze_sentiment(headlines: pd.Series) -> pd.DataFrame:
        """
        Analyze the sentiment of a series of headlines using VADER.

        Args:
            headlines (pd.Series): A pandas Series containing headlines.

        Returns:
            pd.DataFrame: A DataFrame containing the original headlines and their sentiment scores.
        """
        sia = SentimentIntensityAnalyzer()  # Initialize the sentiment intensity analyzer
        sentiments = headlines.apply(lambda x: sia.polarity_scores(x))  # Apply sentiment analysis
        sentiment_df = pd.DataFrame(sentiments.tolist())  # Convert the results to a DataFrame
        sentiment_df = pd.concat([headlines, sentiment_df], axis=1)  # Combine with original headlines
        return sentiment_df

    @staticmethod
    def categorize_sentiment(compound_score: float) -> str:
        """
        Categorize sentiment based on the compound score.

        Args:
            compound_score (float): The compound score of the sentiment.

        Returns:
            str: The sentiment category ('Positive', 'Negative', or 'Neutral').
        """
        if compound_score >= 0.05:
            return 'Positive'  # Positive sentiment
        elif compound_score <= -0.05:
            return 'Negative'  # Negative sentiment
        else:
            return 'Neutral'  # Neutral sentiment

    @staticmethod
    def apply_sentiment_categories(data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sentiment categories to a DataFrame based on the compound score.

        Args:
            data (pd.DataFrame): A DataFrame containing sentiment scores.

        Returns:
            pd.DataFrame: The DataFrame with an additional 'Sentiment' column.
        """
        data['Sentiment'] = data['compound'].apply(SentimentAnalyzer.categorize_sentiment)  # Categorize sentiments
        return data

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess the text for analysis by cleaning and tokenizing.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The cleaned and tokenized text.
        """
        text = text.lower()  # Convert text to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
        words = word_tokenize(text)  # Tokenize the text into words
        stop_words = set(stopwords.words('english'))  # Get English stopwords
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(words)  # Join cleaned words back into a single string

    @staticmethod
    def get_common_keywords(headlines: pd.Series, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get the most common keywords from the headlines.

        Args:
            headlines (pd.Series): A Series of headlines.
            top_n (int): The number of top keywords to return.

        Returns:
            List[Tuple[str, int]]: A list of tuples containing keywords and their frequencies.
        """
        cleaned_headlines = headlines.apply(SentimentAnalyzer.preprocess_text)  # Clean and preprocess headlines
        all_words = ' '.join(cleaned_headlines).split()  # Combine all words into a single list
        word_freq = Counter(all_words)  # Count the frequency of each word
        return word_freq.most_common(top_n)  # Return the top N keywords

    @staticmethod
    def plot_wordcloud(word_freq: Counter) -> None:
        """
        Plot a word cloud from the word frequencies.

        Args:
            word_freq (Counter): A Counter object with word frequencies.
        """
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)  # Create a word cloud
        plt.figure(figsize=(10, 8))  # Set figure size
        plt.imshow(wordcloud, interpolation='bilinear')  # Display the word cloud image
        plt.axis('off')  # Hide axis
        plt.title('Most Common Words in Headlines')  # Set title
        plt.show()  # Show the plot

    @staticmethod
    def perform_nlp_analysis(headlines: pd.Series) -> None:
        """
        Perform NLP analysis on the headlines, including keyword extraction and word cloud visualization.

        Args:
            headlines (pd.Series): A Series of headlines to analyze.
        """
        common_keywords = SentimentAnalyzer.get_common_keywords(headlines)  # Get common keywords
        print("Most common keywords:")  # Print header
        for word, freq in common_keywords:  # Print each keyword and its frequency
            print(f'{word}: {freq}')
        
        word_freq = Counter(dict(common_keywords))  # Convert common keywords to a Counter
        SentimentAnalyzer.plot_wordcloud(word_freq)  # Plot the word cloud

    def calculate_sentiment(df):
        """
        Calculate daily average sentiment scores.

        Parameters:
        - df (pd.DataFrame): DataFrame with sentiment columns and multi-index (Date, stock).

        Returns:
        - pd.DataFrame: DataFrame with daily average sentiment scores for each stock.
        """
        # Define sentiment columns
        sentiment_cols = ['neg', 'neu', 'pos']
        
        # Group by Date and stock, then compute mean of sentiment columns
        daily_sentiment = df.groupby(level=['Date', 'stock'])[sentiment_cols].mean().reset_index()
        
        return daily_sentiment

def plot_sentiment(daily_sentiment, stock):
        """
        Plot sentiment scores for a given stock.

        Parameters:
        - daily_sentiment (pd.DataFrame): DataFrame with daily sentiment scores.
        - stock (str): The stock symbol for which to plot sentiment scores.

        Returns:
        - plt.Figure: Matplotlib figure object.
        """
        # Filter data for the selected stock
        stock_data = daily_sentiment[daily_sentiment['stock'] == stock]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot sentiment scores
        ax.plot(stock_data['Date'], stock_data['neg'], label='Negative Sentiment', color='red')
        ax.plot(stock_data['Date'], stock_data['neu'], label='Neutral Sentiment', color='grey')
        ax.plot(stock_data['Date'], stock_data['pos'], label='Positive Sentiment', color='green')
        # ax.plot(stock_data['Date'], stock_data['compound'], label='Compound Sentiment', color='blue')
        
        # Set plot title and labels
        ax.set_title(f'Daily Sentiment Scores for {stock}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig