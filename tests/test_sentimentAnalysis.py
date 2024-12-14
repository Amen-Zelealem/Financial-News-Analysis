import unittest
import pandas as pd
from collections import Counter
from scripts.sentimentAnalysis import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for the unit tests.
        """
        self.headlines = pd.Series([
            "The market is up today!",
            "Economic downturn expected next quarter.",
            "Company XYZ reports record earnings."
        ])

    def test_preprocess_text(self):
        text = "Hello, World! This is a test sentence."
        expected_output = "hello world test sentence"
        self.assertEqual(SentimentAnalyzer.preprocess_text(text), expected_output)

    def test_analyze_sentiment(self):
        result_df = SentimentAnalyzer.analyze_sentiment(self.headlines)
        self.assertEqual(result_df.shape[0], len(self.headlines))
        self.assertIn('compound', result_df.columns)
    
    def test_categorize_sentiment(self):
        self.assertEqual(SentimentAnalyzer.categorize_sentiment(0.1), 'Positive')
        self.assertEqual(SentimentAnalyzer.categorize_sentiment(-0.1), 'Negative')
        self.assertEqual(SentimentAnalyzer.categorize_sentiment(0), 'Neutral')
    
    def test_get_common_keywords(self):
        common_keywords = SentimentAnalyzer.get_common_keywords(self.headlines)
        self.assertGreater(len(common_keywords), 0)
        self.assertTrue(all(isinstance(word, str) and isinstance(freq, int) for word, freq in common_keywords))
    
    def test_plot_wordcloud(self):
        word_freq = Counter({'test': 3, 'sentence': 2, 'word': 1})
        try:
            SentimentAnalyzer.plot_wordcloud(word_freq)
        except Exception as e:
            self.fail(f"plot_wordcloud raised {e} unexpectedly!")

if __name__ == '__main__':
    unittest.main()