import unittest
import pandas as pd
from scripts.publicationAnalysis import analyze_annual_trends, analyze_quarterly_trends, plot_long_term_trends, decompose_time_series

class TestPublicationAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample data
        cls.data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100, freq='W'),
        })
        # Add dummy 'no_of_articles' column for testing
        cls.data['no_of_articles'] = 1

    def test_analyze_annual_trends(self):
        annual_counts = analyze_annual_trends(self.data)
        
        self.assertIsInstance(annual_counts, pd.DataFrame)
        self.assertIn('date', annual_counts.columns)
        self.assertIn('no_of_articles', annual_counts.columns)
        self.assertEqual(annual_counts['date'].dtype, 'datetime64[ns]')

    def test_analyze_quarterly_trends(self):
        quarterly_counts = analyze_quarterly_trends(self.data)
        
        self.assertIsInstance(quarterly_counts, pd.DataFrame)
        self.assertIn('date', quarterly_counts.columns)
        self.assertIn('no_of_articles', quarterly_counts.columns)
        self.assertEqual(quarterly_counts['date'].dtype, 'datetime64[ns]')

    def test_plot_long_term_trends(self):
        annual_counts = analyze_annual_trends(self.data)
        quarterly_counts = analyze_quarterly_trends(self.data)
        
        try:
            plot_long_term_trends(annual_counts, quarterly_counts)
        except Exception as e:
            self.fail(f"plot_long_term_trends failed: {e}")

if __name__ == "__main__":
    unittest.main()