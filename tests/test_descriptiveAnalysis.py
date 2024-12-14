import unittest
import pandas as pd
from scripts.descriptiveAnalysis import (
    headline_length_stats,
    articles_per_publisher,
    articles_by_day_of_week,
    articles_by_time,
    extract_domains,
    identify_unique_domains
)

class TestDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample data for testing
        cls.data = pd.DataFrame({
            'headline': [
                'Stock Prices Surge After Market News',
                'Company Reports Record Earnings',
                'Analysts Predict Market Downturn'
            ],
            'publisher': [
                'john.doe@example.com',
                'jane.smith@domain.com',
                'info@anotherdomain.org'
            ],
            'date': pd.to_datetime([
                '2024-08-01 10:00:00',
                '2024-08-02 14:30:00',
                '2024-08-03 09:00:00'
            ])
        })

    def test_headline_length_stats(self):
        stats = headline_length_stats(self.data)
        
        self.assertIsInstance(stats, pd.Series)
        self.assertIn('mean', stats.index)
        self.assertIn('std', stats.index)
        self.assertIn('min', stats.index)
        self.assertIn('max', stats.index)

    def test_articles_per_publisher(self):
        publisher_counts = articles_per_publisher(self.data)
        
        self.assertIsInstance(publisher_counts, pd.Series)
        self.assertEqual(len(publisher_counts), 3)
        self.assertIn('john.doe@example.com', publisher_counts.index)
        self.assertIn('jane.smith@domain.com', publisher_counts.index)

    def test_articles_by_day_of_week(self):
        day_of_week_counts = articles_by_day_of_week(self.data)
        
        self.assertIsInstance(day_of_week_counts, pd.Series)
        self.assertIn('Thursday', day_of_week_counts.index)
        self.assertIn('Friday', day_of_week_counts.index)

    def test_articles_by_time(self):
        time_counts = articles_by_time(self.data)
        
        self.assertIsInstance(time_counts, pd.Series)
        
        # Correctly check if the times are in the result
        expected_times = [pd.Timestamp('2024-08-01 10:00:00').time(),
                          pd.Timestamp('2024-08-02 14:30:00').time(),
                          pd.Timestamp('2024-08-03 09:00:00').time()]
        
        for expected_time in expected_times:
            self.assertIn(expected_time, time_counts.index)

    def test_extract_domains(self):
        domain = extract_domains('john.doe@example.com')
        self.assertEqual(domain, 'example.com')
        
        domain = extract_domains('info@anotherdomain.org')
        self.assertEqual(domain, 'anotherdomain.org')
        
        domain = extract_domains('invalid-email')
        self.assertIsNone(domain)

    def test_identify_unique_domains(self):
        domain_counts = identify_unique_domains(self.data)
        
        self.assertIsInstance(domain_counts, pd.DataFrame)
        self.assertIn('domain', domain_counts.columns)
        self.assertIn('count', domain_counts.columns)
        self.assertEqual(len(domain_counts), 3)
        self.assertIn('example.com', domain_counts['domain'].values)
        self.assertIn('domain.com', domain_counts['domain'].values)

if __name__ == "__main__":
    unittest.main()