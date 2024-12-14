import unittest
import os
import pandas as pd
import zipfile
from scripts.dataProcessing import extract_zip, load_csv_from_zip, load_data

class TestDataProcessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Prepare a test zip file and CSV for testing
        cls.test_zip = 'test_data.zip'
        cls.extract_dir = 'test_extract'
        cls.csv_filename = 'test_data.csv'
        
        # Create test directory
        os.makedirs(cls.extract_dir, exist_ok=True)
        
        # Create a test CSV file
        cls.test_csv_content = "index,date,value\n1,2024-01-01,100\n2,2024-01-02,200"
        with open(cls.csv_filename, 'w') as f:
            f.write(cls.test_csv_content)
        
        # Create a zip file containing the CSV
        with zipfile.ZipFile(cls.test_zip, 'w') as zipf:
            zipf.write(cls.csv_filename)
    
    @classmethod
    def tearDownClass(cls):
        # Clean up test files and directories
        if os.path.exists(cls.test_zip):
            os.remove(cls.test_zip)
        if os.path.exists(cls.csv_filename):
            os.remove(cls.csv_filename)
        if os.path.exists(cls.extract_dir):
            for root, dirs, files in os.walk(cls.extract_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(cls.extract_dir)

    def test_extract_zip(self):
        extract_zip(self.test_zip, self.extract_dir)
        # Check if the CSV file was extracted
        self.assertTrue(os.path.isfile(os.path.join(self.extract_dir, self.csv_filename)))
    
    def test_load_csv_from_zip(self):
        extract_zip(self.test_zip, self.extract_dir)
        df = load_csv_from_zip(self.extract_dir, self.csv_filename)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))  # 2 rows and 2 columns
        self.assertIn('date', df.columns)
        self.assertIn('value', df.columns)
    
    def test_load_data(self):
        df = load_data(self.test_zip, self.csv_filename)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))  # 2 rows and 2 columns
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']))

if __name__ == "__main__":
    unittest.main()