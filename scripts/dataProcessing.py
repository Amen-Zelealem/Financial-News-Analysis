import os
import zipfile
import pandas as pd


def extract_zip(zip_path: str, extract_to: str) -> None:
    # Open the zip file and extract its contents to the specified directory
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def load_csv_from_zip(extracted_dir: str, filename: str) -> pd.DataFrame:
    # Construct the full file path for the CSV file and load it into a DataFrame
    file_path = os.path.join(extracted_dir, filename)
    return pd.read_csv(file_path, index_col=0)


def load_data(zip_path: str, filename: str) -> pd.DataFrame:
    try:
        extract_to = (
            "../data"  # Define the directory where the zip contents will be extracted
        )
        extract_zip(zip_path, extract_to)  # Extract the zip file

        df = load_csv_from_zip(
            extract_to, filename
        )  # Load the CSV file into a DataFrame
        # Convert 'Date' column to datetime format, coerce errors to NaT
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        # Convert 'date' column to datetime format with a mixed format
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)

        return df  # Return the loaded DataFrame

    except Exception as e:
        # Raise a RuntimeError with a message if any exceptions occur
        raise RuntimeError(f"Error loading data: {str(e)}")
