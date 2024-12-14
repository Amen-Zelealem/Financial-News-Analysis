import pandas as pd
import re


def headline_length_stats(data: pd.DataFrame) -> pd.Series:
    # Add a new column to the DataFrame that contains the length of each headline
    data["headline_length"] = data["headline"].apply(len)

    # Return descriptive statistics (count, mean, std, min, max, etc.) for headline lengths
    return data["headline_length"].describe()


def articles_per_publisher(data: pd.DataFrame) -> pd.Series:
    # Count the number of articles published by each publisher
    # Returns a Series where the index is the publisher name and the value is the count
    return data["publisher"].value_counts()


def articles_by_day_of_week(data: pd.DataFrame) -> pd.Series:
    # Check if the 'date' column is in a valid datetime format; raise an error if not
    if not pd.api.types.is_datetime64_any_dtype(data["date"]):
        raise ValueError("The 'date' column must be in datetime format.")

    # Extract the day of the week from the 'date' column and add it as a new column
    data["day_of_week"] = data["date"].dt.day_name()

    # Count the number of articles published on each day of the week
    return data["day_of_week"].value_counts()


def articles_by_time(data: pd.DataFrame) -> pd.Series:
    # Ensure the 'date' column is in datetime format; raise an error if it's not
    if not pd.api.types.is_datetime64_any_dtype(data["date"]):
        raise ValueError("The 'date' column must be in datetime format.")

    # Extract the time (hour and minute) from the 'date' column and add it as a new column
    data["time"] = data["date"].dt.time

    # Count the number of articles published at each time and return the counts sorted by time
    return data["time"].value_counts().sort_index()


def extract_domains(email: str) -> str:
    # Use a regular expression to extract the domain part of the email address
    # The domain is the part after the '@' symbol
    match = re.search(r"@([\w\.-]+)", email)

    # If a match is found, return the domain; otherwise, return None
    return match.group(1) if match else None


def identify_unique_domains(data: pd.DataFrame) -> pd.DataFrame:
    # Apply the extract_domains function to the 'publisher' column to extract email domains
    data["domain"] = data["publisher"].apply(extract_domains)

    # Count the number of occurrences of each domain and reset the index to create a DataFrame
    domain_counts = data["domain"].value_counts().reset_index()

    # Rename the columns of the resulting DataFrame for clarity
    domain_counts.columns = ["domain", "count"]

    # Return the DataFrame containing domains and their respective counts
    return domain_counts
