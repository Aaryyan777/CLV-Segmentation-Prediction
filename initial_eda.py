import pandas as pd

# Correctly formatted Windows path and added encoding
file_path = 'C:/Users/DELL/Desktop/archive/online_retail_II.csv'

try:
    # Added encoding='latin1' to handle potential character issues
    df = pd.read_csv(file_path, nrows=100000, encoding='latin1')

    print("Successfully read a sample of 100,000 rows.")
    print("\nFile Info:")
    df.info(verbose=False) # Using verbose=False for a cleaner summary

    print("\n\nFirst 5 Rows:")
    print(df.head())

    print("\n\nMissing Values:")
    print(df.isnull().sum())

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")