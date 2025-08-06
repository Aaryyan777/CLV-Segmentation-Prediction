import pandas as pd
import datetime as dt

print("Starting data cleaning and feature engineering...")

# Define file path
file_path = 'C:/Users/DELL/Desktop/archive/online_retail_II.csv'
output_path = 'C:/Users/DELL/rfm_data.csv'

try:
    # Load the full dataset
    df = pd.read_csv(file_path, encoding='latin1')
    print(f"Successfully loaded {len(df)} rows.")

    # --- Data Cleaning ---
    # Drop rows with missing Customer ID
    df.dropna(subset=['Customer ID'], inplace=True)
    print(f"Removed rows with missing Customer ID. Rows remaining: {len(df)}")

    # Remove returns (negative quantity)
    df = df[df['Quantity'] > 0]
    print(f"Removed returns. Rows remaining: {len(df)}")
    
    # Convert Customer ID to integer
    df['Customer ID'] = df['Customer ID'].astype(int)

    # --- Data Transformation ---
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Calculate TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # --- RFM Feature Engineering ---
    # Set a snapshot date for calculating recency. This will be the day after the last transaction.
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    print(f"Snapshot date for RFM analysis: {snapshot_date}")

    # Group by customer
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })

    # Rename columns
    rfm.rename(columns={'InvoiceDate': 'Recency',
                        'Invoice': 'Frequency',
                        'TotalPrice': 'MonetaryValue'}, inplace=True)

    print("\nRFM Features successfully created.")
    print("First 5 rows of RFM data:")
    print(rfm.head())

    # Save the RFM dataframe to a new CSV
    rfm.to_csv(output_path)
    print(f"\nSuccessfully saved cleaned RFM data to {output_path}")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
