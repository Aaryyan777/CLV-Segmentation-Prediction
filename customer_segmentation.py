import pandas as pd
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting customer segmentation...")

# Load the prepared RFM data
file_path = 'C:/Users/DELL/rfm_data.csv'
model_path = 'C:/Users/DELL/xgboost_clv_model.joblib'

try:
    rfm = pd.read_csv(file_path)
    print(f"Successfully loaded {len(rfm)} customers from {file_path}")

    # Load the trained XGBoost model
    xgb_model = joblib.load(model_path)
    print("Successfully loaded the trained XGBoost model.")

    # Predict CLV for all customers
    rfm['PredictedCLV'] = xgb_model.predict(rfm[['Recency', 'Frequency']])

    # --- Customer Segmentation using K-Means ---
    # Determine optimal number of clusters using Elbow Method
    sse = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm[['PredictedCLV']])
        sse[k] = kmeans.inertia_

    # Plot Elbow Method (optional, for visualization/analysis)
    plt.figure(figsize=(10, 6))
    plt.plot(list(sse.keys()), list(sse.values()), marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('SSE')
    plt.savefig('C:/Users/DELL/elbow_method.png')
    print("Elbow method plot saved to elbow_method.png")

    # Based on visual inspection of elbow method, let's assume 3 clusters for now (can be adjusted)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm[['PredictedCLV']])

    # Analyze segments
    segment_analysis = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean',
        'PredictedCLV': 'mean',
        'Customer ID': 'count'
    }).rename(columns={'CustomerID': 'NumCustomers'})

    print("\n--- Customer Segment Analysis ---")
    print(segment_analysis)

    # Optional: Visualize segments
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Frequency', y='PredictedCLV', hue='Cluster', data=rfm, palette='viridis', s=100)
    plt.title('Customer Segments by Frequency and Predicted CLV')
    plt.xlabel('Frequency')
    plt.ylabel('Predicted CLV')
    plt.savefig('C:/Users/DELL/customer_segments.png')
    print("Customer segments plot saved to customer_segments.png")

    print("Customer segmentation complete.")

except FileNotFoundError:
    print(f"Error: Make sure 'rfm_data.csv' and 'xgboost_clv_model.joblib' are in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")
