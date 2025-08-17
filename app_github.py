import pandas as pd
import joblib
from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np

# --- Path Setup ---
# Get the absolute path of the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path for the templates folder (it's in a subdirectory)
template_dir = os.path.join(base_dir, 'templates')

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=template_dir)

# --- Load Model and Data ---
# Construct absolute paths to your data files relative to this script
model_path = os.path.join(base_dir, 'xgboost_clv_model.joblib')
rfm_path = os.path.join(base_dir, 'rfm_data.csv')

# Load the model and data
xgb_model = joblib.load(model_path)
rfm = pd.read_csv(rfm_path)

# --- Initial Analysis (Run once at startup) ---
# Predict CLV for all customers and perform segmentation
rfm['PredictedCLV_log'] = xgb_model.predict(rfm[['Recency_log', 'Frequency_log']])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm[['PredictedCLV_log']])

# --- Web Routes ---
@app.route('/')
def index():
    # --- Segment Analysis for Display ---
    segment_analysis = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean',
        'PredictedCLV_log': 'mean',
        'Customer ID': 'count'
    }).rename(columns={'CustomerID': 'NumCustomers'}).reset_index()

    # Convert predicted CLV to a readable dollar amount
    segment_analysis['PredictedCLV'] = np.expm1(segment_analysis['PredictedCLV_log'])

    # Format currency columns
    segment_analysis['MonetaryValue'] = segment_analysis['MonetaryValue'].map('${:,.2f}'.format)
    segment_analysis['PredictedCLV'] = segment_analysis['PredictedCLV'].map('${:,.2f}'.format)

    # Rename columns for clarity and drop the log column for display
    segment_analysis.rename(columns={'Recency': 'Recency (days)'}, inplace=True)
    segment_analysis.drop(columns=['PredictedCLV_log'], inplace=True)

    # Convert DataFrame to HTML
    segment_analysis_html = segment_analysis.to_html(classes='table table-striped', index=False)

    return render_template('index.html', segment_analysis=segment_analysis_html)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    recency = int(request.form['recency'])
    frequency = int(request.form['frequency'])
    
    # Log transform the input for the model
    recency_log = np.log1p(recency)
    frequency_log = np.log1p(frequency)

    # Create a DataFrame for the new customer
    new_customer = pd.DataFrame({'Recency_log': [recency_log], 'Frequency_log': [frequency_log]})
    
    # Predict CLV and convert back to dollar amount
    predicted_clv_log = xgb_model.predict(new_customer)[0]
    predicted_clv = np.expm1(predicted_clv_log)
    
    # Re-run the main index function to get the segment analysis again
    # This is simpler than re-calculating everything separately
    return index(prediction_text=f'Predicted CLV: ${predicted_clv:,.2f}')

@app.route('/customer_segments.png')
def customer_segments_png():
    # Serve the image file from the base directory
    return send_from_directory(base_dir, 'customer_segments.png')

# --- Main Execution ---
if __name__ == '__main__':
    # The host='0.0.0.0' makes it accessible on your local network
    app.run(debug=True, host='0.0.0.0')
