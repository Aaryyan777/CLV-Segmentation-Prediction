

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

print("Starting model building and training...")

# Load the prepared RFM data
file_path = 'C:/Users/DELL/rfm_data.csv'

try:
    rfm = pd.read_csv(file_path)
    print(f"Successfully loaded {len(rfm)} customers from {file_path}")

    # --- Target Variable Definition ---
    X = rfm[['Recency', 'Frequency']]
    y = rfm['MonetaryValue']

    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- Model 1: Linear Regression (Baseline) ---
    print("\n--- Training Baseline Model: Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    print(f"Linear Regression RMSE: ${lr_rmse:.2f}")

    # --- Model 2: XGBoost (Advanced) ---
    print("\n--- Training Advanced Model: XGBoost ---")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, 
                                 max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
    
    # Corrected the .fit() call by removing all unsupported parameters.
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                  
    xgb_preds = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"XGBoost RMSE: ${xgb_rmse:.2f}")

    print("\nModel training complete.")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

