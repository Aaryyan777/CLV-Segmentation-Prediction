

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

print("Starting model evaluation...")

# Load the prepared RFM data
file_path = 'C:/Users/DELL/rfm_data.csv'
model_output_path = 'C:/Users/DELL/xgboost_clv_model.joblib'

try:
    rfm = pd.read_csv(file_path)

    # --- Prepare Data ---
    X = rfm[['Recency', 'Frequency']]
    y = rfm['MonetaryValue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Re-train Models ---
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, 
                                 max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_preds = xgb_model.predict(X_test)

    # --- Evaluation Metrics ---
    print("\n--- Model Performance Metrics ---")
    # R-squared
    lr_r2 = r2_score(y_test, lr_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)
    print(f"Linear Regression R-squared: {lr_r2:.3f}")
    print(f"XGBoost R-squared: {xgb_r2:.3f}")

    # RMSE (for confirmation)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"Linear Regression RMSE: ${lr_rmse:.2f}")
    print(f"XGBoost RMSE: ${xgb_rmse:.2f}")

    # --- Feature Importance ---
    print("\n--- Feature Importance (from XGBoost) ---")
    feature_importances = pd.DataFrame(xgb_model.feature_importances_,
                                       index = X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)

    # --- Save the Final Model ---
    joblib.dump(xgb_model, model_output_path)
    print(f"\nSuccessfully saved the trained XGBoost model to {model_output_path}")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

