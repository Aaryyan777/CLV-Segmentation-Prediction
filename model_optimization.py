import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import joblib

print("Starting model optimization and comparison...")

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

    # --- XGBoost Model Optimization ---
    print("\n--- Optimizing XGBoost Model ---")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_param_dist = {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_dist,
                                           n_iter=50, scoring='neg_mean_squared_error', cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                           verbose=1, n_jobs=-1, random_state=42)
    xgb_random_search.fit(X_train, y_train)
    best_xgb_model = xgb_random_search.best_estimator_
    print(f"Best XGBoost parameters: {xgb_random_search.best_params_}")
    xgb_preds = best_xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_r2 = r2_score(y_test, xgb_preds)
    print(f"Optimized XGBoost RMSE: ${xgb_rmse:.2f}")
    print(f"Optimized XGBoost R-squared: {xgb_r2:.3f}")

    # --- LightGBM Model Optimization ---
    print("\n--- Optimizing LightGBM Model ---")
    lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)
    lgb_param_dist = {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40],
        'max_depth': [3, 5, 7, -1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    lgb_random_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=lgb_param_dist,
                                           n_iter=10, scoring='neg_mean_squared_error', cv=KFold(n_splits=3, shuffle=True, random_state=42),
                                           verbose=1, n_jobs=-1, random_state=42)
    lgb_random_search.fit(X_train, y_train)
    best_lgb_model = lgb_random_search.best_estimator_
    print(f"Best LightGBM parameters: {lgb_random_search.best_params_}")
    lgb_preds = best_lgb_model.predict(X_test)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    lgb_r2 = r2_score(y_test, lgb_preds)
    print(f"Optimized LightGBM RMSE: ${lgb_rmse:.2f}")
    print(f"Optimized LightGBM R-squared: {lgb_r2:.3f}")

    # --- Compare and Save Best Model ---
    print("\n--- Comparing Models ---")
    if xgb_rmse < lgb_rmse:
        print("XGBoost performed better. Saving XGBoost model.")
        best_model = best_xgb_model
        model_name = "xgboost_clv_model.joblib"
    else:
        print("LightGBM performed better. Saving LightGBM model.")
        best_model = best_lgb_model
        model_name = "lightgbm_clv_model.joblib"

    model_output_path = f'C:/Users/DELL/{model_name}'
    joblib.dump(best_model, model_output_path)
    print(f"Successfully saved the best trained model to {model_output_path}")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
