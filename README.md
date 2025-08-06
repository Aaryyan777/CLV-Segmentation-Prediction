# Customer Lifetime Value (CLV) Prediction Project Summary

## 1. Project Goal
The primary goal of this project was to predict Customer Lifetime Value (CLV) using historical transactional data. By understanding and predicting CLV, businesses can identify high-value customers, optimize marketing strategies, and improve customer retention.

## 2. Methodology

### Data Preparation
- Loaded transactional data and transformed it into an RFM (Recency, Frequency, Monetary) format.
- Recency: Days since last purchase.
- Frequency: Total number of purchases.
- MonetaryValue: Total spend.

### Model Training & Optimization (Phase 1)
- Initial models: Linear Regression and XGBoost.
- **Hyperparameter Tuning:** Used `RandomizedSearchCV` with `KFold` cross-validation to find optimal hyperparameters for both XGBoost and LightGBM.
    - **XGBoost Best Parameters:** `{'subsample': 0.6, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.1, 'colsample_bytree': 0.6}`
    - **LightGBM Best Parameters:** `{'subsample': 1.0, 'reg_lambda': 0.1, 'reg_alpha': 0.5, 'num_leaves': 31, 'n_estimators': 1000, 'max_depth': -1, 'learning_rate': 0.01, 'colsample_bytree': 0.6}`
- **Model Performance:**
    - **Optimized XGBoost:** RMSE: $12718.57, R-squared: 0.537
    - **Optimized LightGBM:** RMSE: $15101.28, R-squared: 0.348
- **Conclusion:** The optimized XGBoost model demonstrated superior performance and was selected as the final model.

### Customer Segmentation (Phase 3)
- Predicted CLV for all customers using the best-performing XGBoost model.
- Applied K-Means clustering on the predicted CLV to segment customers into distinct groups.
- **Segment Analysis:**
    - **Cluster 0 (Low-Value Customers):**
        - Average Recency: 203 days
        - Average Frequency: 5 purchases
        - Average Monetary Value: $2211
        - Average Predicted CLV: $2332
        - Number of Customers: 5818
        - *Characteristics:* These customers have not purchased recently, have a low frequency of purchases, and contribute the least to CLV. This is the largest segment.
    - **Cluster 1 (Medium-Value Customers):**
        - Average Recency: 26 days
        - Average Frequency: 83 purchases
        - Average Monetary Value: $50607
        - Average Predicted CLV: $51580
        - Number of Customers: 58
        - *Characteristics:* These customers have purchased relatively recently, have a moderate purchase frequency, and contribute significantly to CLV. This is a smaller, but valuable segment.
    - **Cluster 2 (High-Value Customers):**
        - Average Recency: 3.6 days
        - Average Frequency: 198 purchases
        - Average Monetary Value: $388765
        - Average Predicted CLV: $200819
        - Number of Customers: 5
        - *Characteristics:* These are the most recent, most frequent, and highest-spending customers, contributing the most to CLV. This is the smallest, most valuable segment.

## 3. Actionable Insights & Recommendations

Based on the customer segmentation, here are some actionable insights and recommendations:

### For Low-Value Customers (Cluster 0):
- **Insight:** This is the largest segment with the lowest CLV. They are likely inactive or at risk of churn.
- **Recommendations:**
    - **Re-engagement Campaigns:** Implement targeted email campaigns or promotions to re-activate these customers. Offer incentives for their next purchase.
    - **Win-back Strategies:** Analyze reasons for their inactivity (e.g., surveys, feedback forms) and address pain points.
    - **Personalized Offers:** Use their past purchase history (if available) to offer personalized recommendations to encourage repeat purchases.

### For Medium-Value Customers (Cluster 1):
- **Insight:** A valuable segment with good engagement, but potential for growth.
- **Recommendations:**
    - **Loyalty Programs:** Introduce or enhance loyalty programs to reward their continued engagement and encourage higher spending.
    - **Upselling/Cross-selling:** Recommend complementary products or higher-value items based on their purchase history.
    - **Exclusive Content/Early Access:** Offer exclusive content, early access to new products, or special discounts to make them feel valued and encourage more frequent purchases.

### For High-Value Customers (Cluster 2):
- **Insight:** The most valuable segment, crucial for business growth.
- **Recommendations:**
    - **VIP Treatment:** Provide exclusive benefits, dedicated support, or personalized communication to maintain their loyalty.
    - **Feedback & Co-creation:** Involve them in product development or feedback sessions to strengthen their connection and gather valuable insights.
    - **Referral Programs:** Encourage them to refer new customers with attractive incentives, leveraging their satisfaction.
    - **Prevent Churn:** Proactively monitor their activity for any signs of decreased engagement and intervene with personalized outreach.

## 4. Initial Development & Project History

This project began with the goal of predicting Customer Lifetime Value (CLV) to help businesses optimize their strategies. The initial development involved the following key steps:

- **Problem Definition & Data Exploration:** The problem was defined, and initial exploratory data analysis (EDA) was performed on the 'Online Retail II' dataset.
- **Feature Engineering:** Customer data was transformed using the RFM (Recency, Frequency, Monetary) framework to create relevant features for modeling.
- **Model Building & Evaluation:** Initial regression models, including Linear Regression and XGBoost, were built and evaluated. The XGBoost model was identified as superior, achieving an R-squared of 0.523 and an RMSE of $12,920.85.
- **Key Finding:** Purchase 'Frequency' was identified as the most important driver of CLV in the initial analysis.
- **Project Artifacts:** Key scripts developed in the initial phase included `initial_eda.py`, `rfm_preparation.py`, `model_training.py`, and `model_evaluation.py`. The final trained model was saved as `xgboost_clv_model.joblib`.

## 5. Future Enhancements
- Incorporate more features into the CLV model (e.g., customer demographics, product categories, marketing channel data).
- Implement more sophisticated CLV models (e.g., probabilistic models like BG/NBD and Gamma-Gamma for more nuanced behavioral insights).
- Develop a real-time CLV prediction system for immediate insights.
- Integrate with marketing automation platforms to trigger automated, personalized campaigns based on CLV segments.
