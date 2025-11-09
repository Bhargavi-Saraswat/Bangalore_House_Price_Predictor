# --- generate_plots.py ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# File names (Must match)
file_name = 'house_prices_bangalore.csv' 

def generate_and_save_plots():
    print("Starting plot generation...")
    
    # --- Part 1: Replicate Data Cleaning for Plotting ---
    # This ensures we have the same cleaned DataFrame structure as the model
    df = pd.read_csv(file_name)
    df_cleaned = df.drop(['balcony', 'parking', 'furnishing', 'property_type', 'age'], axis='columns')
    df_cleaned.dropna(inplace=True)
    df_cleaned['price_per_sqft'] = df_cleaned['price'] / df_cleaned['area']
    df_cleaned['price_to_bhk_ratio'] = df_cleaned['price'] / df_cleaned['bhk']
    df_cleaned['has_more_than_3_bath'] = df_cleaned['bath'].apply(lambda x: 1 if x > 3 else 0)

    # Outlier removal (Must be identical to the training data prep)
    df_clean_sqft = df_cleaned[~(df_cleaned['area'] / df_cleaned['bhk'] < 300)].copy()
    
    def remove_pps_outliers(df_in):
        df_out = pd.DataFrame()
        for key, subdf in df_in.groupby('location'):
            m = subdf['price_per_sqft'].mean()
            st = subdf['price_per_sqft'].std()
            reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out
    df_final = remove_pps_outliers(df_clean_sqft)

    df_final['location'] = df_final['location'].apply(lambda x: x.strip())
    location_counts = df_final['location'].value_counts(ascending=False)
    location_less_than_10 = location_counts[location_counts <= 10]
    df_final['location'] = df_final['location'].apply(
        lambda x: 'other' if x in location_less_than_10 else x
    )

    # --- Plot 1: Price Distribution by BHK (EDA) ---
    df_final['price_lakhs'] = df_final['price'] / 100000
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='bhk', y='price_lakhs', data=df_final[df_final.bhk < 6]) 
    plt.title('Visualization 1: Price Distribution by Number of Bedrooms (BHK)')
    plt.xlabel('Number of BHK')
    plt.ylabel('Price (Lakhs INR)')
    plt.savefig('price_distribution_by_bhk.png')
    plt.close()
    print("Saved price_distribution_by_bhk.png")

    # --- Part 2: Load Model and Prepare ML Data for Residuals/Feature Importance ---
    
    # Load Model Assets
    model = joblib.load('bangalore_house_price_model.pkl')
    column_names = joblib.load('column_names.pkl')

    # Prepare ML Data (X/y for plotting purposes)
    df_final_ml = df_final.drop(['price_per_sqft', 'price_lakhs'], axis='columns')
    df_encoded = pd.get_dummies(df_final_ml, columns=['location'], drop_first=True)
    
    X_full = df_encoded.drop('price', axis='columns')
    y_full = df_encoded['price']
    
    # Align columns and split data (must match original split)
    X = X_full.reindex(columns=column_names, fill_value=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.2, random_state=42)

    # Make predictions (y_pred is needed for the residual plot)
    y_pred = model.predict(X_test)

    # --- Plot 2: Residual Plot (Model Evaluation) ---
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='red', linestyles='--')
    plt.title('Visualization 2: Residual Plot (Model Error Analysis)')
    plt.xlabel('Predicted Price (INR)')
    plt.ylabel('Residuals (Actual - Predicted Price)')
    plt.savefig('residual_plot.png')
    plt.close()
    print("Saved residual_plot.png")

    # --- Plot 3: Feature Importance (Analysis) ---
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Visualization 3: Top 10 Most Important Features for Price Prediction')
    plt.savefig('feature_importance.png')
    plt.close()
    print("Saved feature_importance.png")

if __name__ == '__main__':
    generate_and_save_plots()