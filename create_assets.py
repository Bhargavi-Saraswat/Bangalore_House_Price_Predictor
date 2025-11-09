# --- create_assets.py ---

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# NOTE: The provided file name has a space, use quotes!
file_name = 'house_prices_bangalore.csv' 

def create_and_save_model():
    print("Starting data processing and model training...")
    # 1. Load and Clean Data
    df = pd.read_csv(file_name)
    df_cleaned = df.drop(['balcony', 'parking', 'furnishing', 'property_type', 'age'], axis='columns')
    df_cleaned.dropna(inplace=True)

    # 2. Feature Engineering
    df_cleaned['price_per_sqft'] = df_cleaned['price'] / df_cleaned['area']
    df_cleaned['price_to_bhk_ratio'] = df_cleaned['price'] / df_cleaned['bhk']
    df_cleaned['has_more_than_3_bath'] = df_cleaned['bath'].apply(lambda x: 1 if x > 3 else 0)

    # 3. Outlier Removal 
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

    # Location Grouping
    df_final['location'] = df_final['location'].apply(lambda x: x.strip())
    location_counts = df_final['location'].value_counts(ascending=False)
    location_less_than_10 = location_counts[location_counts <= 10]
    df_final['location'] = df_final['location'].apply(
        lambda x: 'other' if x in location_less_than_10 else x
    )

    # 4. ML Data Prep
    df_final_ml = df_final.drop(['price_per_sqft'], axis='columns')
    df_encoded = pd.get_dummies(df_final_ml, columns=['location'], drop_first=True)
    
    X = df_encoded.drop('price', axis='columns')
    y = df_encoded['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train and Save Model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    model.fit(X_train, y_train)

    # Save the trained model and feature names
    joblib.dump(model, 'bangalore_house_price_model.pkl')
    joblib.dump(X.columns.tolist(), 'column_names.pkl')
    
    print("\nSUCCESS: Model assets saved to bangalore_house_price_model.pkl and column_names.pkl")

if __name__ == '__main__':
    create_and_save_model()