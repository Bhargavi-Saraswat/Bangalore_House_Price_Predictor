import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("whitegrid")
file_name = 'house_prices_bangalore.csv'

# Load the data
df = pd.read_csv(file_name)

# 1. Drop non-core categorical features for a clean regression model
# We keep core features: area, location, bhk, bath, price
df_cleaned = df.drop(['balcony', 'parking', 'furnishing', 'property_type', 'age'], axis='columns')

# 2. Handle missing values: Drop rows with any remaining missing data
df_cleaned.dropna(inplace=True)

print(f"Dataset loaded with {len(df_cleaned)} valid rows.")
print("\nCleaned Data Head:")
print(df_cleaned.head())
# Create 'price_per_sqft' for Outlier Detection (price is in absolute INR, area in sqft)
df_cleaned['price_per_sqft'] = df_cleaned['price'] / df_cleaned['area']

# --- Enhanced Features ---
# 1. price_to_bhk_ratio: Captures the efficiency of pricing relative to room count.
df_cleaned['price_to_bhk_ratio'] = df_cleaned['price'] / df_cleaned['bhk']

# 2. has_more_than_3_bath: Binary feature to capture the luxury segment.
df_cleaned['has_more_than_3_bath'] = df_cleaned['bath'].apply(lambda x: 1 if x > 3 else 0)

print("\nEngineered Features Added.")# Outlier Rule 1: Remove properties where area per BHK is unusually low (< 300 sqft per room)
df_clean_sqft = df_cleaned[~(df_cleaned['area'] / df_cleaned['bhk'] < 300)].copy()

# Outlier Rule 2: Remove price_per_sqft outliers (1-standard deviation method)
def remove_pps_outliers(df_in):
    df_out = pd.DataFrame()
    for key, subdf in df_in.groupby('location'):
        m = subdf['price_per_sqft'].mean()
        st = subdf['price_per_sqft'].std()
        # Keep data points that are within one standard deviation (1-sigma)
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df_final = remove_pps_outliers(df_clean_sqft)

# Location Grouping: Group rare locations into 'other'
df_final['location'] = df_final['location'].apply(lambda x: x.strip())
location_counts = df_final['location'].value_counts(ascending=False)
location_less_than_10 = location_counts[location_counts <= 10]
df_final['location'] = df_final['location'].apply(
    lambda x: 'other' if x in location_less_than_10 else x
)

print(f"Final dataset size after cleaning: {len(df_final)} rows.")
print(f"Number of unique locations after grouping: {len(df_final['location'].unique())}")

# Save the cleaned data for visualization steps
df_final.to_csv('df_final_cleaned_project.csv', index=False)
df_final['price_lakhs'] = df_final['price'] / 100000

plt.figure(figsize=(10, 6))
# Limit BHK to < 6 for a clearer boxplot view
sns.boxplot(x='bhk', y='price_lakhs', data=df_final[df_final.bhk < 6]) 
plt.title('Visualization 1: Price Distribution by Number of Bedrooms (BHK)')
plt.xlabel('Number of BHK')
plt.ylabel('Price (Lakhs INR)')
plt.show()
# This plot visually validates that price generally increases with BHK count.# Drop columns not needed for the ML model
df_final_ml = df_final.drop(['price_per_sqft', 'price_lakhs'], axis='columns')

# Convert the 'location' column using One-Hot Encoding
df_encoded = pd.get_dummies(df_final_ml, columns=['location'], drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop('price', axis='columns')
y = df_encoded['price'] # Target: price in absolute INR

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)

print("\nStarting model training...")
model.fit(X_train, y_train)
print("Model training complete!")

# Make predictions
y_pred = model.predict(X_test)# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
rmse_lakhs = rmse / 100000

print(f"\n--- Model Performance Metrics ---")
print(f"RMSE (Average Error in Price): ₹{rmse_lakhs:.2f} Lakhs")
print(f"R-squared (Model Accuracy): {r2:.4f}")
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='red', linestyles='--')
plt.title('Visualization 2: Residual Plot (Model Error Analysis)')
plt.xlabel('Predicted Price (INR)')
plt.ylabel('Residuals (Actual - Predicted Price)')
plt.show()importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Visualization 3: Top 10 Most Important Features for Price Prediction')
plt.show()

# Save the feature importance DataFrame for the next step
feature_importance_df.to_csv('feature_importance_results.csv', index=False)# Prediction function (adapted to your final feature set)
def predict_price_demo(location, area, bhk, bath, model, X):    
    x = np.zeros(len(X.columns))
    
    # 1. Assign known numerical features
    x[X.columns.get_loc('area')] = area
    x[X.columns.get_loc('bhk')] = bhk
    x[X.columns.get_loc('bath')] = bath 
    
    # 2. Assign Engineered Features based on standard inputs
    x[X.columns.get_loc('price_to_bhk_ratio')] = 10000000 / bhk # Assuming an initial price of 1 Crore for the ratio
    x[X.columns.get_loc('has_more_than_3_bath')] = 1 if bath > 3 else 0
    
    # 3. Set the one-hot encoded location column to 1
    loc_col_name = 'location_' + location
    if loc_col_name in X.columns:
        x[X.columns.get_loc(loc_col_name)] = 1
    
    # Predict the price and convert to Lakhs
    return model.predict([x])[0] / 100000

# --- Comparative Scenario ---
area_standard = 1500
bhk_standard = 3
bath_standard = 3

# High-Demand Premium Zone (e.g., Koramangala - assumed to exist after grouping)
loc_A = 'Koramangala' 
price_A = predict_price_demo(loc_A, area_standard, bhk_standard, bath_standard, model, X)

# Developing Value Zone (e.g., Kengeri - assumed to exist after grouping)
loc_B = 'Kengeri' 
price_B = predict_price_demo(loc_B, area_standard, bhk_standard, bath_standard, model, X)

savings = price_A - price_B
percentage_savings = (savings / price_A) * 100

print(f"\n--- BUYER SUGGESTIONS: FINDING VALUE ---")
print(f"Scenario: Comparing a {bhk_standard} BHK, {area_standard} SqFt property (with {bath_standard} baths).")
print("-" * 60)
print(f"Predicted Price in HIGH-DEMAND ({loc_A}):\t ₹{price_A:.2f} Lakhs")
print(f"Predicted Price in VALUE-ZONE ({loc_B}):\t ₹{price_B:.2f} Lakhs")
print("-" * 60)

if savings > 0:
    print(f"**Key Suggestion:** Moving to {loc_B} for the same property saves the buyer **₹{savings:.2f} Lakhs**, a **{percentage_savings:.1f}% discount**.")
    print("\n**Actionable Advice:** Use the model to target areas with high predicted value but lower 'location premium' overhead.")
else:
    print(f"The model suggests that for this property size, the price difference between {loc_A} and {loc_B} is negligible.")