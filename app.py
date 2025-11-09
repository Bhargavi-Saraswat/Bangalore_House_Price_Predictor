# --- app.py (FINAL VERSION with Visualizations) ---

import streamlit as st
import numpy as np
import joblib
from PIL import Image # Library to open and display images

# --- Load the saved model and feature columns ---
try:
    model = joblib.load('bangalore_house_price_model.pkl')
    column_names = joblib.load('column_names.pkl')
except FileNotFoundError:
    st.error("Error: Model assets (PKL files) not found. Please ensure 'create_assets.py' was run successfully.")
    st.stop()

# --- Prediction Function ---
def predict_price(location, area, bhk, bath):
    # Create the zero array based on the columns the model expects
    x = np.zeros(len(column_names))
    
    # Map raw inputs to the correct indices
    x[column_names.index('area')] = area
    x[column_names.index('bhk')] = bhk
    x[column_names.index('bath')] = bath 
    
    # Map Engineered Features
    x[column_names.index('price_to_bhk_ratio')] = 3000000 / bhk 
    x[column_names.index('has_more_than_3_bath')] = 1 if bath > 3 else 0
    
    # Set the one-hot encoded location column to 1
    loc_col_name = 'location_' + location
    try:
        loc_index = column_names.index(loc_col_name)
        x[loc_index] = 1
    except ValueError:
        pass

    price_inr = model.predict([x])[0]
    return price_inr / 100000

# --- Streamlit Web Interface ---

st.set_page_config(layout="wide")
st.title("🏠 Bangalore House Price Predictor")
#st.markdown("### A Full-Stack Data Science Project")

# -----------------------------------------------
# 1. INTERACTIVE PREDICTOR SECTION (User Input)
# -----------------------------------------------

st.header("1. Live Price Estimation")

col1, col2, col3 = st.columns(3)

with col1:
    bhk = st.slider("Number of BHK:", 1, 6, 3)
    bath = st.slider("Number of Bathrooms:", 1, 6, 2)
    
with col2:
    area = st.number_input("Total Area (SqFt):", min_value=300, max_value=5000, value=1500, step=50)
    location_options = [col.replace('location_', '') for col in column_names if col.startswith('location_')]
    location_options.sort()
    location = st.selectbox("Select Location:", location_options)

with col3:
    st.text("") # Spacer
    st.text("") # Spacer
    if st.button("Estimate Price", type="primary"):
        predicted_price_lakhs = predict_price(location, area, bhk, bath)
        st.success(f"## Estimated Price: ₹{predicted_price_lakhs:.2f} Lakhs")
        
        # Confidence interval display
        rmse = 12.92 # Lakhs (RMSE from model evaluation)
        st.info(f"Price Range: ₹{predicted_price_lakhs - rmse:.2f}L - ₹{predicted_price_lakhs + rmse:.2f}L (±₹{rmse} Lakhs)")


st.markdown("---")

# -----------------------------------------------
# 2. MODEL ANALYSIS SECTION (Visualizations)
# -----------------------------------------------

st.header("2. Data Science Analysis")

tab1, tab2, tab3 = st.tabs(["Feature Importance", "Price Distribution (EDA)", "Model Performance (Residuals)"])

with tab1:
    st.subheader("Top Drivers of House Price")
    st.markdown("This chart shows which factors (features) the model found most influential in determining the final price.")
    try:
        image = Image.open('feature_importance.png')
        st.image(image, caption='Visualization 3: Feature Importance')
    except FileNotFoundError:
        st.warning("Feature Importance graph not found. Please ensure all previous steps ran successfully.")

with tab2:
    st.subheader("Price vs. Number of Bedrooms")
    st.markdown("A look at how price varies as the size of the house (BHK) increases, validating the core relationship in the dataset.")
    try:
        image = Image.open('price_distribution_by_bhk.png')
        st.image(image, caption='Visualization 1: Price Distribution by BHK')
    except FileNotFoundError:
        st.warning("Price Distribution graph not found. Please ensure all previous steps ran successfully.")

with tab3:
    st.subheader("Model Error Analysis")
    st.markdown("The Residual Plot confirms model quality: errors are randomly scattered around zero, meaning the model is not systematically biased.")
    try:
        image = Image.open('residual_plot.png')
        st.image(image, caption='Visualization 2: Residual Plot')
    except FileNotFoundError:
        st.warning("Residual Plot graph not found. Please ensure all previous steps ran successfully.")

st.markdown("---")
st.success(f"**Final Model Metrics:** R-squared: 0.8987 | RMSE: ₹12.92 Lakhs")