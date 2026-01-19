# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# --- Load the trained model ---
with open("house_rent_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Page config ---
st.set_page_config(
    page_title="Hyderabad House Rent Prediction",
    page_icon="🏠",
    layout="centered",
)

# --- Background image and styling ---
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1560184897-6dfb12076f4a?auto=format&fit=crop&w=1470&q=80");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏠 Hyderabad House Rent Prediction")
st.markdown("Enter the details of the house and get the predicted monthly rent!")

# --- Sidebar for inputs ---
st.sidebar.header("House Features")

bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=2)
washrooms = st.sidebar.number_input("Washrooms", min_value=1, max_value=10, value=2)
area = st.sidebar.number_input("Area (sqft)", min_value=200, max_value=10000, value=1000)
furnishing = st.sidebar.selectbox("Furnishing", ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'])
tennants = st.sidebar.selectbox("Tennants", ['Family', 'Bachelor', 'Any'])
locality = st.sidebar.selectbox("Locality", [
    'Gachibowli', 'Kondapur', 'Madhapur', 'Hitech City', 'Banjara Hills', 'Others'
])

# --- Button to predict ---
if st.sidebar.button("Predict Rent"):
    # Create dataframe for prediction
    new_house = pd.DataFrame({
        'Bedrooms': [bedrooms],
        'Washrooms': [washrooms],
        'Area': [area],
        'Furnishing': [furnishing],
        'Tennants': [tennants],
        'Locality': [locality]
    })

    predicted_rent = model.predict(new_house)[0]
    st.success(f"💰 Predicted Monthly Rent: ₹{predicted_rent:,.0f}")

    # Display bubbles / fun visualization
    st.markdown("### Visualizing Features")
    bubble_df = pd.DataFrame({
        'Feature': ['Bedrooms', 'Washrooms', 'Area'],
        'Value': [bedrooms, washrooms, area]
    })

    import plotly.express as px
    fig = px.scatter(
        bubble_df, x='Feature', y='Value', size='Value', color='Feature',
        size_max=60, title="Feature Bubble Chart"
    )
    st.plotly_chart(fig)

# --- Footer / image ---
st.markdown("---")
st.markdown("Made with ❤️ by jyothikaharshini")
st.image("https://images.unsplash.com/photo-1599423300746-b62533397364?auto=format&fit=crop&w=1470&q=80", use_column_width=True)
