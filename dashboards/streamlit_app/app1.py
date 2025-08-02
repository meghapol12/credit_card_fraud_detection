import streamlit as st
import pandas as pd
import joblib

st.title("💳 Credit Card Fraud Detection")

# Load model and features
try:
    model = joblib.load("fraud_model.pkl")
    features = joblib.load("model_features.pkl")
except Exception as e:
    st.error(f"Error loading model or features: {e}")
    st.stop()

# Get input from user
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Transaction is Legitimate")
