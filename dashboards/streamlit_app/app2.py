import streamlit as st
import pandas as pd
import joblib

# Load model and features
try:
    model = joblib.load("fraud_model.pkl")
    features = joblib.load("model_features.pkl")
except Exception as e:
    st.error(f"Error loading model or features: {e}")
    st.stop()

# Mapping for categorical indices (replace with your actual labels)
gender_map = {0.0: "Female", 1.0: "Male"}
category_map = {
    0.0: "Food", 1.0: "Travel", 2.0: "Shopping", 3.0: "Utilities", 4.0: "Others"
}
state_map = {
    0.0: "CA", 1.0: "TX", 2.0: "NY", 3.0: "FL", 4.0: "IL"  # example states
}

# Style the page background and fonts using markdown & CSS
page_style = """
<style>
    body {
        background-color: #0e1117;
        color: #bbe1fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #1b262c;
        color: #bbe1fa;
        border-radius: 8px;
        height: 40px;
        width: 150px;
        font-weight: bold;
        font-size: 16px;
    }
    .stSidebar {
        background-color: #1b262c;
        color: #bbe1fa;
    }
    hr {
        border: 1px solid #bbe1fa;
    }
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Title and subtitle
st.title("💳 Credit Card Fraud Detection")
st.markdown("### Predict fraudulent transactions using advanced machine learning")
st.markdown("---")

# Sidebar menu
menu = st.sidebar.radio("Navigation", ["Home", "About"])

if menu == "Home":
    st.header("Transaction Input")

    # Inputs
    amt = st.number_input("Amount ($)", min_value=0.0, step=0.01, format="%.2f")
    city_pop = st.number_input("City Population (in thousands)", min_value=0, step=1)
    age = st.number_input("Age of Cardholder", min_value=1, step=1)
    trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
    trans_dayofweek = st.selectbox("Day of Week", 
                                  options=[1,2,3,4,5,6,7], 
                                  format_func=lambda x: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][x-1])
    trans_month = st.selectbox("Transaction Month", options=list(range(1,13)))
    gender = st.selectbox("Gender", options=list(gender_map.values()))
    category = st.selectbox("Category", options=list(category_map.values()))
    state = st.selectbox("State", options=list(state_map.values()))
    distance = st.number_input("Distance (miles)", min_value=0.0, step=0.01, format="%.2f")

    # Map inputs to indices
    gender_index = list(gender_map.keys())[list(gender_map.values()).index(gender)]
    category_index = list(category_map.keys())[list(category_map.values()).index(category)]
    state_index = list(state_map.keys())[list(state_map.values()).index(state)]

    if st.button("Predict Fraud"):
        input_dict = {
            'amt': amt,
            'city_pop': city_pop,
            'age': age,
            'trans_hour': trans_hour,
            'trans_dayofweek': trans_dayofweek,
            'trans_month': trans_month,
            'gender_index': gender_index,
            'category_index': category_index,
            'state_index': state_index,
            'distance': distance
        }

        input_df = pd.DataFrame([input_dict], columns=features)

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(f"🚨 Fraud Detected! Probability: {proba:.2%}")
        else:
            st.success(f"✅ Transaction is Legitimate. Probability of Fraud: {proba:.2%}")

elif menu == "About":
    st.header("About This App")
    st.markdown("""
    This credit card fraud detection app is built using **Streamlit** and trained machine learning models.
    
    - Detects fraudulent transactions based on multiple features.
    - Uses a pre-trained model saved in pickle format.
    - Designed with a sleek dark theme with blue accents.
    - Developed by **MegharaniPol**.
    """)
    st.markdown("---")

# Footer
st.markdown(
    """
    <div style='text-align:center; font-size:12px; color:#7aa7cc; padding-top:10px;'>
        This UI is developed by MegharaniPol
    </div>
    """, unsafe_allow_html=True
)
