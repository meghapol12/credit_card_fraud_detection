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

# Mapping for categorical indices (update with your actual labels)
gender_map = {0.0: "Female", 1.0: "Male"}
category_map = {
    0.0: "Food", 1.0: "Travel", 2.0: "Shopping", 3.0: "Utilities", 4.0: "Others"
}
state_map = {
    0.0: "CA", 1.0: "TX", 2.0: "NY", 3.0: "FL", 4.0: "IL"
}

# --- CSS Styling ---
page_style = """
<style>
    /* Page background */
    .main {
        background-color: #f0f9fb !important;  /* very faint sea blue */
        font-family: Verdana, Geneva, Tahoma, sans-serif;
        color: #0b3c5d;  /* dark blue text */
    }

    /* Sidebar background and font */
    .css-1d391kg {
        background-color: #000000 !important;  /* pure black */
    }
    .css-1d391kg .css-1v0mbdj {
        color: white !important;
        font-weight: 700 !important;
        font-size: 22px !important;
        font-family: Verdana, Geneva, Tahoma, sans-serif !important;
    }

    /* Main headings and labels */
    h1, h2, h3, h4, label {
        font-family: Verdana, Geneva, Tahoma, sans-serif !important;
        font-weight: 600 !important;
        font-size: 20px !important;
        color: #0b3c5d !important;
    }

    /* Input boxes styling */
    div[data-baseweb="input"] > div {
        border: 2px solid #1a73e8 !important; /* bright blue border */
        border-radius: 6px !important;
        background-color: #ffffff !important;
    }

    /* Buttons styling */
    button[kind="primary"] {
        background-color: #1a73e8 !important; /* blue */
        color: white !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-family: Verdana, Geneva, Tahoma, sans-serif !important;
    }

    /* Dropdown menus font size */
    div[role="listbox"] {
        font-family: Verdana, Geneva, Tahoma, sans-serif !important;
        font-size: 18px !important;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        font-size: 13px;
        font-weight: 600;
        color: #0b3c5d;
        font-family: Verdana, Geneva, Tahoma, sans-serif;
        margin-top: 30px;
    }
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Title and subtitle
st.title("💳 Credit Card Fraud Detection")
st.markdown("### Predict fraudulent transactions using advanced machine learning")
st.markdown("---")

# Sidebar menu with big white bold text
menu = st.sidebar.radio("Navigation", ["Home", "About"])

if menu == "Home":
    st.header("Transaction Input")

    # Amount input with USD$ prefix (custom)
    amt = st.number_input(
        "Amount ($)", min_value=0.0, step=0.01, format="%.2f", help="Transaction amount in USD"
    )
    city_pop = st.number_input(
        "City Population (in thousands)", min_value=0, step=1, help="Population of city where transaction occurred"
    )
    age = st.number_input(
        "Age of Cardholder", min_value=1, step=1, help="Age of cardholder"
    )
    trans_hour = st.slider(
        "Transaction Hour (0-23)", 0, 23, 12, help="Hour of the transaction"
    )
    trans_dayofweek = st.selectbox(
        "Day of Week",
        options=[1, 2, 3, 4, 5, 6, 7],
        format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x - 1],
        help="Day when transaction occurred"
    )
    trans_month = st.selectbox(
        "Transaction Month",
        options=list(range(1, 13)),
        help="Month when transaction occurred"
    )
    gender = st.selectbox("Gender", options=list(gender_map.values()), help="Cardholder gender")
    category = st.selectbox("Category", options=list(category_map.values()), help="Transaction category")
    state = st.selectbox("State", options=list(state_map.values()), help="State of transaction")
    distance = st.number_input(
        "Distance (miles)", min_value=0.0, step=0.01, format="%.2f", help="Distance between merchant and cardholder"
    )

    # Map to indices
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
    - Designed with a sleek dark sidebar and subtle sea blue page background.
    - Developed by **MegharaniPol**.
    """)
    st.markdown("---")

# Footer
st.markdown(
    """
    <div class="footer">
        This UI is developed by MegharaniPol
    </div>
    """,
    unsafe_allow_html=True,
)
