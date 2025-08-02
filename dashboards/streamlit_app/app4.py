import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Verdana&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Verdana', sans-serif;
        background-color: #e6f0f7;  /* very faint sea blue */
        color: #000000;
    }
    .sidebar .sidebar-content {
        background-color: #000000;
        color: white;
        font-weight: bold;
        font-size: 22px;
    }
    .stButton>button {
        background-color: #004080;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .title {
        font-weight: bold;
        font-size: 28px;
        color: #00264d;
    }
    .subheader {
        font-weight: 600;
        font-size: 20px;
        color: #003366;
        margin-bottom: 10px;
    }
    .box {
        padding: 15px;
        background-color: #cce6ff;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #99ccff;
    }
    footer {
        font-size: 12px;
        font-weight: bold;
        color: #666666;
        margin-top: 30px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load model and features with caching
@st.cache_data
def load_model_and_features():
    try:
        model = joblib.load("fraud_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except Exception as e:
        st.error(f"Error loading model or features: {e}")
        return None, None

model, features = load_model_and_features()

# Load dataset for exploration
@st.cache_data
def load_data():
    df = pd.read_csv("processed_fraud_data_single.csv")
    return df

df = load_data()

# Sidebar menu
menu = st.sidebar.radio("Menu", options=["Home", "Data Exploration", "About"])

# Sidebar style override for menu items
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content label[for="menu"] {
        font-weight: 700;
        font-size: 22px;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True
)

def prediction_page():
    st.markdown('<div class="title">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Enter transaction details below to predict fraud:</div>', unsafe_allow_html=True)
    
    with st.form(key='fraud_form'):
        with st.container():
            st.markdown('<div class="box">', unsafe_allow_html=True)
            amt = st.number_input("Amount", min_value=0.0, step=0.01, format="%.2f")
            currency = st.selectbox("Currency", ["USD", "INR", "EUR", "GBP", "Other"])
            
            # Convert amount to USD for simplicity (dummy conversion)
            conversion_rates = {"USD":1, "INR":0.013, "EUR":1.08, "GBP":1.25, "Other":1}
            amt_usd = amt * conversion_rates[currency]
            
            city_pop = st.number_input("City Population (scaled)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
            age = st.number_input("Age", min_value=10, max_value=120, step=1)
            trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
            trans_dayofweek = st.slider("Transaction Day of Week (1=Sun,7=Sat)", 1, 7, 3)
            trans_month = st.slider("Transaction Month (1-12)", 1, 12, 6)
            gender_index = st.selectbox("Gender", options=[0.0, 1.0], format_func=lambda x: "Female" if x == 0.0 else "Male")
            category_index = st.selectbox("Category Index", options=sorted(df['category_index'].dropna().unique()))
            state_index = st.selectbox("State Index", options=sorted(df['state_index'].dropna().unique()))
            distance = st.number_input("Distance (miles)", min_value=0.0, step=0.01, format="%.2f")
            st.markdown('</div>', unsafe_allow_html=True)
        
        submit = st.form_submit_button("Predict Fraud")
    
    if submit:
        if not model:
            st.error("Model not loaded. Cannot predict.")
            return
        
        input_dict = {
            "amt": amt_usd,
            "city_pop": city_pop,
            "age": age,
            "trans_hour": trans_hour,
            "trans_dayofweek": trans_dayofweek,
            "trans_month": trans_month,
            "gender_index": gender_index,
            "category_index": category_index,
            "state_index": state_index,
            "distance": distance
        }
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df[features])[0]
        
        if prediction == 1:
            st.error("🚨 Fraud Detected!")
        else:
            st.success("✅ Transaction appears genuine.")

def data_exploration_page():
    st.markdown('<div class="title">📊 Data Exploration</div>', unsafe_allow_html=True)
    
    # Show basic data stats
    st.markdown('<div class="subheader">Dataset Overview</div>', unsafe_allow_html=True)
    st.write(df.head(10))
    st.write(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    
    # Bar chart: Fraud vs Non-fraud counts
    st.markdown('<div class="subheader">Fraud vs Non-Fraud Transactions</div>', unsafe_allow_html=True)
    fraud_counts = df['is_fraud'].value_counts().rename({0: "Non-Fraud", 1: "Fraud"})
    fig_bar = px.bar(fraud_counts, x=fraud_counts.index, y=fraud_counts.values,
                     labels={'x': 'Transaction Type', 'y': 'Count'}, color=fraud_counts.index,
                     color_discrete_map={"Non-Fraud": "blue", "Fraud": "red"})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Heatmap: correlation
    st.markdown('<div class="subheader">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    # Histogram: transaction amount distribution
    st.markdown('<div class="subheader">Transaction Amount Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(df, x="amt", nbins=50, title="Transaction Amount Distribution", color="is_fraud",
                            color_discrete_map={0:"blue",1:"red"})
    st.plotly_chart(fig_hist, use_container_width=True)

def about_page():
    st.markdown('<div class="title">ℹ️ About This App</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This **Credit Card Fraud Detection** app allows users to input transaction details and get real-time fraud predictions.  
        It includes an interactive data exploration page to understand dataset characteristics visually.

        <br>
        <b>Features:</b>
        - Real-time fraud prediction using pre-trained ML model  
        - Data exploration with bar charts, heatmaps, and histograms  
        - Clean and intuitive UI with dark sidebar and soft sea blue background  
        <br>
        <b>Developed by:</b> MegharaniPol
        """, unsafe_allow_html=True
    )

# Main control flow
if menu == "Home":
    prediction_page()
elif menu == "Data Exploration":
    data_exploration_page()
else:
    about_page()

# Footer
st.markdown(
    """
    <footer>
    This UI is developed by <b>MegharaniPol</b>
    </footer>
    """,
    unsafe_allow_html=True
)
