import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Titanium Analytics", layout="wide", page_icon="📊")
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #2c3e50; color: white; width: 100%; border-radius: 5px;}
    h1, h2, h3 {color: #2c3e50;}
    .metric-container {background-color: white; padding: 10px; border-radius: 5px; box-shadow: 1px 1px 5px #ccc;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_assets():
    assets = {}
    path = 'models'
    
    
    try:
        assets['lr'] = joblib.load(f'{path}/stock_lr_model.pkl')
        assets['stock_scaler'] = joblib.load(f'{path}/stock_scaler.pkl')
        
        assets['lstm'] = load_model(f'{path}/stock_lstm_model.h5')
    except: pass

    try:
        assets['nb'] = joblib.load(f'{path}/news_sentiment_model.pkl')
        assets['tfidf'] = joblib.load(f'{path}/news_vectorizer.pkl')
        assets['le'] = joblib.load(f'{path}/news_label_encoder.pkl')
    except: pass

    try:
        assets['kmeans'] = joblib.load(f'{path}/fund_kmeans_model.pkl')
        assets['fund_scaler'] = joblib.load(f'{path}/fund_scaler.pkl')
    except: pass

    return assets

assets = load_all_assets()


st.sidebar.title("TITANIUM ANALYTICS")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Dashboard", "Stock Prediction", "News Sentiment", "Company Classifier"])
st.sidebar.markdown("---")
st.sidebar.caption("v2.0 | Advanced Analytics Mode")

if page == "Dashboard":
    st.title("Executive Dashboard")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Stock Model", "Active" if 'lr' in assets else "Offline")
    c2.metric("NLP Engine", "Active" if 'tfidf' in assets else "Offline")
    c3.metric("Clustering", "Active" if 'kmeans' in assets else "Offline")

    st.subheader("Data Overview")
    if os.path.exists('data/stock_data.csv'):
        df = pd.read_csv('data/stock_data.csv')
        st.line_chart(df['Close/Last'].astype(str).str.replace('$','').astype(float))
    else:
        st.warning("Stock Data not found.")

elif page == "Stock Prediction":
    st.title("Stock Price Forecasting")
    st.markdown("Uses **Linear Regression** and **LSTM** models.")
    
    col1, col2 = st.columns(2)
    with col1:
        open_p = st.number_input("Open Price", 100.0)
        high_p = st.number_input("High Price", 105.0)
        low_p = st.number_input("Low Price", 95.0)
    with col2:
        vol = st.number_input("Volume", 1000000)
        
        sma_10 = st.number_input("SMA 10 (Optional)", 100.0)
        sma_50 = st.number_input("SMA 50 (Optional)", 98.0)
        rsi = st.number_input("RSI (Optional)", 50.0)

    if st.button("Predict Close Price"):
        if 'lr' in assets and 'stock_scaler' in assets:
            try:
                
                ema_20 = sma_10 
                macd = 0.5 
                
                input_data = np.array([[open_p, high_p, low_p, vol, sma_10, sma_50, ema_20, rsi, macd]])
                input_scaled = assets['stock_scaler'].transform(input_data)
                
                prediction = assets['lr'].predict(input_scaled)[0]
                st.success(f"Linear Regression Forecast: **${prediction:.2f}**")
                
            except ValueError as e:
                st.error(f"Shape Mismatch. Notebook trained on more features than App provides. Error: {e}")
        else:
            st.error("Models not loaded.")


elif page == "News Sentiment":
    st.title("Financial News Analyzer")
    text = st.text_area("Enter Headline:", "Stocks rallied today on strong earnings.")
    
    if st.button("Analyze Sentiment"):
        if 'tfidf' in assets and 'nb' in assets:
            vec = assets['tfidf'].transform([text]).toarray()
            pred = assets['nb'].predict(vec)[0]
            if 'le' in assets:
                label = assets['le'].inverse_transform([pred])[0]
            else:
                label = pred
            
            st.info(f"Predicted Sentiment: **{label.upper()}**")
            
            
            probs = assets['nb'].predict_proba(vec)
            st.bar_chart(pd.DataFrame(probs.T, index=assets['le'].classes_))
        else:
            st.error("NLP Models Missing.")


elif page == "Company Classifier":
    st.title("Company Fundamentals Clustering")
    
    pe = st.number_input("P/E Ratio", 15.0)
    eps = st.number_input("Earnings Per Share", 2.0)
    mcap = st.number_input("Market Cap (Billions)", 10.0) * 1e9
    price = st.number_input("Stock Price", 50.0)
    ebitda = st.number_input("EBITDA", 1000000.0)
    
    if st.button("Determine Cluster"):
        if 'kmeans' in assets and 'fund_scaler' in assets:
            
            arr = np.array([[price, pe, eps, mcap, ebitda]])
            scaled = assets['fund_scaler'].transform(arr)
            cluster = assets['kmeans'].predict(scaled)[0]
            
            st.success(f"Cluster Group: **{cluster}**")
            if cluster == 0: st.caption("Characteristics: Blue Chip / Stable")
            elif cluster == 1: st.caption("Characteristics: High Growth")
            else: st.caption("Characteristics: Undervalued")
        else:
            st.error("Models Missing.")