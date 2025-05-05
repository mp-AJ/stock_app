# stock_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Get list of popular Indian stocks
STOCK_LIST = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "HUL": "HINDUNILVR.NS",
    "SBI": "SBIN.NS",
    "ONGC": "ONGC.NS",
    "Maruti": "MARUTI.NS"
}

st.title("📈 Indian Stock Price Predictor")
st.markdown("Predict next-day stock price using recent trends (basic AI model).")

# User selects stock
selected_stock = st.selectbox("Choose a stock:", list(STOCK_LIST.keys()))
symbol = STOCK_LIST[selected_stock]

# Fetch data
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

df = load_data(symbol)

if df.empty or 'Close' not in df.columns or df['Close'].isna().all():
    st.error("⚠️ Failed to fetch valid stock data. This might be due to an invalid ticker or no recent data.")
else:
    st.success(f"Loaded {len(df)} records for {selected_stock}")
    
    last_valid_close = df['Close'].dropna().iloc[-1]
    st.metric("Latest Close Price", f"₹{last_valid_close:.2f}")

    # Prepare data
    df['Day'] = np.arange(len(df))
    X = df[['Day']]
    y = df['Close']

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # Predict next 7 days
    future_days = 7
    future_X = pd.DataFrame({'Day': np.arange(len(df), len(df)+future_days)})
    future_preds = model.predict(future_X)

    st.subheader("📅 Next 7-Day Price Prediction")
    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(future_days)]
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': future_preds
    })
    st.dataframe(pred_df.set_index('Date').style.format("₹{:.2f}"))

    # Optional: Plot chart
    st.subheader("📊 Recent vs Predicted Prices")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(df.index, y, label='Historical')
    ax.plot(future_dates, future_preds, label='Predicted', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)
