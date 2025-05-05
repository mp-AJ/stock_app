# stock_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ---------------------------
# 🔽 STOCK LIST (NSE Symbols)
# ---------------------------
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

# ---------------------------
# 🎯 App UI
# ---------------------------
st.set_page_config(page_title="Indian Stock Predictor", layout="centered")
st.title("📈 Indian Stock Price Predictor")
st.markdown("Predict the next 7 days of stock prices using recent trends.")

selected_stock = st.selectbox("Choose a stock:", list(STOCK_LIST.keys()))
symbol = STOCK_LIST[selected_stock]

# ---------------------------
# 📥 Fetch Data
# ---------------------------
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

@st.cache_data
def load_data(symbol):
    return yf.download(symbol, start=start_date, end=end_date)

df = load_data(symbol)

# ---------------------------
# ✅ Validate Data
# ---------------------------
if df is None or df.empty:
    st.error("⚠️ No data found for this stock. Try another.")
elif 'Close' not in df.columns:
    st.error("⚠️ 'Close' column is missing. Cannot predict.")
else:
    valid_closes = df['Close'].dropna()

    if valid_closes.empty:
        st.error("⚠️ No valid closing price data found.")
    else:
        st.success(f"✅ Loaded {len(df)} records for {selected_stock} from {df.index.min().date()} to {df.index.max().date()}")

        last_valid_close = valid_closes.iloc[-1]

        if pd.isna(last_valid_close):
            st.warning("⚠️ Last closing price is not a number (NaN).")
        else:
            st.metric("Latest Close Price", f"₹{last_valid_close:.2f}")

        # ---------------------------
        # 🧠 Model Training
        # ---------------------------
        df = df.dropna()
        df['Day'] = np.arange(len(df))  # Turn dates into integer days
        X = df[['Day']]
        y = df['Close']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predict next 7 days
        future_days = 7
        future_X = pd.DataFrame({'Day': np.arange(len(df), len(df)+future_days)})
        future_preds = model.predict(future_X)

        future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(future_days)]
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close Price': future_preds
        })

        st.subheader("📅 Next 7-Day Prediction")
        st.dataframe(pred_df.set_index('Date').style.format("₹{:.2f}"))

        # ---------------------------
        # 📊 Plot Chart
        # ---------------------------
        st.subheader("📉 Price Chart: Past + Prediction")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label='Historical')
        ax.plot(future_dates, future_preds, label='Predicted', linestyle='--', marker='o')
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (₹)")
        ax.set_title(f"{selected_stock} - Price Forecast")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
