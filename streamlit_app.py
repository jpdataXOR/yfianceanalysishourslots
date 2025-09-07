import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

st.title("🕒 Hourly % Changes Table")

# Symbols
symbols = {
    # Crypto
    "BTC-USD": "BTC-USD",

    # Forex majors
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "USDJPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/JPY": "EURJPY=X",

    # Major indices
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq 100": "^NDX",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225"
}

symbol_name = st.sidebar.selectbox("Select Instrument", list(symbols.keys()))
symbol = symbols[symbol_name]

# Dropdown for averaging period
period_option = st.sidebar.selectbox("Select Averaging Period (days)", [10, 30, 100])
days_to_load = period_option

@st.cache_data(ttl=1800)
def load_hourly(symbol, days):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end, interval="1h", progress=False)
    df.dropna(inplace=True)
    return df

df = load_hourly(symbol, days_to_load)
if df.empty:
    st.error("No data found")
    st.stop()

# Compute absolute % changes
df["PctChange"] = df["Close"].pct_change().abs() * 100
df = df.dropna()

# Extract date and hour
df["Date"] = df.index.date
df["Hour"] = df.index.hour

# Build empty table
dates = sorted(df["Date"].unique())
hourly_matrix = pd.DataFrame(0.0, index=dates, columns=range(24))

# Fill table using loop
for date in dates:
    day_data = df[df["Date"] == date]
    for hour in range(24):
        hour_data = day_data[day_data["Hour"] == hour]
        if not hour_data.empty:
            hourly_matrix.loc[date, hour] = hour_data["PctChange"].mean()

# Last N days as per dropdown
lastN = hourly_matrix.tail(days_to_load)

# Compute average per hour
hourly_avg = lastN.mean()
hourly_avg.name = "Average"

# Append average row
hourly_with_avg = pd.concat([lastN, pd.DataFrame([hourly_avg], index=["Average"])])

# Round for readability
hourly_with_avg = hourly_with_avg.applymap(lambda x: round(x, 5))

# Display table
st.subheader(f"Hourly % Changes Table — {symbol_name} ({days_to_load} days)")
st.dataframe(hourly_with_avg)

# Bar chart of averages
st.subheader(f"Bar Chart of Hourly Average % Change — {days_to_load} days")
st.bar_chart(hourly_avg)
