import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pytz

st.title("🕒 Hourly % Changes Table")

# --- Symbols: Crypto, Forex majors, indices ---
symbols = {
    "BTC-USD": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "USDJPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq 100": "^NDX",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225"
}

# --- Sidebar selections ---
symbol_name = st.sidebar.selectbox("Select Instrument", list(symbols.keys()))
symbol = symbols[symbol_name]

period_option = st.sidebar.selectbox("Select Averaging Period (days)", [10, 30, 100])
days_to_load = period_option

timezone_option = st.sidebar.selectbox("Select Your Timezone", pytz.all_timezones, index=pytz.all_timezones.index('UTC'))

# --- Load hourly data ---
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

# --- Compute absolute % changes ---
df["PctChange"] = df["Close"].pct_change().abs() * 100
df = df.dropna()

# --- Extract date and hour in UTC ---
df["Date"] = df.index.date
df["Hour"] = df.index.hour

# --- Build hourly matrix UTC ---
dates = sorted(df["Date"].unique())
hourly_matrix = pd.DataFrame(0.0, index=dates, columns=range(24))

for date in dates:
    day_data = df[df["Date"] == date]
    for hour in range(24):
        hour_data = day_data[day_data["Hour"] == hour]
        if not hour_data.empty:
            hourly_matrix.loc[date, hour] = hour_data["PctChange"].mean()

# Last N days in UTC
lastN = hourly_matrix.tail(days_to_load)

# Compute average per hour UTC
hourly_avg = lastN.mean()
hourly_avg.name = "Average"

# Append average row
hourly_with_avg = pd.concat([lastN, pd.DataFrame([hourly_avg], index=["Average"])])

# Round values
hourly_with_avg = hourly_with_avg.applymap(lambda x: round(x, 5))

# Ensure numeric columns
hourly_with_avg = hourly_with_avg.astype(float, errors='ignore')

# Convert index to string for Streamlit
hourly_with_avg_display = hourly_with_avg.copy()
hourly_with_avg_display.index = hourly_with_avg_display.index.astype(str)

# --- Display table ---
st.subheader(f"Hourly % Changes Table — {symbol_name} ({days_to_load} days, UTC)")
st.dataframe(hourly_with_avg_display)

# --- Bar chart of averages UTC ---
st.subheader(f"Bar Chart of Hourly Average % Change — UTC ({days_to_load} days)")
st.bar_chart(hourly_avg)

# --- Local timezone conversion ---
df_local = df.copy()

# Only convert timezone; the index is already tz-aware
df_local.index = df_local.index.tz_convert(timezone_option)
df_local["DateLocal"] = df_local.index.date
df_local["HourLocal"] = df_local.index.hour

# Build hourly matrix in local time
dates_local = sorted(df_local["DateLocal"].unique())
hourly_matrix_local = pd.DataFrame(0.0, index=dates_local, columns=range(24))

for date in dates_local:
    day_data = df_local[df_local["DateLocal"] == date]
    for hour in range(24):
        hour_data = day_data[day_data["HourLocal"] == hour]
        if not hour_data.empty:
            hourly_matrix_local.loc[date, hour] = hour_data["PctChange"].mean()

# Last N days in local time
lastN_local = hourly_matrix_local.tail(days_to_load)

# Compute average per hour local
hourly_avg_local = lastN_local.mean()
hourly_avg_local.name = "Average"

# --- Bar chart local time ---
st.subheader(f"Bar Chart of Hourly Average % Change — Local Time ({timezone_option}, {days_to_load} days)")
st.bar_chart(hourly_avg_local)