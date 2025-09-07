import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np

st.title("🕒 Local-Time Hourly Avg % Change — Positive vs Negative Days (robust)")

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
days_to_load = int(period_option)

timezone_option = st.sidebar.selectbox(
    "Select Your Timezone", pytz.all_timezones, index=pytz.all_timezones.index("UTC")
)

# --- Load hourly data (defensive) ---
@st.cache_data(ttl=1800)
def load_hourly(symbol: str, days: int) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=days + 3)  # small padding for diffs/timezones
    try:
        df = yf.download(symbol, start=start, end=end, interval="1h", progress=False)
    except Exception as e:
        st.error(f"yfinance download error: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(how="all")
    return df

raw_df = load_hourly(symbol, days_to_load)
if raw_df.empty:
    st.error("No data returned from Yahoo Finance for this symbol / period.")
    st.stop()

st.write(f"Raw data rows: {len(raw_df)} — columns: {list(raw_df.columns[:8])} {'...' if len(raw_df.columns)>8 else ''}")

# --- Helper: extract a single-close series from possibly MultiIndex dataframe ---
def extract_close_series(df: pd.DataFrame):
    # If DataFrame has MultiIndex columns like ('Close','BTC-USD'), try to pull level 0 == 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        if "Close" in lvl0:
            close_df = df.xs("Close", axis=1, level=0)  # may be DataFrame with tickers as columns
            # If close_df is a Series (single), return it, otherwise pick first column that matches symbol if present
            if isinstance(close_df, pd.Series):
                return close_df
            else:
                # try to pick the matching ticker column
                # columns could be e.g. 'BTC-USD' or '^GSPC' etc.
                if symbol in close_df.columns:
                    return close_df[symbol]
                else:
                    # fallback: first column
                    return close_df.iloc[:, 0]
        else:
            # try to find a column tuple that mentions 'Close' in level 0
            for col in df.columns:
                if isinstance(col, tuple) and "Close" in col[0]:
                    return df[col]
            # as last resort, pick any numeric column that looks like a close by name
            for col in df.columns:
                if "close" in str(col).lower():
                    return df[col]
            raise KeyError("No Close column found in MultiIndex columns.")
    else:
        # single-level columns
        if "Close" in df.columns:
            return df["Close"]
        for col in df.columns:
            if "close" in str(col).lower():
                return df[col]
        raise KeyError("No Close column found in dataframe columns.")

# --- Extract close series and prepare a single-column dataframe for processing ---
try:
    close_series = extract_close_series(raw_df)
except KeyError as e:
    st.error(f"Could not find a Close column: {e}")
    st.stop()

# Make sure the extracted series has the expected name (for messaging)
st.write(f"Using Close column: `{getattr(close_series.name, '__str__', lambda: str(close_series.name))()}`")

# Build processing dataframe (single 'Close' column)
df = pd.DataFrame({"Close": close_series.astype(float)})

# --- Standardize index to UTC timezone ---
if getattr(df.index, "tz", None) is None:
    try:
        df.index = df.index.tz_localize("UTC")
    except Exception:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
else:
    try:
        df.index = df.index.tz_convert("UTC")
    except Exception:
        df.index = pd.to_datetime(df.index).tz_convert("UTC")

# --- Compute signed % changes safely ---
df["PctChange"] = df["Close"].pct_change() * 100
df = df.dropna(subset=["PctChange"])
if df.empty:
    st.warning("No rows remain after computing pct changes; cannot compute charts.")
    # show empty placeholders
    avg_pos = pd.Series([0.0]*24, index=range(24))
    avg_neg = pd.Series([0.0]*24, index=range(24))
else:
    # --- Build UTC 23:00 close series (one value per UTC day at 23:00) ---
    utc_23_mask = df.index.hour == 23
    utc_close_23_series = df.loc[utc_23_mask, "Close"].dropna().sort_index()

    if utc_close_23_series.empty:
        st.warning("No UTC 23:00 close values found in the data. Positive/negative day split cannot be computed.")
        day_type_dict = {}
    else:
        # Compare today's 23:00 to yesterday's 23:00
        day_comp = (utc_close_23_series.diff() > 0).dropna()
        # build dict keyed by datetime.date (UTC date)
        day_type_dict = {}
        for idx, val in zip(day_comp.index, day_comp.values):
            ts = pd.to_datetime(idx)
            if getattr(ts, "tz", None) is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            day_type_dict[ts.date()] = bool(val)

    # --- Convert df to selected local timezone for hourly aggregation ---
    df_local = df.copy()
    try:
        df_local.index = df_local.index.tz_convert(timezone_option)
    except Exception:
        df_local.index = pd.to_datetime(df_local.index).tz_localize("UTC").tz_convert(timezone_option)

    # Add local-date and local-hour columns
    df_local["DateLocal"] = df_local.index.date
    df_local["HourLocal"] = df_local.index.hour

    # --- Map DayType (positive/negative) to local dates safely ---
    df_local["DayType"] = df_local["DateLocal"].map(lambda d: day_type_dict.get(d, np.nan))

    # Drop rows where DayType couldn't be determined
    df_local = df_local.dropna(subset=["DayType"]) if "DayType" in df_local.columns else df_local.iloc[0:0]

    # Select last N local dates that actually exist
    unique_local_dates = sorted(pd.unique(df_local["DateLocal"])) if not df_local.empty else []
    selected_dates = unique_local_dates[-days_to_load:] if len(unique_local_dates) >= days_to_load else unique_local_dates
    if selected_dates:
        df_local = df_local[df_local["DateLocal"].isin(selected_dates)]

    # --- Build hourly matrices (signed pct change) for pos/neg days ---
    dates_local_sorted = sorted(pd.unique(df_local["DateLocal"])) if not df_local.empty else []
    hourly_matrix_pos = pd.DataFrame(0.0, index=dates_local_sorted, columns=range(24))
    hourly_matrix_neg = pd.DataFrame(0.0, index=dates_local_sorted, columns=range(24))

    for date in dates_local_sorted:
        day_df = df_local[df_local["DateLocal"] == date]
        if day_df.empty:
            continue
        is_positive_day = bool(day_df["DayType"].iloc[0])
        for hour in range(24):
            hour_df = day_df[day_df["HourLocal"] == hour]
            if not hour_df.empty:
                val = hour_df["PctChange"].mean()
                if is_positive_day:
                    hourly_matrix_pos.loc[date, hour] = val
                else:
                    hourly_matrix_neg.loc[date, hour] = val

    # Remove rows that are all zeros (no data)
    if not hourly_matrix_pos.empty:
        hourly_matrix_pos = hourly_matrix_pos.loc[hourly_matrix_pos.abs().sum(axis=1) != 0]
    if not hourly_matrix_neg.empty:
        hourly_matrix_neg = hourly_matrix_neg.loc[hourly_matrix_neg.abs().sum(axis=1) != 0]

    # --- Compute average per hour across chosen days (signed average, not absolute) ---
    avg_pos = hourly_matrix_pos.mean() if not hourly_matrix_pos.empty else pd.Series([0.0] * 24, index=range(24))
    avg_neg = hourly_matrix_neg.mean() if not hourly_matrix_neg.empty else pd.Series([0.0] * 24, index=range(24))

    avg_pos = avg_pos.round(5)
    avg_neg = avg_neg.round(5)

# --- Display results (local-time charts only) ---
st.markdown(f"**Instrument:** {symbol_name} — Local timezone: **{timezone_option}** — Period: **{days_to_load} days**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Positive-Day Hourly Avg % Change (local time)")
    st.bar_chart(avg_pos)

with col2:
    st.subheader("Negative-Day Hourly Avg % Change (local time)")
    st.bar_chart(avg_neg)

# --- Show numeric tables under charts for inspection ---
with st.expander("Show Positive-day hourly table (signed % change)"):
    if 'hourly_matrix_pos' not in locals() or hourly_matrix_pos.empty:
        st.write("No positive days found in selected period (or insufficient UTC 23:00 data).")
    else:
        df_show_pos = hourly_matrix_pos.copy()
        df_show_pos.index = df_show_pos.index.astype(str)
        st.dataframe(df_show_pos.round(5))

with st.expander("Show Negative-day hourly table (signed % change)"):
    if 'hourly_matrix_neg' not in locals() or hourly_matrix_neg.empty:
        st.write("No negative days found in selected period (or insufficient UTC 23:00 data).")
    else:
        df_show_neg = hourly_matrix_neg.copy()
        df_show_neg.index = df_show_neg.index.astype(str)
        st.dataframe(df_show_neg.round(5))