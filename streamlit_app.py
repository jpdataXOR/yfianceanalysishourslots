import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np

st.title("🕒 Local-Time Hourly Avg % Change — Positive / Negative / Overall")

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
    start = end - timedelta(days=days + 3)  # small padding
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
def extract_close_series(df: pd.DataFrame, symbol_key: str):
    # If DataFrame has MultiIndex columns like ('Close','BTC-USD'), try to pull level 0 == 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        if "Close" in lvl0:
            close_df = df.xs("Close", axis=1, level=0)
            if isinstance(close_df, pd.Series):
                return close_df
            else:
                if symbol_key in close_df.columns:
                    return close_df[symbol_key]
                else:
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

# --- Extract close series and prepare processing df ---
try:
    close_series = extract_close_series(raw_df, symbol)
except KeyError as e:
    st.error(f"Could not find a Close column: {e}")
    st.stop()

st.write(f"Using Close column: `{getattr(close_series.name, '__str__', lambda: str(close_series.name))()}`")

# Build single-column dataframe
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
    # prepare empty placeholders
    avg_pos = pd.Series([0.0]*24, index=range(24))
    avg_neg = pd.Series([0.0]*24, index=range(24))
    avg_all = pd.Series([0.0]*24, index=range(24))
else:
    # --- Build UTC 23:00 close series (one value per UTC day at 23:00) ---
    utc_23_mask = df.index.hour == 23
    utc_close_23_series = df.loc[utc_23_mask, "Close"].dropna().sort_index()

    if utc_close_23_series.empty:
        st.warning("No UTC 23:00 close values found in the data. Positive/negative day split cannot be computed.")
        day_type_dict = {}
    else:
        day_comp = (utc_close_23_series.diff() > 0).dropna()
        day_type_dict = {}
        for idx, val in zip(day_comp.index, day_comp.values):
            ts = pd.to_datetime(idx)
            if getattr(ts, "tz", None) is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            day_type_dict[ts.date()] = bool(val)

    # --- Convert to selected local timezone for aggregation (all data) ---
    df_local_all = df.copy()
    try:
        df_local_all.index = df_local_all.index.tz_convert(timezone_option)
    except Exception:
        df_local_all.index = pd.to_datetime(df_local_all.index).tz_localize("UTC").tz_convert(timezone_option)

    # Add local-date and local-hour columns
    df_local_all["DateLocal"] = df_local_all.index.date
    df_local_all["HourLocal"] = df_local_all.index.hour

    # --- Select last N local dates available (from df_local_all) ---
    unique_local_dates = sorted(pd.unique(df_local_all["DateLocal"]))
    selected_dates = unique_local_dates[-days_to_load:] if len(unique_local_dates) >= days_to_load else unique_local_dates

    if not selected_dates:
        st.warning("No local dates found in the data for the selected period.")
        # empty placeholders
        avg_pos = pd.Series([0.0]*24, index=range(24))
        avg_neg = pd.Series([0.0]*24, index=range(24))
        avg_all = pd.Series([0.0]*24, index=range(24))
    else:
        # Filter df_local_all to the selected dates
        df_local_selected = df_local_all[df_local_all["DateLocal"].isin(selected_dates)].copy()

        # --- Map DayType to selected local rows (safe map) ---
        df_local_selected["DayType"] = df_local_selected["DateLocal"].map(lambda d: day_type_dict.get(d, np.nan))

        # Build DataFrame with DayType present (for pos/neg split)
        df_local_with_type = df_local_selected.dropna(subset=["DayType"]) if "DayType" in df_local_selected.columns else df_local_selected.iloc[0:0]

        # --- Positive and Negative pivots ---
        if not df_local_with_type.empty:
            df_pos = df_local_with_type[df_local_with_type["DayType"] == True]
            df_neg = df_local_with_type[df_local_with_type["DayType"] == False]
        else:
            df_pos = pd.DataFrame(columns=df_local_selected.columns)
            df_neg = pd.DataFrame(columns=df_local_selected.columns)

        # Pivot positive days
        if not df_pos.empty:
            hourly_pos = df_pos.pivot_table(index="DateLocal", columns="HourLocal", values="PctChange", aggfunc="mean", fill_value=0)
            avg_pos = hourly_pos.mean().round(5)
        else:
            hourly_pos = pd.DataFrame(columns=range(24))
            avg_pos = pd.Series([0.0]*24, index=range(24))

        # Pivot negative days
        if not df_neg.empty:
            hourly_neg = df_neg.pivot_table(index="DateLocal", columns="HourLocal", values="PctChange", aggfunc="mean", fill_value=0)
            avg_neg = hourly_neg.mean().round(5)
        else:
            hourly_neg = pd.DataFrame(columns=range(24))
            avg_neg = pd.Series([0.0]*24, index=range(24))

        # --- Overall pivot (ALL selected days, regardless of DayType) ---
        if not df_local_selected.empty:
            hourly_all = df_local_selected.pivot_table(index="DateLocal", columns="HourLocal", values="PctChange", aggfunc="mean", fill_value=0)
            avg_all = hourly_all.mean().round(5)
        else:
            hourly_all = pd.DataFrame(columns=range(24))
            avg_all = pd.Series([0.0]*24, index=range(24))

# --- Display results (local-time charts only) ---
st.markdown(f"**Instrument:** {symbol_name} — Local timezone: **{timezone_option}** — Period: **{days_to_load} days**")

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.subheader("Positive-Day Hourly Avg % Change (local time)")
    st.bar_chart(avg_pos)

with col2:
    st.subheader("Negative-Day Hourly Avg % Change (local time)")
    st.bar_chart(avg_neg)

with col3:
    st.subheader("Overall Hourly Avg % Change (local time)")
    st.bar_chart(avg_all)

# --- Show numeric tables under charts for inspection ---
with st.expander("Show Positive-day hourly table (signed % change)"):
    if 'hourly_pos' not in locals() or hourly_pos.empty:
        st.write("No positive days found in selected period (or insufficient UTC 23:00 data).")
    else:
        df_show_pos = hourly_pos.copy()
        df_show_pos.index = df_show_pos.index.astype(str)
        st.dataframe(df_show_pos.round(5))

with st.expander("Show Negative-day hourly table (signed % change)"):
    if 'hourly_neg' not in locals() or hourly_neg.empty:
        st.write("No negative days found in selected period (or insufficient UTC 23:00 data).")
    else:
        df_show_neg = hourly_neg.copy()
        df_show_neg.index = df_show_neg.index.astype(str)
        st.dataframe(df_show_neg.round(5))

with st.expander("Show Overall hourly table (signed % change, all selected days)"):
    if 'hourly_all' not in locals() or hourly_all.empty:
        st.write("No overall data available for selected dates.")
    else:
        df_show_all = hourly_all.copy()
        df_show_all.index = df_show_all.index.astype(str)
        st.dataframe(df_show_all.round(5))
