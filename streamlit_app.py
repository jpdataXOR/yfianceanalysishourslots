import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np

st.set_page_config(layout="wide")
st.title("🕒 Local-Time Hourly Avg % Change — Positive / Negative / Overall")

# ------------------------------------------------------------
# Symbols (dropdown)
# ------------------------------------------------------------
symbols = {
    "BTC-USD": "BTC-USD",

    # Forex
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "USDJPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X",
    "USD/CAD": "USDCAD=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/NZD": "AUDNZD=X",
    "NZD/JPY": "NZDJPY=X",
    "USD/NOK": "USDNOK=X",
    "USD/SEK": "USDSEK=X",

    # Indices
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq 100": "^NDX",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225",
    "ASX 200": "^AXJO",
    "S&P/TSX": "^GSPTSE",
    "Hang Seng": "^HSI",
    "VIX": "^VIX",

    # Commodities
    "Gold (GC)": "GC=F",
    "Silver (SI)": "SI=F",
    "WTI Crude (CL)": "CL=F",
    "Brent Crude (BZ)": "BZ=F",
    "Natural Gas (NG)": "NG=F",
    "Copper (HG)": "HG=F",
    "Platinum (PL)": "PL=F",
    "Palladium (PA)": "PA=F",
}

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
symbol_name_dropdown = st.sidebar.selectbox(
    "Select Instrument (dropdown)",
    list(symbols.keys())
)

st.sidebar.markdown("---")

custom_symbol = st.sidebar.text_input(
    "Or type any Yahoo Finance symbol",
    placeholder="e.g. AAPL, GC=F, ^SSEC, EURUSD=X"
)

days_to_load = st.sidebar.selectbox("Select Averaging Period (days)", [10, 30, 100])

timezone_option = st.sidebar.selectbox(
    "Select Your Timezone",
    pytz.all_timezones,
    index=pytz.all_timezones.index("Australia/Sydney")
)

# ------------------------------------------------------------
# Final symbol resolution
# ------------------------------------------------------------
if custom_symbol.strip():
    symbol_name = custom_symbol.strip()
    symbol = custom_symbol.strip()
    st.sidebar.caption(f"Using custom symbol: {symbol}")
else:
    symbol_name = symbol_name_dropdown
    symbol = symbols[symbol_name]

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
@st.cache_data(ttl=1800)
def load_hourly(symbol: str, days: int) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=days + 3)
    try:
        df = yf.download(symbol, start=start, end=end, interval="1h", progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna(how="all")


def extract_close_series(df: pd.DataFrame):
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            return df.xs("Close", axis=1, level=0).iloc[:, 0]
    if "Close" in df.columns:
        return df["Close"]
    return df.iloc[:, 0]

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["Charts", "Best Hour"])

# ============================================================
# TAB 1 — CHARTS
# ============================================================
with tab1:
    raw_df = load_hourly(symbol, days_to_load)

    if raw_df.empty:
        st.error("No data returned for this symbol.")
        st.stop()

    close_series = extract_close_series(raw_df)
    df = pd.DataFrame({"Close": close_series.astype(float)})

    # Normalize to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df["PctChange"] = df["Close"].pct_change() * 100
    df.dropna(inplace=True)

    # --------------------------------------------------------
    # Day classification (robust fallback)
    # --------------------------------------------------------
    df_utc = df.copy()
    df_utc["DateUTC"] = df_utc.index.date
    day_type_dict = {}

    utc_23 = df_utc[df_utc.index.hour == 23]

    if not utc_23.empty:
        daily_close = utc_23.groupby("DateUTC")["Close"].last()
    else:
        daily_close = df_utc.groupby("DateUTC")["Close"].last()

    if not daily_close.empty:
        day_comp = (daily_close.diff() > 0).dropna()
        day_type_dict = {d: bool(v) for d, v in day_comp.items()}

    # --------------------------------------------------------
    # Convert to local time
    # --------------------------------------------------------
    df_local = df.copy()
    df_local.index = df_local.index.tz_convert(timezone_option)
    df_local["DateLocal"] = df_local.index.date
    df_local["HourLocal"] = df_local.index.hour
    df_local["DayType"] = df_local["DateLocal"].map(day_type_dict)

    df_pos = df_local[df_local["DayType"] == True]
    df_neg = df_local[df_local["DayType"] == False]

    def avg_by_hour(dfx):
        if dfx.empty:
            return pd.Series([0.0]*24, index=range(24))
        pivot = dfx.pivot_table(
            index="DateLocal",
            columns="HourLocal",
            values="PctChange",
            aggfunc="mean"
        )
        return pivot.mean().reindex(range(24), fill_value=0).round(5)

    avg_pos = avg_by_hour(df_pos)
    avg_neg = avg_by_hour(df_neg)
    avg_all = avg_by_hour(df_local)

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------
    st.subheader(f"{symbol_name} — {timezone_option}")

    st.markdown("### Positive days")
    st.bar_chart(avg_pos)

    st.markdown("### Negative days")
    st.bar_chart(avg_neg)

    st.markdown("### Overall")
    st.bar_chart(avg_all)

# ============================================================
# TAB 2 — BEST HOUR SUMMARY
# ============================================================
with tab2:
    rows = []

    for name, sym in symbols.items():
        raw = load_hourly(sym, days_to_load)
        if raw.empty:
            rows.append({"Instrument": name})
            continue

        close = extract_close_series(raw)
        df = pd.DataFrame({"Close": close.astype(float)})

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df["PctChange"] = df["Close"].pct_change() * 100
        df.dropna(inplace=True)

        if df.empty:
            rows.append({"Instrument": name})
            continue

        df.index = df.index.tz_convert(timezone_option)
        df["HourLocal"] = df.index.hour
        df["DateLocal"] = df.index.date

        pivot = df.pivot_table(
            index="DateLocal",
            columns="HourLocal",
            values="PctChange",
            aggfunc="mean"
        )

        avg = pivot.mean()
        if avg.isna().all():
            rows.append({"Instrument": name})
            continue

        best_hour = int(avg.abs().idxmax())

        rows.append({
            "Instrument": name,
            "BestHourInt": best_hour,
            "BestHour": f"{best_hour:02d}:00",
            "BestHour_Avg%": round(avg.loc[best_hour], 5)
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("BestHourInt", na_position="last")

    st.subheader("Best Hour per Instrument (dropdown symbols only)")
    st.dataframe(summary_df, use_container_width=True)