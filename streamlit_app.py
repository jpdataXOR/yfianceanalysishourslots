import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np

st.set_page_config(layout="wide")
st.title("🕒 Local-Time Hourly Avg % Change — Positive / Negative / Overall")

# --- Default Symbols: Crypto, Forex majors, Nordic, NZD, crosses, indices, commodities ---
symbols = {
    "BTC-USD": "BTC-USD",
    # Forex majors & common crosses
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
    # Nordic pairs (USD base)
    "USD/NOK": "USDNOK=X",
    "USD/SEK": "USDSEK=X",
    # Major indices
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
    # Commodities (Yahoo Finance futures tickers)
    "Gold (GC)": "GC=F",
    "Silver (SI)": "SI=F",
    "WTI Crude (CL)": "CL=F",
    "Brent Crude (BZ)": "BZ=F",
    "Natural Gas (NG)": "NG=F",
    "Copper (HG)": "HG=F",
    "Platinum (PL)": "PL=F",
    "Palladium (PA)": "PA=F",
}

# --- Sidebar selections ---
symbol_name = st.sidebar.selectbox("Select Instrument (charts)", list(symbols.keys()), index=0)
symbol = symbols[symbol_name]

period_option = st.sidebar.selectbox("Select Averaging Period (days)", [10, 30, 100], index=0)
days_to_load = int(period_option)

# default timezone set to Australia/Sydney
timezone_option = st.sidebar.selectbox(
    "Select Your Timezone",
    pytz.all_timezones,
    index=pytz.all_timezones.index("Australia/Sydney")
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Add custom Yahoo Finance symbol(s)**")
st.sidebar.markdown("You can add new symbols to analyze. Provide either `Label=SYMBOL` or just the `SYMBOL`. For multiple entries separate with commas. Example: `MyGold=GC=F, BTC-TEST=BTC-USD, EURUSD=X`.")
custom_input = st.sidebar.text_input("Custom symbols (comma-separated)", value="")

# Parse custom input and append to symbols dict (do not overwrite existing keys; add with suffix if necessary)
if custom_input:
    entries = [e.strip() for e in custom_input.split(",") if e.strip()]
    for ent in entries:
        if "=" in ent and ent.count("=") >= 1 and not ent.upper().endswith("=F"):
            # allow user to enter Label=SYMBOL form OR symbol that contains '=' like GC=F
            # We'll split on the first '=' to allow symbol values that themselves have '=' (e.g. GC=F)
            parts = ent.split("=", 1)
            label = parts[0].strip()
            symbol_val = parts[1].strip()
            # if there was more (like GC=F) we already handled by splitting only first '='
        elif "=" in ent and ent.upper().endswith("=F"):
            # if user provides only a Yahoo style symbol like GC=F we treat label same as symbol
            label = ent
            symbol_val = ent
        else:
            # no explicit label, assume the token is the symbol and use it also as label
            label = ent
            symbol_val = ent

        # ensure uniqueness of label in symbols dict
        orig_label = label
        i = 1
        while label in symbols:
            label = f"{orig_label} (custom {i})"
            i += 1
        symbols[label] = symbol_val

st.sidebar.markdown("---")
st.sidebar.caption("Pro-tip: For index/commodity tickers use Yahoo " "tickers like GC=F (gold futures), CL=F (WTI crude) and ^GSPC (S&P 500).")

# --- Load hourly data (defensive) ---
@st.cache_data(ttl=1800)
def load_hourly(symbol: str, days: int) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=days + 3)  # small padding for diffs/timezones
    try:
        df = yf.download(symbol, start=start, end=end, interval="1h", progress=False)
    except Exception as e:
        # return empty df on failure
        st.error(f"yfinance download error for {symbol}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(how="all")
    return df

# --- Helper: extract a single-close series from possibly MultiIndex dataframe ---
def extract_close_series(df: pd.DataFrame, symbol_key: str):
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
            # fallback: look for any column name containing 'close'
            for col in df.columns:
                if isinstance(col, tuple) and "Close" in col[0]:
                    return df[col]
            for col in df.columns:
                if "close" in str(col).lower():
                    return df[col]
            raise KeyError("No Close column found in MultiIndex columns.")
    else:
        if "Close" in df.columns:
            return df["Close"]
        for col in df.columns:
            if "close" in str(col).lower():
                return df[col]
        # If there's only one column (like some ticker responses) use it
        if len(df.columns) == 1:
            return df.iloc[:, 0]
        raise KeyError("No Close column found in dataframe columns.")

# --- Tabs: charts and best-hour summary ---
tab1, tab2 = st.tabs(["Charts", "Best Hour"])

# -------------------------
# Tab 1: charts (unchanged stacked charts)
# -------------------------
with tab1:
    raw_df = load_hourly(symbol, days_to_load)
    if raw_df.empty:
        st.error("No data returned from Yahoo Finance for this symbol / period.")
    else:
        st.write(f"Raw data rows: {len(raw_df)} — columns: {list(raw_df.columns[:8])} {'...' if len(raw_df.columns)>8 else ''}")
        try:
            close_series = extract_close_series(raw_df, symbol)
        except KeyError as e:
            st.error(f"Could not find a Close column: {e}")
            st.stop()

        st.write(f"Using Close column: `{getattr(close_series.name, '__str__', lambda: str(close_series.name))()}`")

        # single-column df for processing
        df = pd.DataFrame({"Close": close_series.astype(float)})

        # standardize index to UTC
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

        # pct changes (signed)
        df["PctChange"] = df["Close"].pct_change() * 100
        df = df.dropna(subset=["PctChange"])
        if df.empty:
            st.warning("No rows remain after computing pct changes; cannot compute charts.")
            avg_pos = pd.Series([0.0]*24, index=range(24))
            avg_neg = pd.Series([0.0]*24, index=range(24))
            avg_all = pd.Series([0.0]*24, index=range(24))
        else:
            # UTC 23:00 close series -> day_type dict keyed by UTC date
            utc_23_mask = df.index.hour == 23
            utc_close_23_series = df.loc[utc_23_mask, "Close"].dropna().sort_index()
            if utc_close_23_series.empty:
                st.warning("No UTC 23:00 close values found. Positive/negative split cannot be computed.")
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

            # local conversion for all data
            df_local_all = df.copy()
            try:
                df_local_all.index = df_local_all.index.tz_convert(timezone_option)
            except Exception:
                df_local_all.index = pd.to_datetime(df_local_all.index).tz_localize("UTC").tz_convert(timezone_option)

            df_local_all["DateLocal"] = df_local_all.index.date
            df_local_all["HourLocal"] = df_local_all.index.hour

            # pick last N local dates
            unique_local_dates = sorted(pd.unique(df_local_all["DateLocal"]))
            selected_dates = unique_local_dates[-days_to_load:] if len(unique_local_dates) >= days_to_load else unique_local_dates

            if not selected_dates:
                st.warning("No local dates found in the data for the selected period.")
                avg_pos = pd.Series([0.0]*24, index=range(24))
                avg_neg = pd.Series([0.0]*24, index=range(24))
                avg_all = pd.Series([0.0]*24, index=range(24))
            else:
                df_local_selected = df_local_all[df_local_all["DateLocal"].isin(selected_dates)].copy()
                df_local_selected["DayType"] = df_local_selected["DateLocal"].map(lambda d: day_type_dict.get(d, np.nan))
                df_local_with_type = df_local_selected.dropna(subset=["DayType"]) if "DayType" in df_local_selected.columns else df_local_selected.iloc[0:0]

                if not df_local_with_type.empty:
                    df_pos = df_local_with_type[df_local_with_type["DayType"] == True]
                    df_neg = df_local_with_type[df_local_with_type["DayType"] == False]
                else:
                    df_pos = pd.DataFrame(columns=df_local_selected.columns)
                    df_neg = pd.DataFrame(columns=df_local_selected.columns)

                # pivots
                if not df_pos.empty:
                    hourly_pos = df_pos.pivot_table(index="DateLocal", columns="HourLocal", values="PctChange", aggfunc="mean", fill_value=0)
                    avg_pos = hourly_pos.mean().round(5)
                else:
                    hourly_pos = pd.DataFrame(columns=range(24))
                    avg_pos = pd.Series([0.0]*24, index=range(24))

                if not df_neg.empty:
                    hourly_neg = df_neg.pivot_table(index="DateLocal", columns="HourLocal", values="PctChange", aggfunc="mean", fill_value=0)
                    avg_neg = hourly_neg.mean().round(5)
                else:
                    hourly_neg = pd.DataFrame(columns=range(24))
                    avg_neg = pd.Series([0.0]*24, index=range(24))

                if not df_local_selected.empty:
                    hourly_all = df_local_selected.pivot_table(index="DateLocal", columns="HourLocal", values="PctChange", aggfunc="mean", fill_value=0)
                    avg_all = hourly_all.mean().round(5)
                else:
                    hourly_all = pd.DataFrame(columns=range(24))
                    avg_all = pd.Series([0.0]*24, index=range(24))

        # stacked charts
        st.markdown(f"**Instrument:** {symbol_name} — Local timezone: **{timezone_option}** — Period: **{days_to_load} days**")
        st.subheader("Positive-Day Hourly Avg % Change (local time)")
        st.bar_chart(avg_pos)

        st.subheader("Negative-Day Hourly Avg % Change (local time)")
        st.bar_chart(avg_neg)

        st.subheader("Overall Hourly Avg % Change (local time)")
        st.bar_chart(avg_all)

        # expanders with tables
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

# -------------------------
# Tab 2: Best Hour summary across all instruments (table only) - sorted by BestHourInt ascending
# -------------------------
with tab2:
    st.write("Scanning all instruments to find the best (most-moving) local hour per instrument...")
    rows = []
    for display_name, sym in symbols.items():
        raw = load_hourly(sym, days_to_load)
        if raw.empty:
            rows.append({
                "Instrument": display_name,
                "BestHourInt": np.nan,
                "BestHour": np.nan,
                "BestHour_Avg%": np.nan,
                "BestHour_Latest%": np.nan,
                "BestHour_Diff%": np.nan,
                "SortKey": np.nan
            })
            continue

        # extract close
        try:
            close_s = extract_close_series(raw, sym)
        except Exception:
            rows.append({
                "Instrument": display_name,
                "BestHourInt": np.nan,
                "BestHour": np.nan,
                "BestHour_Avg%": np.nan,
                "BestHour_Latest%": np.nan,
                "BestHour_Diff%": np.nan,
                "SortKey": np.nan
            })
            continue

        df_sym = pd.DataFrame({"Close": close_s.astype(float)})
        # standardize to UTC
        if getattr(df_sym.index, "tz", None) is None:
            try:
                df_sym.index = df_sym.index.tz_localize("UTC")
            except Exception:
                df_sym.index = pd.to_datetime(df_sym.index).tz_localize("UTC")
        else:
            try:
                df_sym.index = df_sym.index.tz_convert("UTC")
            except Exception:
                df_sym.index = pd.to_datetime(df_sym.index).tz_convert("UTC")

        df_sym["PctChange"] = df_sym["Close"].pct_change() * 100
        df_sym = df_sym.dropna(subset=["PctChange"])
        if df_sym.empty:
            rows.append({
                "Instrument": display_name,
                "BestHourInt": np.nan,
                "BestHour": np.nan,
                "BestHour_Avg%": np.nan,
                "BestHour_Latest%": np.nan,
                "BestHour_Diff%": np.nan,
                "SortKey": np.nan
            })
            continue

        # convert to local tz and limit to last N local dates
        try:
            df_sym_local_all = df_sym.copy()
            df_sym_local_all.index = df_sym_local_all.index.tz_convert(timezone_option)
        except Exception:
            df_sym_local_all.index = pd.to_datetime(df_sym_local_all.index).tz_localize("UTC").tz_convert(timezone_option)

        df_sym_local_all["DateLocal"] = df_sym_local_all.index.date
        df_sym_local_all["HourLocal"] = df_sym_local_all.index.hour

        unique_dates_sym = sorted(pd.unique(df_sym_local_all["DateLocal"]))
        sel_dates_sym = unique_dates_sym[-days_to_load:] if len(unique_dates_sym) >= days_to_load else unique_dates_sym
        if not sel_dates_sym:
            rows.append({
                "Instrument": display_name,
                "BestHourInt": np.nan,
                "BestHour": np.nan,
                "BestHour_Avg%": np.nan,
                "BestHour_Latest%": np.nan,
                "BestHour_Diff%": np.nan,
                "SortKey": np.nan
            })
            continue

        df_sel = df_sym_local_all[df_sym_local_all["DateLocal"].isin(sel_dates_sym)].copy()

        # overall average per hour (signed) for this instrument
        pivot_all = df_sel.pivot_table(index="DateLocal", columns="HourLocal", values="PctChange", aggfunc="mean", fill_value=np.nan)

        # ensure columns 0..23 exist
        for h in range(24):
            if h not in pivot_all.columns:
                pivot_all[h] = np.nan
        pivot_all = pivot_all.reindex(sorted(pivot_all.columns), axis=1)

        # average across days
        avg_per_hour = pivot_all.mean(skipna=True)

        # pick best hour = argmax absolute(avg)
        if avg_per_hour.isna().all():
            rows.append({
                "Instrument": display_name,
                "BestHourInt": np.nan,
                "BestHour": np.nan,
                "BestHour_Avg%": np.nan,
                "BestHour_Latest%": np.nan,
                "BestHour_Diff%": np.nan,
                "SortKey": np.nan
            })
            continue

        abs_avg = avg_per_hour.abs()
        best_hour_int = int(abs_avg.idxmax())
        best_hour_avg = float(avg_per_hour.loc[best_hour_int])  # signed average

        # latest observed change for that hour (most recent timestamp in df_sel matching HourLocal)
        latest_best = np.nan
        latest_best_rows = df_sel[df_sel["HourLocal"] == best_hour_int]
        if not latest_best_rows.empty:
            latest_best = float(latest_best_rows["PctChange"].iloc[-1])

        # difference
        diff_best = np.nan if np.isnan(latest_best) else (latest_best - best_hour_avg)

        # sort key (kept for tie-breaker)
        sort_key = (best_hour_avg if not np.isnan(best_hour_avg) else 0.0) + (latest_best if not np.isnan(latest_best) else 0.0)

        rows.append({
            "Instrument": display_name,
            "BestHourInt": best_hour_int,
            "BestHour": f"{best_hour_int:02d}:00",
            "BestHour_Avg%": round(best_hour_avg, 5),
            "BestHour_Latest%": (round(latest_best, 5) if not np.isnan(latest_best) else np.nan),
            "BestHour_Diff%": (round(diff_best, 5) if not np.isnan(diff_best) else np.nan),
            "SortKey": round(sort_key, 6)
        })

    # build DataFrame & sort by BestHourInt ascending (0 -> 1 -> 2 ...)
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df[[
            "Instrument", "BestHourInt", "BestHour", "BestHour_Avg%", "BestHour_Latest%", "BestHour_Diff%", "SortKey"
        ]]
        # Replace NaN BestHourInt with a large number so they go to bottom
        summary_df["BestHourInt_sort"] = summary_df["BestHourInt"].apply(lambda x: 999 if pd.isna(x) else int(x))
        # sort by BestHourInt_sort then SortKey as tiebreaker
        summary_df = summary_df.sort_values(by=["BestHourInt_sort", "SortKey"], ascending=[True, True], na_position="last").reset_index(drop=True)
        summary_df = summary_df.drop(columns=["BestHourInt_sort"])

    st.subheader("Best hour summary (per instrument) — sorted by BestHour (0 → 1 → 2 ...)")
    st.dataframe(summary_df)
    st.caption("BestHour chosen by largest |avg| movement across selected days. Instruments with no data appear at the bottom.")
