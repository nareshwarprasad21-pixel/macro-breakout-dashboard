"""Weekly ACAC (ATH + CIP + 30 SMA) scanner for the NIFTY 500 dashboard.

Integration in app.py:
    from acac_scanner import render_acac_page
    render_acac_page(load_nifty500, download_prices, extract_one)
"""
import numpy as np
import pandas as pd
import streamlit as st


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Convert daily OHLCV to completed Friday-ending weekly candles."""
    if daily is None or daily.empty:
        return pd.DataFrame()
    x = daily.copy()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    columns = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    columns = {key: value for key, value in columns.items() if key in x.columns}
    weekly = x.resample("W-FRI").agg(columns).dropna(subset=["Open", "High", "Low", "Close"])
    # A mid-week bar is a developing candle, never an ACAC confirmation.
    if len(weekly) and x.index.max().weekday() < 4:
        weekly = weekly.iloc[:-1]
    return weekly


def candle_type(candle: pd.Series) -> str:
    body = abs(float(candle.Close) - float(candle.Open))
    rng = max(float(candle.High) - float(candle.Low), 1e-9)
    lower = min(float(candle.Open), float(candle.Close)) - float(candle.Low)
    if lower >= body * 2 and float(candle.Close) > float(candle.Open):
        return "Bullish Hammer"
    if float(candle.Close) > float(candle.Open) and body / rng >= 0.55:
        return "Strong Bullish"
    return "Bullish"


def detect_acac(
    weekly: pd.DataFrame,
    retest_weeks: int = 20,
    cip_tolerance: float = 2.0,
    confluence_tolerance: float = 3.0,
    max_ath_distance: float = 8.0,
) -> dict | None:
    """Return a confirmed ACAC setup from the latest *completed* weekly bar.

    CIP is the former ATH that was closed above on breakout, then tested as support.
    """
    if weekly is None or len(weekly) < 42:
        return None
    w = weekly.copy().dropna(subset=["Open", "High", "Low", "Close"])
    if len(w) < 42:
        return None
    w["SMA30"] = w["Close"].rolling(30).mean()
    i = len(w) - 1
    cur = w.iloc[i]
    sma = float(cur.SMA30)
    if not np.isfinite(sma) or i < 31:
        return None

    # Latest breakout that closed over the ATH available before that week.
    breakout = None
    for j in range(30, i):
        old_ath = float(w["High"].iloc[:j].max())
        if float(w["Close"].iloc[j]) > old_ath:
            breakout = (j, old_ath)
    if breakout is None:
        return None
    j, cip = breakout
    weeks_since = i - j
    if not 1 <= weeks_since <= retest_weeks:
        return None

    actual_ath = float(w["High"].max())
    ath_distance = (actual_ath / float(cur.Close) - 1) * 100
    if ath_distance > max_ath_distance:
        return None
    if float(cur.Close) <= float(cur.Open) or float(cur.Close) <= float(w["Close"].iloc[i - 1]):
        return None
    if float(cur.Close) <= sma or sma <= float(w["SMA30"].iloc[i - 1]):
        return None

    # Low must touch/retest each support area. The zone condition prevents a loose SMA-only signal.
    cip_touch = abs(float(cur.Low) / cip - 1) * 100 <= cip_tolerance
    sma_touch = abs(float(cur.Low) / sma - 1) * 100 <= cip_tolerance
    confluence = abs(cip / sma - 1) * 100 <= confluence_tolerance
    if not (cip_touch and sma_touch and confluence and float(cur.Close) > cip):
        return None

    return {
        "Signal Date": w.index[i], "CIP / Previous ATH": cip, "Actual ATH": actual_ath,
        "Weekly Close": float(cur.Close), "Weekly Low": float(cur.Low), "30 SMA": sma,
        "Weeks Since Breakout": weeks_since, "ATH Distance %": ath_distance,
        "CIP–30SMA Gap %": abs(cip / sma - 1) * 100,
        "Candle": candle_type(cur), "Suggested Stop-loss": min(float(cur.Low), cip, sma) * 0.985,
    }


def render_acac_page(load_nifty500, download_prices, extract_one):
    st.title("🎯 ACAC Weekly Scanner — ATH + CIP + 30 SMA")
    st.caption("Confirmed setup only: former ATH becomes support (CIP), aligns with a rising 30-week SMA, and latest completed week closes bullish.")
    universe = load_nifty500()
    industries = sorted(universe["Industry"].dropna().unique().tolist())
    c1, c2, c3 = st.columns(3)
    chosen = c1.multiselect("Sector / Industry (optional)", industries, key="acac_industry")
    batch = c2.selectbox("Scan universe", [100, 200, 500], index=2, key="acac_batch")
    weeks = c3.slider("Maximum weeks after breakout", 4, 30, 20, key="acac_weeks")
    st.caption("Default safeguards: CIP/30-SMA touch ±2%, CIP–30SMA confluence ≤3%, price within 8% of ATH. Scan after Friday close for a confirmed candle.")
    filtered = universe[universe["Industry"].isin(chosen)] if chosen else universe
    filtered = filtered.head(int(batch)).copy()
    if not st.button(f"Run ACAC scan on {len(filtered)} stocks", type="primary", use_container_width=True):
        previous = st.session_state.get("acac_results", pd.DataFrame())
        if previous.empty:
            st.info("Choose an optional sector filter and run the weekly scan.")
        else:
            st.dataframe(previous, use_container_width=True, hide_index=True)
        return
    progress = st.progress(0, text="Downloading NIFTY 500 daily history…")
    raw = download_prices(filtered["Ticker"].tolist(), period="max")
    rows = []
    for n, (_, row) in enumerate(filtered.reset_index(drop=True).iterrows(), start=1):
        daily = extract_one(raw, row.Ticker, len(filtered))
        signal = detect_acac(to_weekly(daily), retest_weeks=int(weeks))
        if signal:
            rows.append({"Symbol": row.Symbol, "Company": row["Company Name"], "Industry": row.Industry, **signal})
        progress.progress(n / len(filtered), text=f"Checking {n}/{len(filtered)}: {row.Symbol}")
    progress.empty()
    result = pd.DataFrame(rows)
    st.session_state["acac_results"] = result
    if result.empty:
        st.warning("No confirmed ACAC setup found under the strict rules in this universe. This is normal; the setup is deliberately selective.")
        return
    result = result.sort_values(["ATH Distance %", "CIP–30SMA Gap %"], ascending=True)
    st.success(f"{len(result)} confirmed ACAC candidate(s) found.")
    st.dataframe(result, use_container_width=True, hide_index=True, column_config={
        "ATH Distance %": st.column_config.NumberColumn(format="%.2f%%"),
        "CIP–30SMA Gap %": st.column_config.NumberColumn(format="%.2f%%"),
        "CIP / Previous ATH": st.column_config.NumberColumn(format="₹%.2f"),
        "Actual ATH": st.column_config.NumberColumn(format="₹%.2f"),
        "Weekly Close": st.column_config.NumberColumn(format="₹%.2f"),
        "30 SMA": st.column_config.NumberColumn(format="₹%.2f"),
        "Suggested Stop-loss": st.column_config.NumberColumn(format="₹%.2f"),
    })
    st.download_button("Download ACAC candidates CSV", result.to_csv(index=False).encode(), "acac_weekly_candidates.csv", "text/csv", use_container_width=True)
