from pathlib import Path

PAGE = Path("pages/2_Final_Opportunities.py")
STRICT_MARKER = "STRICT WEEKLY RS BREAKOUT ENGINE"


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def strict_helpers():
    return r'''# STRICT WEEKLY RS BREAKOUT ENGINE
NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]


@st.cache_data(ttl=86400, show_spinner=False)
def load_official_nifty500():
    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    for url in NIFTY500_URLS:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if "Symbol" not in df.columns:
                continue
            if "Industry" not in df.columns:
                df["Industry"] = "Unknown"
            if "Company Name" not in df.columns:
                df["Company Name"] = df["Symbol"]
            df["Symbol"] = df["Symbol"].astype(str).str.strip()
            df["Ticker"] = df["Symbol"] + ".NS"
            return df[["Company Name", "Industry", "Symbol", "Ticker"]].drop_duplicates("Symbol")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"NIFTY 500 constituent feed unavailable: {last_error}")


def _weekly_close(raw, ticker, total):
    try:
        if raw is None or raw.empty:
            return pd.Series(dtype=float)
        if total == 1 and not isinstance(raw.columns, pd.MultiIndex):
            d = raw
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
            d = raw[ticker]
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(-1):
            d = raw.xs(ticker, axis=1, level=-1)
        else:
            return pd.Series(dtype=float)
        s = pd.to_numeric(d["Close"], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _period_return_weekly(c, weeks):
    c = pd.to_numeric(c, errors="coerce").dropna()
    if len(c) <= weeks:
        return np.nan
    base = safe_float(c.iloc[-weeks - 1])
    last = safe_float(c.iloc[-1])
    return (last / base - 1) * 100 if pd.notna(base) and base else np.nan


def _recent_breakout_info(series, lookback=52, recent_weeks=8):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < lookback + 2:
        return False, False, np.nan, np.nan
    prior_high = s.shift(1).rolling(lookback, min_periods=lookback).max()
    signal = s > prior_high
    recent_signal = signal.iloc[-recent_weeks:]
    if bool(recent_signal.any()):
        positions = np.flatnonzero(recent_signal.values)
        weeks_since = len(recent_signal) - 1 - int(positions[-1])
        breakout_level = safe_float(prior_high.iloc[-1])
        breakout_pct = (safe_float(s.iloc[-1]) / breakout_level - 1) * 100 if pd.notna(breakout_level) and breakout_level else np.nan
        return bool(signal.iloc[-1]), True, weeks_since, breakout_pct
    breakout_level = safe_float(prior_high.iloc[-1])
    breakout_pct = (safe_float(s.iloc[-1]) / breakout_level - 1) * 100 if pd.notna(breakout_level) and breakout_level else np.nan
    return bool(signal.iloc[-1]), False, np.nan, breakout_pct


@st.cache_data(ttl=3600, show_spinner=False)
def strict_weekly_sector_leadership():
    """Full NIFTY 500 sector leadership on WEEKLY data relative to NIFTY 50.

    A sector is a STRICT leader only when its equal-weight sector proxy / NIFTY 50
    relative-strength line has broken its prior 52-week high now or during the last
    8 completed weekly observations, 13W and 26W relative strength are positive,
    and at least half its available stocks beat NIFTY 50 over 13 weeks.
    """
    universe = load_official_nifty500()
    tickers = universe["Ticker"].tolist()
    stock_series = {}
    chunk_size = 80
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start:start + chunk_size]
        try:
            raw = yf.download(
                tickers=chunk, period="5y", interval="1wk", group_by="ticker",
                auto_adjust=True, threads=True, progress=False, timeout=35,
            )
        except TypeError:
            raw = yf.download(
                tickers=chunk, period="5y", interval="1wk", group_by="ticker",
                auto_adjust=True, threads=True, progress=False,
            )
        except Exception:
            raw = pd.DataFrame()
        for ticker in chunk:
            c = _weekly_close(raw, ticker, len(chunk))
            if len(c) >= 60:
                stock_series[ticker] = c

    try:
        nifty_raw = yf.download("^NSEI", period="5y", interval="1wk", auto_adjust=True, progress=False, timeout=30)
    except TypeError:
        nifty_raw = yf.download("^NSEI", period="5y", interval="1wk", auto_adjust=True, progress=False)
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = nifty_raw.columns.get_level_values(0)
    nifty = pd.to_numeric(nifty_raw.get("Close"), errors="coerce").dropna()
    nifty.index = pd.to_datetime(nifty.index).tz_localize(None)
    nifty = nifty.sort_index()
    if len(nifty) < 60:
        raise RuntimeError("NIFTY 50 weekly history unavailable")

    nifty_r13 = _period_return_weekly(nifty, 13)
    nifty_r26 = _period_return_weekly(nifty, 26)
    meta = universe.set_index("Ticker")
    stock_rows = []
    sector_rows = []

    for industry, group in universe.groupby("Industry"):
        series_map = {t: stock_series[t] for t in group["Ticker"] if t in stock_series}
        if not series_map:
            continue
        close_panel = pd.concat(series_map, axis=1).sort_index()
        weekly_returns = close_panel.pct_change(fill_method=None)
        sector_weekly_ret = weekly_returns.median(axis=1, skipna=True).dropna()
        if len(sector_weekly_ret) < 55:
            continue
        sector_proxy = (1 + sector_weekly_ret).cumprod() * 100.0
        n = nifty.reindex(sector_proxy.index).ffill().dropna()
        common = sector_proxy.index.intersection(n.index)
        sector_proxy = sector_proxy.loc[common]
        n = n.loc[common]
        if len(common) < 55:
            continue
        nifty_proxy = n / n.iloc[0] * 100.0
        rs_line = sector_proxy / nifty_proxy * 100.0
        fresh, recent, weeks_since, rs_breakout_pct = _recent_breakout_info(rs_line, 52, 8)
        rs13 = _period_return_weekly(rs_line, 13)
        rs26 = _period_return_weekly(rs_line, 26)

        beat13 = []
        beat26 = []
        for ticker, c in series_map.items():
            r13 = _period_return_weekly(c, 13)
            r26 = _period_return_weekly(c, 26)
            b13 = bool(pd.notna(r13) and pd.notna(nifty_r13) and r13 > nifty_r13)
            b26 = bool(pd.notna(r26) and pd.notna(nifty_r26) and r26 > nifty_r26)
            beat13.append(b13)
            beat26.append(b26)
            stock_rows.append({
                "Symbol": meta.loc[ticker, "Symbol"], "Company": meta.loc[ticker, "Company Name"],
                "Industry": industry, "Ticker": ticker,
                "13W %": r13, "26W %": r26,
                "RS vs NIFTY 13W %": r13 - nifty_r13 if pd.notna(r13) and pd.notna(nifty_r13) else np.nan,
                "RS vs NIFTY 26W %": r26 - nifty_r26 if pd.notna(r26) and pd.notna(nifty_r26) else np.nan,
                "Beat NIFTY 13W": b13, "Beat NIFTY 26W": b26,
            })

        breadth13 = 100 * np.mean(beat13) if beat13 else np.nan
        breadth26 = 100 * np.mean(beat26) if beat26 else np.nan
        strict = bool(recent and pd.notna(rs13) and rs13 > 0 and pd.notna(rs26) and rs26 > 0 and pd.notna(breadth13) and breadth13 >= 50)
        recency_score = 10 if fresh else (max(2, 9 - safe_float(weeks_since)) if recent else 0)
        score = float(np.clip(
            0.35 * recency_score +
            0.25 * (5 + 5 * np.tanh((rs13 if pd.notna(rs13) else -10) / 8)) +
            0.20 * (5 + 5 * np.tanh((rs26 if pd.notna(rs26) else -10) / 12)) +
            0.20 * ((breadth13 if pd.notna(breadth13) else 0) / 10), 0, 10
        ))
        status = "🟢 FRESH RS BREAKOUT" if strict and fresh else "🟢 RECENT RS BREAKOUT" if strict else "🟡 WATCH" if recent else "⚪ NO RS BREAKOUT"
        sector_rows.append({
            "Industry": industry, "Stocks": len(series_map), "Strict Leader": strict,
            "RS Breakout Status": status, "Weeks Since RS Breakout": weeks_since,
            "RS Breakout %": rs_breakout_pct, "RS 13W %": rs13, "RS 26W %": rs26,
            "13W Breadth vs NIFTY %": breadth13, "26W Breadth vs NIFTY %": breadth26,
            "Sector Score": score,
        })

    sectors = pd.DataFrame(sector_rows)
    stocks = pd.DataFrame(stock_rows)
    if sectors.empty:
        return universe, stocks, sectors
    sectors["_leader"] = sectors["Strict Leader"].astype(int)
    sectors["_weeks"] = sectors["Weeks Since RS Breakout"].fillna(999)
    sectors = sectors.sort_values(["_leader", "_weeks", "Sector Score"], ascending=[False, True, False]).drop(columns=["_leader", "_weeks"]).reset_index(drop=True)
    return universe, stocks, sectors


@st.cache_data(ttl=1800, show_spinner=False)
def strict_stock_breakout_snapshot(ticker):
    """Weekly stock setup: catch 2Y/3Y breakout, recent breakout, or within 5% of breakout."""
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        h = t.history(period="10y", interval="1wk", auto_adjust=False)
    except Exception:
        h = pd.DataFrame()
    try:
        n = yf.download("^NSEI", period="3y", interval="1wk", auto_adjust=True, progress=False)
        if isinstance(n.columns, pd.MultiIndex):
            n.columns = n.columns.get_level_values(0)
        nifty = pd.to_numeric(n.get("Close"), errors="coerce").dropna()
    except Exception:
        nifty = pd.Series(dtype=float)

    out = {"Company": info.get("longName") or info.get("shortName") or ticker.replace(".NS", ""),
           "CMP": safe_float(info.get("currentPrice", info.get("regularMarketPrice"))),
           "Setup": "N/A", "2Y Breakout %": np.nan, "3Y Breakout %": np.nan,
           "Weeks Since 2Y Breakout": np.nan, "RS vs NIFTY 13W %": np.nan,
           "RS vs NIFTY 26W %": np.nan, "Fundamental Score %": np.nan,
           "Fundamental Checks": 0, "Trailing PE": np.nan}
    if h is not None and not h.empty and "Close" in h.columns and "High" in h.columns:
        h.index = pd.to_datetime(h.index).tz_localize(None)
        close = pd.to_numeric(h["Close"], errors="coerce").dropna()
        high = pd.to_numeric(h["High"], errors="coerce").reindex(close.index)
        if pd.isna(out["CMP"]) and len(close):
            out["CMP"] = safe_float(close.iloc[-1])
        prior2 = high.shift(1).rolling(104, min_periods=104).max()
        prior3 = high.shift(1).rolling(156, min_periods=156).max()
        sig2 = close > prior2
        recent2 = sig2.iloc[-8:] if len(sig2) >= 8 else sig2
        if bool(recent2.any()):
            pos = np.flatnonzero(recent2.values)[-1]
            out["Weeks Since 2Y Breakout"] = len(recent2) - 1 - int(pos)
        lev2 = safe_float(prior2.iloc[-1]) if len(prior2) else np.nan
        lev3 = safe_float(prior3.iloc[-1]) if len(prior3) else np.nan
        last = safe_float(close.iloc[-1])
        out["2Y Breakout %"] = (last / lev2 - 1) * 100 if pd.notna(lev2) and lev2 else np.nan
        out["3Y Breakout %"] = (last / lev3 - 1) * 100 if pd.notna(lev3) and lev3 else np.nan
        if pd.notna(out["3Y Breakout %"]) and out["3Y Breakout %"] > 0:
            out["Setup"] = "🔥 3Y+ MULTIYEAR BREAKOUT"
        elif pd.notna(out["Weeks Since 2Y Breakout"]) and out["Weeks Since 2Y Breakout"] <= 8:
            out["Setup"] = "✅ RECENT 2Y BREAKOUT"
        elif pd.notna(out["2Y Breakout %"]) and -5 <= out["2Y Breakout %"] <= 0:
            out["Setup"] = "👀 WITHIN 5% OF 2Y BREAKOUT"
        else:
            out["Setup"] = "Below multiyear breakout zone"
        sr13 = _period_return_weekly(close, 13)
        sr26 = _period_return_weekly(close, 26)
        nr13 = _period_return_weekly(nifty, 13)
        nr26 = _period_return_weekly(nifty, 26)
        out["RS vs NIFTY 13W %"] = sr13 - nr13 if pd.notna(sr13) and pd.notna(nr13) else np.nan
        out["RS vs NIFTY 26W %"] = sr26 - nr26 if pd.notna(sr26) and pd.notna(nr26) else np.nan
    try:
        fs, assessed, pe = fundamental_quality(ticker)
        out["Fundamental Score %"] = fs
        out["Fundamental Checks"] = assessed
        out["Trailing PE"] = pe
    except Exception:
        pass
    return out


'''


def strict_ui():
    return r'''st.divider()
st.subheader("🎯 Strict Weekly Sector RS Breakout → Stock Analyzer")
st.caption(
    "STRICT rule: every NIFTY 500 industry is compared with NIFTY 50 on WEEKLY data. A sector becomes a leader only when its "
    "relative-strength line (sector proxy ÷ NIFTY 50) breaks its prior 52-week RS high now or within the last 8 weeks, both 13W and 26W RS are positive, "
    "and at least 50% of its stocks outperform NIFTY 50 over 13 weeks. A high momentum score alone CANNOT make a sector leader."
)

try:
    with st.spinner("Scanning all NIFTY 500 sectors on weekly relative strength vs NIFTY 50…"):
        nse_universe, strict_stocks, strict_sectors = strict_weekly_sector_leadership()
except Exception as exc:
    nse_universe, strict_stocks, strict_sectors = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    st.warning(f"Strict weekly sector feed is temporarily unavailable: {exc}")

if not strict_sectors.empty:
    strict_leaders = strict_sectors[strict_sectors["Strict Leader"]].copy()
    strict_map = strict_sectors.set_index("Industry")["Sector Score"].to_dict()
    st.session_state["strict_sector_scores"] = strict_map
    leader_name = strict_leaders.iloc[0]["Industry"] if not strict_leaders.empty else "NO STRICT LEADER"
    leader_score = strict_leaders.iloc[0]["Sector Score"] if not strict_leaders.empty else np.nan
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Strict Leader", leader_name)
    x2.metric("Leader Score", "N/A" if pd.isna(leader_score) else f"{leader_score:.1f}/10")
    x3.metric("NIFTY 500 Stocks", int(strict_stocks["Symbol"].nunique()))
    x4.metric("Strict Leaders Now", len(strict_leaders))

    if strict_leaders.empty:
        st.warning("No sector currently satisfies ALL strict weekly RS-breakout conditions. The table below is watchlist only; no sector is labelled leader.")
    else:
        st.success(f"Current strict leader: **{leader_name}**. It has a fresh/recent weekly RS breakout versus NIFTY 50 with positive 13W/26W relative strength and required breadth.")

    sector_cols = ["Industry", "Stocks", "Strict Leader", "RS Breakout Status", "Weeks Since RS Breakout",
                   "RS Breakout %", "RS 13W %", "RS 26W %", "13W Breadth vs NIFTY %", "26W Breadth vs NIFTY %", "Sector Score"]
    st.dataframe(
        strict_sectors[sector_cols], use_container_width=True, hide_index=True,
        column_config={
            "Sector Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
            "13W Breadth vs NIFTY %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "26W Breadth vs NIFTY %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        },
    )

    selectable_sectors = strict_leaders["Industry"].tolist() if not strict_leaders.empty else strict_sectors.head(5)["Industry"].tolist()
    chosen_sector = st.selectbox(
        "Select strict leader sector" if not strict_leaders.empty else "No strict leader — select a WATCH sector only",
        selectable_sectors,
        help="When strict leaders exist, only sectors that pass every weekly RS-breakout rule are offered here."
    )
    sector_stock_rows = strict_stocks[strict_stocks["Industry"] == chosen_sector].copy()
    if not sector_stock_rows.empty:
        # Fast first-pass ranking to put relative-strength stocks at the top before detailed weekly breakout analysis.
        sector_stock_rows["RS Rank Score"] = (
            sector_stock_rows["RS vs NIFTY 13W %"].fillna(-50) * 0.6 +
            sector_stock_rows["RS vs NIFTY 26W %"].fillna(-50) * 0.4
        )
        sector_stock_rows = sector_stock_rows.sort_values("RS Rank Score", ascending=False)
        st.markdown("#### Stocks inside selected sector — relative strength first")
        st.dataframe(
            sector_stock_rows[["Symbol", "Company", "13W %", "26W %", "RS vs NIFTY 13W %", "RS vs NIFTY 26W %", "RS Rank Score"]],
            use_container_width=True, hide_index=True,
        )
        options = sector_stock_rows.apply(lambda r: f"{r['Symbol']} — {r['Company']}", axis=1).tolist()
        selected_label = st.selectbox("Select stock for strict breakout analysis", options)
        selected_symbol = selected_label.split(" — ", 1)[0]
        if st.button(f"🔎 Analyse {selected_symbol} weekly breakout", type="primary", use_container_width=True):
            with st.spinner(f"Checking 2Y/3Y weekly breakout levels for {selected_symbol}…"):
                snap = strict_stock_breakout_snapshot(selected_symbol + ".NS")
            st.session_state["strict_selected_stock_snapshot"] = (chosen_sector, selected_symbol, snap)
        saved = st.session_state.get("strict_selected_stock_snapshot")
        if saved and saved[0] == chosen_sector and saved[1] == selected_symbol:
            snap = saved[2]
            st.markdown(f"### {selected_symbol} — {snap['Company']}")
            st.success(f"Setup: **{snap['Setup']}**") if ("BREAKOUT" in str(snap['Setup']) or "WITHIN 5%" in str(snap['Setup'])) else st.info(f"Setup: **{snap['Setup']}**")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("CMP", "N/A" if pd.isna(snap["CMP"]) else f"₹{snap['CMP']:,.2f}")
            a2.metric("2Y Breakout Distance", "N/A" if pd.isna(snap["2Y Breakout %"]) else f"{snap['2Y Breakout %']:+.2f}%")
            a3.metric("3Y Breakout Distance", "N/A" if pd.isna(snap["3Y Breakout %"]) else f"{snap['3Y Breakout %']:+.2f}%")
            a4.metric("Weeks Since 2Y Breakout", "N/A" if pd.isna(snap["Weeks Since 2Y Breakout"]) else int(snap["Weeks Since 2Y Breakout"]))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("RS vs NIFTY 13W", "N/A" if pd.isna(snap["RS vs NIFTY 13W %"]) else f"{snap['RS vs NIFTY 13W %']:+.1f}%")
            b2.metric("RS vs NIFTY 26W", "N/A" if pd.isna(snap["RS vs NIFTY 26W %"]) else f"{snap['RS vs NIFTY 26W %']:+.1f}%")
            b3.metric("Fundamental Score", "N/A" if pd.isna(snap["Fundamental Score %"]) else f"{snap['Fundamental Score %']:.0f}%")
            b4.metric("Trailing PE", "N/A" if pd.isna(snap["Trailing PE"]) else f"{snap['Trailing PE']:.1f}x")
            st.caption("Stock trigger is weekly: priority = 3Y+ breakout → recent 2Y breakout (last 8 weeks) → within 5% of 2Y breakout. Confirm on completed weekly candles before acting.")

'''


def main():
    text = PAGE.read_text(encoding="utf-8")

    if "import io\n" not in text:
        text = replace_once(
            text,
            "import numpy as np\nimport pandas as pd\nimport streamlit as st\nimport yfinance as yf\n",
            "import io\nimport numpy as np\nimport pandas as pd\nimport requests\nimport streamlit as st\nimport yfinance as yf\n",
            "imports",
        )

    # Replace the previously-added momentum-only full-NSE helper block.
    helper_start = text.find("NIFTY500_URLS = [")
    helper_end = text.find("@st.cache_data(ttl=1200, show_spinner=False)\ndef macro_state", helper_start)
    if helper_start != -1 and helper_end != -1:
        text = text[:helper_start] + strict_helpers() + text[helper_end:]
    elif STRICT_MARKER not in text:
        anchor = "@st.cache_data(ttl=1200, show_spinner=False)\ndef macro_state"
        pos = text.find(anchor)
        if pos == -1:
            raise RuntimeError("macro_state anchor not found")
        text = text[:pos] + strict_helpers() + text[pos:]

    # Replace old Full NSE UI block, which sat immediately before the scan-empty guard.
    ui_start = text.find('st.divider()\nst.subheader("🏅 Full NSE 500 Sector Leadership → Stock Analyzer")')
    scan_guard = text.find('if scan.empty:', ui_start if ui_start != -1 else 0)
    if ui_start != -1 and scan_guard != -1:
        text = text[:ui_start] + strict_ui() + text[scan_guard:]
    elif "Strict Weekly Sector RS Breakout → Stock Analyzer" not in text:
        warning_anchor = 'if coverage < 70:\n    st.warning("Macro data coverage is below 70%. Treat the macro score and regime as low-confidence until feeds recover.")\n\n'
        if warning_anchor not in text:
            raise RuntimeError("UI insertion anchor not found")
        text = text.replace(warning_anchor, warning_anchor + strict_ui(), 1)

    # The older expander is scan-only. Make that explicit and stop it from being mistaken for the true leader engine.
    text = text.replace('with st.expander("🔄 Sector leadership used in ranking", expanded=False):',
                        'with st.expander("📎 26M scan-only sector context (NOT the strict leader selector)", expanded=False):')

    # Use full-NSE strict sector score in final ranking whenever it is available; retain legacy fallback only for feed failure.
    old = 'smap = sectors.set_index("Industry")["Sector Score"].to_dict() if not sectors.empty else {}'
    new = ('legacy_smap = sectors.set_index("Industry")["Sector Score"].to_dict() if not sectors.empty else {}\n'
           '    strict_smap = st.session_state.get("strict_sector_scores", {})\n'
           '    smap = strict_smap if strict_smap else legacy_smap')
    if old in text:
        text = text.replace(old, new, 1)

    PAGE.write_text(text, encoding="utf-8")
    print("Strict weekly NIFTY50-relative sector leadership installed")


if __name__ == "__main__":
    main()
