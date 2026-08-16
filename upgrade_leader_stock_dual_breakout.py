from pathlib import Path

PAGE = Path("pages/2_Final_Opportunities.py")
MARKER = "LEADER STOCK WEEKLY MONTHLY BREAKOUT SCANNER"


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def helper_block():
    return r'''# LEADER STOCK WEEKLY MONTHLY BREAKOUT SCANNER

def _ohlc_from_download(raw, ticker, total):
    try:
        if raw is None or raw.empty:
            return pd.DataFrame()
        if total == 1 and not isinstance(raw.columns, pd.MultiIndex):
            d = raw.copy()
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
            d = raw[ticker].copy()
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(-1):
            d = raw.xs(ticker, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
        d = d[cols].copy().dropna(how="all")
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d.sort_index()
    except Exception:
        return pd.DataFrame()


def _breakout_state(close, high, lookback, recent_bars, near_pct=5.0):
    close = pd.to_numeric(close, errors="coerce").dropna()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    if len(close) < lookback + 2:
        return {"fresh": False, "recent": False, "bars_since": np.nan, "distance": np.nan, "level": np.nan}
    prior = high.shift(1).rolling(lookback, min_periods=lookback).max()
    signals = close > prior
    recent_slice = signals.iloc[-recent_bars:] if len(signals) >= recent_bars else signals
    recent = bool(recent_slice.any())
    bars_since = np.nan
    if recent:
        pos = int(np.flatnonzero(recent_slice.values)[-1])
        bars_since = len(recent_slice) - 1 - pos
    level = safe_float(prior.iloc[-1])
    last = safe_float(close.iloc[-1])
    distance = (last / level - 1) * 100 if pd.notna(level) and level else np.nan
    return {
        "fresh": bool(signals.iloc[-1]) if len(signals) else False,
        "recent": recent,
        "bars_since": bars_since,
        "distance": distance,
        "level": level,
        "near": bool(pd.notna(distance) and -near_pct <= distance <= 0),
    }


def _setup_label(w2, w3, m26, m60):
    if m60["fresh"]:
        return "🔥 MONTHLY 60M FRESH BREAKOUT", 100
    if m60["recent"]:
        return "🔥 MONTHLY 60M RECENT BREAKOUT", 96
    if m26["fresh"]:
        return "🟢 MONTHLY 26M FRESH BREAKOUT", 94
    if m26["recent"]:
        return "🟢 MONTHLY 26M RECENT BREAKOUT", 90
    if w3["fresh"]:
        return "🔥 WEEKLY 3Y FRESH BREAKOUT", 88
    if w3["recent"]:
        return "🟢 WEEKLY 3Y RECENT BREAKOUT", 84
    if w2["fresh"]:
        return "🟢 WEEKLY 2Y FRESH BREAKOUT", 82
    if w2["recent"]:
        return "🟢 WEEKLY 2Y RECENT BREAKOUT", 78
    if m60.get("near"):
        return "👀 WITHIN 5% OF MONTHLY 60M BREAKOUT", 74
    if m26.get("near"):
        return "👀 WITHIN 5% OF MONTHLY 26M BREAKOUT", 72
    if w3.get("near"):
        return "👀 WITHIN 5% OF WEEKLY 3Y BREAKOUT", 68
    if w2.get("near"):
        return "👀 WITHIN 5% OF WEEKLY 2Y BREAKOUT", 64
    return "Below breakout zone", 30


@st.cache_data(ttl=1800, show_spinner=False)
def scan_leader_sector_breakouts(symbols):
    """Batch scan one strict-leader sector for weekly + monthly breakout setups.

    Weekly: 2Y (104W) and 3Y (156W), recent = last 8 weeks.
    Monthly: 26M and 60M, recent = last 3 months.
    A signal requires CLOSE above the previous rolling HIGH, not just an intraperiod wick.
    """
    symbols = tuple(str(x).strip().upper() for x in symbols if str(x).strip())
    if not symbols:
        return pd.DataFrame()
    tickers = [s + ".NS" for s in symbols]
    rows = []
    for start in range(0, len(tickers), 50):
        chunk = tickers[start:start + 50]
        try:
            raw = yf.download(
                tickers=chunk, period="10y", interval="1wk", group_by="ticker",
                auto_adjust=False, threads=True, progress=False, timeout=35,
            )
        except TypeError:
            raw = yf.download(
                tickers=chunk, period="10y", interval="1wk", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
        except Exception:
            raw = pd.DataFrame()

        for ticker in chunk:
            d = _ohlc_from_download(raw, ticker, len(chunk))
            if d.empty or "Close" not in d.columns or "High" not in d.columns:
                continue
            weekly = d[["High", "Close"]].dropna()
            if len(weekly) < 110:
                continue
            monthly = weekly.resample("ME").agg({"High": "max", "Close": "last"}).dropna()

            w2 = _breakout_state(weekly["Close"], weekly["High"], 104, 8)
            w3 = _breakout_state(weekly["Close"], weekly["High"], 156, 8)
            m26 = _breakout_state(monthly["Close"], monthly["High"], 26, 3)
            m60 = _breakout_state(monthly["Close"], monthly["High"], 60, 3)
            setup, priority = _setup_label(w2, w3, m26, m60)
            rows.append({
                "Symbol": ticker.replace(".NS", ""),
                "Best Setup": setup,
                "Breakout Priority": priority,
                "Weekly 2Y Distance %": w2["distance"],
                "Weekly 3Y Distance %": w3["distance"],
                "Monthly 26M Distance %": m26["distance"],
                "Monthly 60M Distance %": m60["distance"],
                "W2Y Recent": w2["recent"], "W3Y Recent": w3["recent"],
                "M26 Recent": m26["recent"], "M60 Recent": m60["recent"],
                "Weeks Since W2Y": w2["bars_since"],
                "Months Since M26": m26["bars_since"],
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["Breakout Priority", "Monthly 60M Distance %", "Monthly 26M Distance %"], ascending=[False, False, False], na_position="last") if not out.empty else out


@st.cache_data(ttl=1800, show_spinner=False)
def strict_stock_breakout_dual_snapshot(ticker):
    out = strict_stock_breakout_snapshot(ticker)
    try:
        h = yf.Ticker(ticker).history(period="10y", interval="1d", auto_adjust=False)
    except Exception:
        h = pd.DataFrame()
    out.update({
        "Monthly Setup": "N/A", "Monthly 26M Distance %": np.nan,
        "Monthly 60M Distance %": np.nan, "Months Since 26M Breakout": np.nan,
    })
    if h is not None and not h.empty and "Close" in h.columns and "High" in h.columns:
        h.index = pd.to_datetime(h.index).tz_localize(None)
        m = h[["High", "Close"]].resample("ME").agg({"High": "max", "Close": "last"}).dropna()
        m26 = _breakout_state(m["Close"], m["High"], 26, 3)
        m60 = _breakout_state(m["Close"], m["High"], 60, 3)
        out["Monthly 26M Distance %"] = m26["distance"]
        out["Monthly 60M Distance %"] = m60["distance"]
        out["Months Since 26M Breakout"] = m26["bars_since"]
        if m60["fresh"]:
            out["Monthly Setup"] = "🔥 60M FRESH MONTHLY BREAKOUT"
        elif m60["recent"]:
            out["Monthly Setup"] = "🔥 60M RECENT MONTHLY BREAKOUT"
        elif m26["fresh"]:
            out["Monthly Setup"] = "🟢 26M FRESH MONTHLY BREAKOUT"
        elif m26["recent"]:
            out["Monthly Setup"] = "🟢 26M RECENT MONTHLY BREAKOUT"
        elif m60.get("near"):
            out["Monthly Setup"] = "👀 WITHIN 5% OF 60M MONTHLY BREAKOUT"
        elif m26.get("near"):
            out["Monthly Setup"] = "👀 WITHIN 5% OF 26M MONTHLY BREAKOUT"
        else:
            out["Monthly Setup"] = "Below monthly breakout zone"
    return out


'''


def main():
    text = PAGE.read_text(encoding="utf-8")
    if MARKER not in text:
        anchor = '@st.cache_data(ttl=1200, show_spinner=False)\ndef macro_state'
        pos = text.find(anchor)
        if pos == -1:
            raise RuntimeError("macro_state anchor not found")
        text = text[:pos] + helper_block() + text[pos:]

    # Add a sector-wide scan control immediately after the selected-sector stock set is created.
    anchor = '    sector_stock_rows = strict_stocks[strict_stocks["Industry"] == chosen_sector].copy()\n'
    if 'Scan selected leader sector — Weekly + Monthly' not in text:
        addition = r'''    scan_tf = st.radio(
        "Stock breakout scan timeframe",
        ["Both — Weekly + Monthly", "Weekly only", "Monthly only"],
        horizontal=True,
        help="Weekly catches 2Y/3Y breakouts. Monthly catches 26M/60M multiyear breakouts. Signals require CLOSE above the prior rolling HIGH.",
    )
    if st.button("🚀 Scan selected leader sector — Weekly + Monthly", type="primary", use_container_width=True):
        with st.spinner(f"Scanning every {chosen_sector} stock for weekly/monthly multiyear breakouts…"):
            sector_breakouts = scan_leader_sector_breakouts(sector_stock_rows["Symbol"].tolist())
        st.session_state["leader_sector_breakout_scan"] = (chosen_sector, sector_breakouts)

    saved_scan = st.session_state.get("leader_sector_breakout_scan")
    if saved_scan and saved_scan[0] == chosen_sector:
        breakout_table = saved_scan[1].copy()
        if not breakout_table.empty:
            if scan_tf == "Weekly only":
                keep = breakout_table["W2Y Recent"] | breakout_table["W3Y Recent"] | breakout_table["Weekly 2Y Distance %"].between(-5, 0) | breakout_table["Weekly 3Y Distance %"].between(-5, 0)
                breakout_table = breakout_table[keep]
            elif scan_tf == "Monthly only":
                keep = breakout_table["M26 Recent"] | breakout_table["M60 Recent"] | breakout_table["Monthly 26M Distance %"].between(-5, 0) | breakout_table["Monthly 60M Distance %"].between(-5, 0)
                breakout_table = breakout_table[keep]
            st.markdown("#### Breakout candidates inside selected strict leader sector")
            st.dataframe(breakout_table, use_container_width=True, hide_index=True)
            st.caption("Priority: monthly 60M → monthly 26M → weekly 3Y → weekly 2Y → within 5% of breakout. Use completed weekly/monthly closes for confirmation.")
        else:
            st.info("No usable price history was returned for the selected sector scan.")
'''
        text = replace_once(text, anchor, anchor + addition, "sector-wide breakout scan UI")

    text = text.replace(
        'if st.button(f"🔎 Analyse {selected_symbol} weekly breakout", type="primary", use_container_width=True):',
        'if st.button(f"🔎 Analyse {selected_symbol} — Weekly + Monthly", type="primary", use_container_width=True):'
    )
    text = text.replace(
        'snap = strict_stock_breakout_snapshot(selected_symbol + ".NS")',
        'snap = strict_stock_breakout_dual_snapshot(selected_symbol + ".NS")'
    )

    detail_anchor = '            st.caption("Stock trigger is weekly: priority = 3Y+ breakout → recent 2Y breakout (last 8 weeks) → within 5% of 2Y breakout. Confirm on completed weekly candles before acting.")\n'
    if detail_anchor in text and 'Monthly 26M Distance' not in text[text.find(detail_anchor)-1500:text.find(detail_anchor)+500]:
        detail = r'''            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Monthly Setup", snap.get("Monthly Setup", "N/A"))
            c2.metric("Monthly 26M Distance", "N/A" if pd.isna(snap.get("Monthly 26M Distance %", np.nan)) else f"{snap['Monthly 26M Distance %']:+.2f}%")
            c3.metric("Monthly 60M Distance", "N/A" if pd.isna(snap.get("Monthly 60M Distance %", np.nan)) else f"{snap['Monthly 60M Distance %']:+.2f}%")
            c4.metric("Months Since 26M Breakout", "N/A" if pd.isna(snap.get("Months Since 26M Breakout", np.nan)) else int(snap["Months Since 26M Breakout"]))
            st.caption("Stock trigger hierarchy: Monthly 60M → Monthly 26M → Weekly 3Y → Weekly 2Y → within 5% of breakout. Breakout = CLOSE above the previous rolling HIGH; confirm completed candle before acting.")
'''
        text = text.replace(detail_anchor, detail, 1)

    PAGE.write_text(text, encoding="utf-8")
    print("Leader-sector weekly + monthly stock breakout scanner installed")


if __name__ == "__main__":
    main()
