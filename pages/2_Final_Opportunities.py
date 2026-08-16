from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Final Opportunities", page_icon="⭐", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1600px;}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.20); border-radius:12px; padding:10px; min-width:0;}
[data-testid="stMetricValue"] {font-size:1.7rem; white-space:normal; overflow:visible; text-overflow:clip;}
@media (max-width: 900px) {
  .block-container {padding: 1.15rem .65rem 1.5rem !important;}
  [data-testid="stMetricValue"] {font-size:1.35rem;}
  .stDataFrame {overflow-x:auto;}
  h1 {font-size:1.7rem !important;}
}
</style>
""",
    unsafe_allow_html=True,
)

MACRO_ASSETS = {
    "NIFTY 50": "^NSEI",
    "India VIX": "^INDIAVIX",
    "USD/INR": "INR=X",
    "Brent Crude": "BZ=F",
    "Dollar Index": "DX-Y.NYB",
    "US 10Y Yield": "^TNX",
    "Gold": "GC=F",
}


def safe_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _close_series(raw, ticker, total):
    try:
        if raw.empty:
            return pd.Series(dtype=float)
        if total == 1 and not isinstance(raw.columns, pd.MultiIndex):
            d = raw
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
            d = raw[ticker]
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(-1):
            d = raw.xs(ticker, axis=1, level=-1)
        else:
            return pd.Series(dtype=float)
        return pd.to_numeric(d["Close"], errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=1200, show_spinner=False)
def macro_state():
    tickers = list(MACRO_ASSETS.values())
    try:
        raw = yf.download(tickers=tickers, period="2y", interval="1d", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False, timeout=25)
    except TypeError:
        raw = yf.download(tickers=tickers, period="2y", interval="1d", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False)
    except Exception:
        raw = pd.DataFrame()

    specs = [
        ("NIFTY 50", +1, 2.2, 8.0), ("India VIX", -1, 1.7, 15.0),
        ("USD/INR", -1, 1.5, 4.0), ("Brent Crude", -1, 1.5, 12.0),
        ("Dollar Index", -1, 1.0, 6.0), ("US 10Y Yield", -1, 1.0, 8.0),
        ("Gold", -1, .6, 12.0),
    ]
    rows, weighted, weights = [], 0.0, 0.0
    total_weight = sum(x[2] for x in specs)
    for name, direction, weight, scale in specs:
        ticker = MACRO_ASSETS[name]
        c = _close_series(raw, ticker, len(tickers))
        if len(c) < 25:
            continue
        last = safe_float(c.iloc[-1])
        r1 = (last / safe_float(c.iloc[-22]) - 1) * 100 if len(c) > 22 else np.nan
        r3 = (last / safe_float(c.iloc[-64]) - 1) * 100 if len(c) > 63 else np.nan
        if pd.isna(r3):
            continue
        impulse = .75 * r3 + .25 * (r1 if pd.notna(r1) else r3)
        signal = float(np.tanh(direction * impulse / scale))
        weighted += signal * weight
        weights += weight
        rows.append({"Indicator": name, "Latest": last, "1M %": r1, "3M %": r3,
                     "Contribution": signal * weight, "As Of": pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d")})
    score = float(np.clip(5 + 5 * weighted / weights, 0, 10)) if weights else 5.0
    coverage = int(round(100 * weights / total_weight)) if total_weight else 0
    df = pd.DataFrame(rows)
    def val(name, col="3M %"):
        r = df[df["Indicator"] == name]
        return safe_float(r.iloc[0][col]) if not r.empty else np.nan
    nifty, crude, fx = val("NIFTY 50"), val("Brent Crude"), val("USD/INR")
    vix_row = df[df["Indicator"] == "India VIX"]
    vix = safe_float(vix_row.iloc[0]["Latest"]) if not vix_row.empty else np.nan
    inflation = int(pd.notna(crude) and crude > 10) + int(pd.notna(fx) and fx > 3)
    stress = int(pd.notna(nifty) and nifty < -5) + int(pd.notna(vix) and vix > 20)
    if score >= 7.6 and (pd.isna(nifty) or nifty > 0) and stress == 0:
        regime, short = "EARLY / RISK-ON", "EARLY"
    elif score >= 6.8 and stress == 0:
        regime, short = "MID CYCLE / EXPANSION", "MID"
    elif inflation >= 1 and score >= 3.8:
        regime, short = "LATE CYCLE / INFLATION-SENSITIVE", "LATE"
    elif score < 3.8 or stress >= 2:
        regime, short = "RISK-OFF / CONTRACTION", "RISK-OFF"
    else:
        regime, short = "MID-TO-LATE / MIXED", "MID→LATE"
    return score, regime, short, coverage, df


@st.cache_data(ttl=3600, show_spinner=False)
def fundamental_quality(ticker):
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        inc = t.financials.copy()
    except Exception:
        inc = pd.DataFrame()
    try:
        bs = t.balance_sheet.copy()
    except Exception:
        bs = pd.DataFrame()

    def row(df, names):
        if df is None or df.empty:
            return pd.Series(dtype=float)
        labels = {str(i).lower(): i for i in df.index}
        for name in names:
            if name.lower() in labels:
                s = pd.to_numeric(df.loc[labels[name.lower()]], errors="coerce").dropna()
                s.index = pd.to_datetime(s.index, errors="coerce")
                return s[s.index.notna()].sort_index()
        return pd.Series(dtype=float)

    revenue = row(inc, ["Total Revenue", "Operating Revenue"])
    profit = row(inc, ["Net Income", "Net Income Common Stockholders"])
    ebit = row(inc, ["EBIT", "Operating Income"])
    assets = row(bs, ["Total Assets"])
    current_liab = row(bs, ["Current Liabilities", "Total Current Liabilities"])
    equity = row(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    debt = row(bs, ["Total Debt"])

    def cagr(s):
        s = s.dropna().sort_index()
        if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
            return np.nan, 0
        years = (s.index[-1] - s.index[0]).days / 365.25
        return (((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100, years) if years > 0 else (np.nan, 0)

    sales, sy = cagr(revenue); profits, py = cagr(profit)
    eq = safe_float(equity.iloc[-1]) if len(equity) else np.nan
    td = safe_float(info.get("totalDebt"))
    if pd.isna(td) and len(debt): td = safe_float(debt.iloc[-1])
    de = td / eq if pd.notna(td) and pd.notna(eq) and eq else np.nan
    latest_ebit = safe_float(ebit.iloc[-1]) if len(ebit) else np.nan
    common = assets.index.intersection(current_liab.index)
    cap = safe_float(assets.loc[common[-1]] - current_liab.loc[common[-1]]) if len(common) else np.nan
    roce = latest_ebit / cap * 100 if pd.notna(latest_ebit) and pd.notna(cap) and cap else np.nan
    p = profit.dropna().sort_index()
    increasing = bool(len(p) >= 3 and all(p.iloc[i] >= p.iloc[i-1] for i in range(1, len(p)))) if len(p) >= 3 else None
    checks = []
    for value, available, fn in [
        (de, pd.notna(de), lambda x: x < .5), (roce, pd.notna(roce), lambda x: x > 15),
        (increasing, increasing is not None, lambda x: x is True),
        (sales, pd.notna(sales) and sy >= 5, lambda x: x > 20),
        (profits, pd.notna(profits) and py >= 5, lambda x: x > 15),
    ]:
        if available:
            checks.append(1 if fn(value) else 0)
    return 100 * sum(checks) / len(checks) if checks else np.nan, len(checks), info.get("trailingPE", np.nan)


def technical_score(row):
    status = str(row.get("Status", ""))
    breakout = safe_float(row.get("Breakout %"))
    vr = safe_float(row.get("Volume Ratio"))
    base = 9.5 if status == "Fresh Breakout" else 8.5 if "<=3M" in status else 6.0 if status == "Older Breakout" else max(2.5, 5 + (breakout if pd.notna(breakout) else -2) / 3)
    if pd.notna(vr): base += float(np.clip(vr - 1, -1, 1)) * .5
    return float(np.clip(base, 0, 10))


def build_ranking(scan, monthlies, macro_score, fund_scores):
    if scan is None or scan.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = scan.copy()
    sector_rows = []
    for ind, g in d.groupby("Industry"):
        r3, r6 = [], []
        for sym in g["Symbol"]:
            m = monthlies.get(sym + ".NS", pd.DataFrame())
            if m.empty or "Close" not in m.columns: continue
            c = pd.to_numeric(m["Close"], errors="coerce").dropna()
            if len(c) > 6:
                r3.append((c.iloc[-1] / c.iloc[-4] - 1) * 100)
                r6.append((c.iloc[-1] / c.iloc[-7] - 1) * 100)
        m3 = float(np.median(r3)) if r3 else np.nan
        m6 = float(np.median(r6)) if r6 else np.nan
        recent = int((g["Months Since Signal"].fillna(999) <= 3).sum())
        breadth = 100 * recent / max(1, len(g))
        mom = np.mean([5 + 5*np.tanh(m3/10) if pd.notna(m3) else 5, 5 + 5*np.tanh(m6/18) if pd.notna(m6) else 5])
        sec_score = float(np.clip(.75*mom + .25*(breadth/10), 0, 10))
        sector_rows.append({"Industry": ind, "3M Momentum %": m3, "6M Momentum %": m6, "Breadth %": breadth, "Sector Score": sec_score})
    sectors = pd.DataFrame(sector_rows).sort_values("Sector Score", ascending=False)
    smap = sectors.set_index("Industry")["Sector Score"].to_dict() if not sectors.empty else {}
    d["Technical Score"] = d.apply(technical_score, axis=1)
    d["Sector Score"] = d["Industry"].map(smap).fillna(5.0)
    d["Macro Score"] = macro_score
    d["Fundamental Score %"] = d["Symbol"].map(fund_scores)
    d["Final Score"] = d.apply(lambda r: (
        .45*r["Technical Score"] + .25*r["Sector Score"] + .15*r["Macro Score"] + .15*(r["Fundamental Score %"]/10)
        if pd.notna(r["Fundamental Score %"]) else
        (.45*r["Technical Score"] + .25*r["Sector Score"] + .15*r["Macro Score"]) / .85
    ), axis=1).clip(0, 10)
    d["Data Coverage"] = np.where(d["Fundamental Score %"].notna(), "4/4", "3/4")
    d["Opportunity"] = np.where(d["Final Score"] >= 8, "HIGH", np.where(d["Final Score"] >= 6.5, "WATCH", "LOW"))
    return d.sort_values(["Final Score", "Signal Date"], ascending=[False, False]), sectors


st.title("⭐ Final Opportunities — Professional Ranking")
st.caption("Availability-aware ranking: Macro → Sector → strict 26M ATH technical confirmation → Fundamentals. Missing fundamentals are excluded and remaining weights are normalized.")

mscore, regime, short_regime, coverage, macro = macro_state()
scan = st.session_state.get("pro_scan", pd.DataFrame())
monthlies = st.session_state.get("pro_monthlies", {})
fund_scores = dict(st.session_state.get("pro_final_fund_scores", {}))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Macro Score", f"{mscore:.1f}/10")
c2.metric("Macro Regime", short_regime, help=regime)
c3.metric("Macro Coverage", f"{coverage}%")
c4.metric("Scanned Stocks", len(scan))
st.caption(f"Full regime: **{regime}** · Macro feed TTL: 20 minutes · Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

if scan.empty:
    st.warning("पहले **Professional Research Lab → 26M ATH Scanner** में scan चलाएँ। यह Final Opportunities page उसी verified scan को rank करता है.")
    st.stop()

ranking, sectors = build_ranking(scan, monthlies, mscore, fund_scores)

a, b = st.columns([2, 1])
with a:
    st.subheader("🏆 Top Opportunities")
    st.caption("Base ranking uses Technical 45% + Sector 25% + Macro 15%. When fundamentals are available, Fundamentals add 15%; otherwise weights are normalized rather than assuming a neutral score.")
with b:
    nfund = st.selectbox("Fundamental batch", [5, 10, 20, 30], index=1)
    runfund = st.button(f"🧪 Scan fundamentals — Top {nfund}", type="primary", use_container_width=True)

if runfund:
    targets = ranking.head(int(nfund))
    prog = st.progress(0, text="Fetching fundamentals…")
    for i, (_, r) in enumerate(targets.iterrows()):
        sym = str(r["Symbol"])
        try:
            fs, assessed, pe = fundamental_quality(sym + ".NS")
            fund_scores[sym] = fs
        except Exception:
            fund_scores[sym] = np.nan
        prog.progress((i+1)/len(targets), text=f"Fundamentals {i+1}/{len(targets)}: {sym}")
    prog.empty()
    st.session_state["pro_final_fund_scores"] = fund_scores
    st.rerun()

ranking, sectors = build_ranking(scan, monthlies, mscore, fund_scores)

k1, k2, k3, k4 = st.columns(4)
k1.metric("High Opportunity", int((ranking["Opportunity"] == "HIGH").sum()))
k2.metric("Watch", int((ranking["Opportunity"] == "WATCH").sum()))
k3.metric("Fresh / <=3M", int((ranking["Months Since Signal"].fillna(999) <= 3).sum()))
k4.metric("Fundamentals Covered", f"{ranking['Fundamental Score %'].notna().sum()}/{len(ranking)}")

display_cols = ["Symbol", "Company", "Industry", "Opportunity", "Status", "Months Gap", "Breakout %",
                "Technical Score", "Sector Score", "Macro Score", "Fundamental Score %", "Data Coverage", "Final Score"]
st.dataframe(ranking[display_cols].head(50), use_container_width=True, hide_index=True,
             column_config={
                 "Technical Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
                 "Sector Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
                 "Macro Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
                 "Fundamental Score %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                 "Final Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.2f"),
             })

with st.expander("🔄 Sector leadership used in ranking", expanded=False):
    st.dataframe(sectors, use_container_width=True, hide_index=True,
                 column_config={"Sector Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
                                "Breadth %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})

with st.expander("🔎 Macro driver attribution", expanded=False):
    st.dataframe(macro.sort_values("Contribution", ascending=False), use_container_width=True, hide_index=True)

st.info("Final Score is a research prioritisation score, not a price target or buy/sell signal. Verify NSE/BSE prices, corporate filings, promoter/shareholding and valuation before acting.")
