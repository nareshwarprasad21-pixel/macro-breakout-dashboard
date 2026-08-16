import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Professional Research Lab", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1600px;}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.20); border-radius:12px; padding:10px;}
@media (max-width: 900px) {
  .block-container {padding: 1.25rem .65rem 1.5rem !important;}
  [data-testid="stMetric"] {padding:7px;}
  .stDataFrame {overflow-x:auto;}
  h1 {font-size:1.7rem !important;}
  h2 {font-size:1.35rem !important;}
}
</style>
""",
    unsafe_allow_html=True,
)

NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]

MACRO_ASSETS = {
    "NIFTY 50": "^NSEI",
    "India VIX": "^INDIAVIX",
    "USD/INR": "INR=X",
    "Brent Crude": "BZ=F",
    "Dollar Index": "DX-Y.NYB",
    "US 10Y Yield": "^TNX",
    "Gold": "GC=F",
}


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def month_diff(later, earlier):
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


@st.cache_data(ttl=86400, show_spinner=False)
def load_nifty500():
    headers = {"User-Agent": "Mozilla/5.0"}
    errors = []
    for url in NIFTY500_URLS:
        try:
            r = requests.get(url, timeout=15, headers=headers)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if "Symbol" not in df.columns:
                continue
            if "Industry" not in df.columns:
                df["Industry"] = "Unknown"
            if "Company Name" not in df.columns:
                df["Company Name"] = df["Symbol"]
            df["Ticker"] = df["Symbol"].astype(str).str.strip() + ".NS"
            return df[["Company Name", "Industry", "Symbol", "Ticker"]].drop_duplicates("Ticker")
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("NIFTY 500 official constituent list unavailable: " + " | ".join(errors[-2:]))


@st.cache_data(ttl=1200, show_spinner=False)
def download_prices(tickers, period="2y", auto_adjust=False):
    if not tickers:
        return pd.DataFrame()
    kwargs = dict(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=auto_adjust,
        threads=True,
        progress=False,
    )
    try:
        return yf.download(timeout=25, **kwargs)
    except TypeError:
        return yf.download(**kwargs)


def extract_one(raw, ticker, total):
    try:
        if raw.empty:
            return pd.DataFrame()
        if total == 1 and not isinstance(raw.columns, pd.MultiIndex):
            d = raw.copy()
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
            d = raw[ticker].copy()
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(-1):
            d = raw.xs(ticker, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
        cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in d.columns]
        return d[cols].dropna(how="all")
    except Exception:
        return pd.DataFrame()


def to_monthly(d):
    if d is None or d.empty or "Close" not in d.columns or "High" not in d.columns:
        return pd.DataFrame()
    x = d.copy()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in x.columns}
    return x.resample("ME").agg(agg).dropna(subset=["High", "Close"])


def period_return(close, periods):
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) <= periods:
        return np.nan
    base = safe_float(c.iloc[-periods - 1])
    last = safe_float(c.iloc[-1])
    return (last / base - 1) * 100 if pd.notna(base) and base != 0 else np.nan


def strict_26m_signal(monthly, min_gap=26):
    """Monthly CLOSE must exceed every prior monthly HIGH; old ATH must be >= min_gap months old."""
    if monthly is None or len(monthly) < min_gap + 2:
        return None
    m = monthly.dropna(subset=["High", "Close"]).copy()
    if len(m) < min_gap + 2:
        return None
    highs = pd.to_numeric(m["High"], errors="coerce")
    closes = pd.to_numeric(m["Close"], errors="coerce")
    signals = []
    for i in range(1, len(m)):
        prior = highs.iloc[:i].dropna()
        if prior.empty:
            continue
        old_ath = float(prior.max())
        ath_dates = prior.index[np.isclose(prior.values, old_ath, rtol=1e-10, atol=1e-10)]
        old_date = ath_dates[-1] if len(ath_dates) else prior.idxmax()
        gap = month_diff(m.index[i], old_date)
        close = safe_float(closes.iloc[i])
        if pd.notna(close) and close > old_ath and gap >= min_gap:
            avg12 = safe_float(m["Volume"].iloc[max(0, i - 12):i].mean()) if "Volume" in m.columns else np.nan
            volume = safe_float(m["Volume"].iloc[i]) if "Volume" in m.columns else np.nan
            signals.append({
                "Signal Date": m.index[i],
                "Old ATH": old_ath,
                "ATH Date": old_date,
                "Months Gap": gap,
                "Monthly Close": close,
                "Breakout %": (close / old_ath - 1) * 100,
                "Volume Ratio": volume / avg12 if pd.notna(volume) and pd.notna(avg12) and avg12 > 0 else np.nan,
            })
    if signals:
        sig = signals[-1]
        age = month_diff(m.index[-1], sig["Signal Date"])
        sig["Months Since Signal"] = age
        sig["Status"] = "Fresh Breakout" if age == 0 else "Breakout <=3M" if age <= 3 else "Older Breakout"
        return sig
    prior = highs.iloc[:-1].dropna()
    if prior.empty:
        return None
    old_ath = float(prior.max())
    ath_dates = prior.index[np.isclose(prior.values, old_ath, rtol=1e-10, atol=1e-10)]
    old_date = ath_dates[-1] if len(ath_dates) else prior.idxmax()
    close = safe_float(closes.iloc[-1])
    return {
        "Status": "Near/No Breakout",
        "Signal Date": pd.NaT,
        "Old ATH": old_ath,
        "ATH Date": old_date,
        "Months Gap": month_diff(m.index[-1], old_date),
        "Monthly Close": close,
        "Breakout %": (close / old_ath - 1) * 100 if pd.notna(close) else np.nan,
        "Volume Ratio": np.nan,
        "Months Since Signal": np.nan,
    }


@st.cache_data(ttl=1200, show_spinner=False)
def macro_snapshot():
    tickers = list(MACRO_ASSETS.values())
    try:
        raw = download_prices(tickers, period="2y")
    except Exception:
        raw = pd.DataFrame()
    rows = []
    for name, ticker in MACRO_ASSETS.items():
        d = extract_one(raw, ticker, len(tickers))
        if d.empty or "Close" not in d.columns:
            continue
        c = pd.to_numeric(d["Close"], errors="coerce").dropna()
        if len(c) < 25:
            continue
        rows.append({
            "Indicator": name,
            "Ticker": ticker,
            "Latest": safe_float(c.iloc[-1]),
            "1M %": period_return(c, 21),
            "3M %": period_return(c, 63),
            "12M %": period_return(c, 252),
            "As Of": pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def macro_engine(mdf):
    """Availability-aware India risk/liquidity regime model with confidence and driver table."""
    if mdf is None or mdf.empty:
        return 5.0, "NEUTRAL / LOW DATA", 0, pd.DataFrame(), "Insufficient live market inputs"

    specs = [
        ("NIFTY 50", +1, 2.2, 8.0, "Domestic risk appetite"),
        ("India VIX", -1, 1.7, 15.0, "Volatility / risk aversion"),
        ("USD/INR", -1, 1.5, 4.0, "INR pressure / imported inflation"),
        ("Brent Crude", -1, 1.5, 12.0, "India import-cost pressure"),
        ("Dollar Index", -1, 1.0, 6.0, "Global USD liquidity"),
        ("US 10Y Yield", -1, 1.0, 8.0, "Global discount-rate pressure"),
        ("Gold", -1, 0.6, 12.0, "Defensive / uncertainty proxy"),
    ]
    weighted = 0.0
    weight_sum = 0.0
    drivers = []
    for name, direction, weight, scale, meaning in specs:
        row = mdf[mdf["Indicator"] == name]
        if row.empty:
            continue
        r3 = safe_float(row.iloc[0]["3M %"])
        r1 = safe_float(row.iloc[0]["1M %"])
        if pd.isna(r3):
            continue
        impulse = 0.75 * r3 + 0.25 * (r1 if pd.notna(r1) else r3)
        raw = float(np.tanh(direction * impulse / scale))
        weighted += raw * weight
        weight_sum += weight
        drivers.append({
            "Driver": name,
            "Meaning": meaning,
            "1M %": r1,
            "3M %": r3,
            "Contribution": raw * weight,
            "Signal": "Supportive" if raw > 0.15 else "Adverse" if raw < -0.15 else "Neutral",
        })
    if weight_sum == 0:
        return 5.0, "NEUTRAL / LOW DATA", 0, pd.DataFrame(drivers), "No evaluable macro inputs"
    score = float(np.clip(5 + 5 * weighted / weight_sum, 0, 10))
    coverage = int(round(100 * weight_sum / sum(s[2] for s in specs)))

    def val(name, col="3M %"):
        r = mdf[mdf["Indicator"] == name]
        return safe_float(r.iloc[0][col]) if not r.empty else np.nan

    nifty = val("NIFTY 50")
    vix_level = safe_float(mdf.loc[mdf["Indicator"] == "India VIX", "Latest"].iloc[0]) if (mdf["Indicator"] == "India VIX").any() else np.nan
    crude = val("Brent Crude")
    usd_inr = val("USD/INR")

    inflation_stress = sum([
        1 if pd.notna(crude) and crude > 10 else 0,
        1 if pd.notna(usd_inr) and usd_inr > 3 else 0,
    ])
    risk_stress = sum([
        1 if pd.notna(nifty) and nifty < -5 else 0,
        1 if pd.notna(vix_level) and vix_level > 20 else 0,
    ])

    if score >= 6.8 and (pd.isna(nifty) or nifty > 0) and risk_stress == 0:
        regime = "EARLY / RISK-ON" if score >= 7.6 else "MID CYCLE / EXPANSION"
    elif inflation_stress >= 1 and score >= 3.8:
        regime = "LATE CYCLE / INFLATION-SENSITIVE"
    elif score < 3.8 or risk_stress >= 2:
        regime = "RISK-OFF / CONTRACTION"
    else:
        regime = "MID-TO-LATE / MIXED"

    note = f"{coverage}% weighted input coverage; regime uses market/liquidity proxies, not an official GDP-cycle dating model."
    return score, regime, coverage, pd.DataFrame(drivers).sort_values("Contribution", ascending=False), note


def sector_table(results, monthlies, benchmark_monthly):
    if results is None or results.empty:
        return pd.DataFrame()
    bm6 = period_return(benchmark_monthly["Close"], 6) if not benchmark_monthly.empty else np.nan
    rows = []
    for industry, group in results.groupby("Industry"):
        r3s, r6s, r12s = [], [], []
        for sym in group["Symbol"].head(40):
            m = monthlies.get(sym + ".NS", pd.DataFrame())
            if m.empty:
                continue
            r3s.append(period_return(m["Close"], 3))
            r6s.append(period_return(m["Close"], 6))
            r12s.append(period_return(m["Close"], 12))
        clean = lambda xs: [x for x in xs if pd.notna(x)]
        r3s, r6s, r12s = clean(r3s), clean(r6s), clean(r12s)
        m3 = float(np.median(r3s)) if r3s else np.nan
        m6 = float(np.median(r6s)) if r6s else np.nan
        m12 = float(np.median(r12s)) if r12s else np.nan
        rs6 = m6 - bm6 if pd.notna(m6) and pd.notna(bm6) else np.nan
        recent = int((group["Months Since Signal"].fillna(999) <= 3).sum())
        near = int(((group["Status"] == "Near/No Breakout") & (group["Breakout %"] >= -5)).sum())
        breadth = 100 * (recent + 0.5 * near) / max(1, len(group))
        components = []
        for value, scale, weight in [(m3, 8, .25), (m6, 12, .30), (m12, 22, .20), (rs6, 10, .25)]:
            if pd.notna(value):
                components.append((np.tanh(value / scale), weight))
        mom = sum(v * w for v, w in components) / sum(w for _, w in components) if components else 0
        score = float(np.clip(5 + 3.4 * mom + min(1.8, breadth / 30), 0, 10))
        quadrant = "LEADING" if pd.notna(rs6) and rs6 >= 0 and pd.notna(m3) and m3 >= 0 else "IMPROVING" if pd.notna(m3) and m3 >= 0 else "WEAKENING" if pd.notna(rs6) and rs6 >= 0 else "LAGGING"
        rows.append({
            "Industry": industry,
            "3M Momentum %": m3,
            "6M Momentum %": m6,
            "12M Momentum %": m12,
            "6M RS vs NIFTY %": rs6,
            "Breakout Breadth %": breadth,
            "Sector Score": score,
            "Rotation Quadrant": quadrant,
            "Stocks Evaluated": len(group),
        })
    return pd.DataFrame(rows).sort_values("Sector Score", ascending=False)


def rotation_chart(df):
    fig = go.Figure()
    if df is None or df.empty:
        return fig
    d = df.dropna(subset=["6M RS vs NIFTY %", "3M Momentum %"]).copy()
    for q in ["LEADING", "IMPROVING", "WEAKENING", "LAGGING"]:
        x = d[d["Rotation Quadrant"] == q]
        if x.empty:
            continue
        fig.add_trace(go.Scatter(
            x=x["6M RS vs NIFTY %"], y=x["3M Momentum %"], mode="markers+text",
            text=x["Industry"], textposition="top center", name=q,
            marker={"size": 14 + 4 * x["Sector Score"].clip(0, 10), "opacity": .72},
            customdata=np.stack([x["Sector Score"], x["Breakout Breadth %"]], axis=-1),
            hovertemplate="<b>%{text}</b><br>RS6: %{x:.2f}%<br>M3: %{y:.2f}%<br>Score: %{customdata[0]:.1f}/10<br>Breadth: %{customdata[1]:.1f}%<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=560, xaxis_title="6M Relative Strength vs NIFTY (%)", yaxis_title="3M Momentum (%)", margin=dict(l=10, r=10, t=35, b=10))
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def fundamental_snapshot(ticker):
    stock = yf.Ticker(ticker)
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    try:
        inc = stock.financials.copy()
    except Exception:
        inc = pd.DataFrame()
    try:
        bs = stock.balance_sheet.copy()
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
        return ((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else np.nan, years

    sales_cagr, sales_years = cagr(revenue)
    profit_cagr, profit_years = cagr(profit)
    eq = safe_float(equity.iloc[-1]) if len(equity) else np.nan
    total_debt = safe_float(info.get("totalDebt"))
    if pd.isna(total_debt) and len(debt):
        total_debt = safe_float(debt.iloc[-1])
    de = total_debt / eq if pd.notna(total_debt) and pd.notna(eq) and eq != 0 else np.nan
    latest_ebit = safe_float(ebit.iloc[-1]) if len(ebit) else np.nan
    cap = np.nan
    common = assets.index.intersection(current_liab.index)
    if len(common):
        cap = safe_float(assets.loc[common[-1]] - current_liab.loc[common[-1]])
    roce = latest_ebit / cap * 100 if pd.notna(latest_ebit) and pd.notna(cap) and cap != 0 else np.nan
    p = profit.dropna().sort_index()
    increasing = bool(len(p) >= 3 and all(p.iloc[i] >= p.iloc[i - 1] for i in range(1, len(p)))) if len(p) >= 3 else None
    pe = safe_float(info.get("trailingPE"))

    checks = [
        ("Debt / Equity < 0.5", de, pd.notna(de), lambda x: x < .5),
        ("ROCE > 15%", roce, pd.notna(roce), lambda x: x > 15),
        ("Net profit increasing", increasing, increasing is not None, lambda x: x is True),
        ("Sales CAGR > 20% for >=5Y", sales_cagr, pd.notna(sales_cagr) and sales_years >= 5, lambda x: x > 20),
        ("Profit CAGR > 15% for >=5Y", profit_cagr, pd.notna(profit_cagr) and profit_years >= 5, lambda x: x > 15),
    ]
    rows, passed, assessed = [], 0, 0
    for name, value, available, rule in checks:
        ok = bool(rule(value)) if available else None
        if ok is not None:
            assessed += 1
            passed += int(ok)
        rows.append({
            "Criterion": name,
            "Value": "N/A" if not available else ("Yes" if value is True else "No" if value is False else f"{value:.2f}"),
            "Result": "N/A" if ok is None else "PASS" if ok else "FAIL",
        })
    return {
        "Company": info.get("longName") or ticker,
        "Sector": info.get("sector") or "N/A",
        "Industry": info.get("industry") or "N/A",
        "PE": pe,
        "Score": 100 * passed / assessed if assessed else np.nan,
        "Passed": passed,
        "Assessed": assessed,
        "Checks": pd.DataFrame(rows),
        "Source": "Yahoo Finance quote + reported financial statements",
        "Retrieved": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


st.title("🧠 Professional Investment Research Lab")
st.caption("Live market diagnostics + transparent source coverage + macro regime + sector rotation + strict 26M ATH + fundamental quality. Research tool, not investment advice.")

macro = macro_snapshot()
score, regime, coverage, drivers, regime_note = macro_engine(macro)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Macro Score", f"{score:.1f}/10")
m2.metric("Macro Regime", regime)
m3.metric("Macro Coverage", f"{coverage}%")
m4.metric("Live Data TTL", "20 min")

with st.expander("🔎 Data provenance & freshness", expanded=True):
    st.markdown("**Universe:** official NIFTY Indices constituent CSV. **Market prices / macro / fundamentals:** Yahoo Finance endpoints via `yfinance`. These are convenient research feeds, not an exchange-certified real-time terminal.")
    if not macro.empty:
        newest = macro["As Of"].max()
        oldest = macro["As Of"].min()
        st.caption(f"Macro observation dates: {oldest} to {newest}. {regime_note}")
    st.warning("For order execution or compliance decisions, verify price, corporate filings and shareholding on NSE/BSE/company filings or your broker terminal.")

main_tabs = st.tabs(["🌍 Macro Regime", "📈 26M ATH Scanner", "🔄 Sector Rotation", "🧾 Fundamentals"])

with main_tabs[0]:
    st.subheader("Macro regime diagnostics")
    if macro.empty:
        st.warning("Macro feed unavailable right now.")
    else:
        st.dataframe(macro, use_container_width=True, hide_index=True, column_config={"As Of": st.column_config.TextColumn("As Of")})
        st.markdown("#### Driver attribution")
        st.dataframe(drivers, use_container_width=True, hide_index=True)
        supportive = drivers.sort_values("Contribution", ascending=False).head(2)["Driver"].tolist() if not drivers.empty else []
        adverse = drivers.sort_values("Contribution").head(2)["Driver"].tolist() if not drivers.empty else []
        st.info(f"Largest supportive drivers: {', '.join(supportive) or 'N/A'} · Largest adverse drivers: {', '.join(adverse) or 'N/A'}")

with main_tabs[1]:
    st.subheader("Strict monthly 26M ATH scanner")
    st.caption("Signal = monthly CLOSE above every prior monthly HIGH, and the old ATH month must be at least the selected gap old. Current unfinished month can change before month-end.")
    try:
        universe = load_nifty500()
    except Exception as exc:
        st.error(str(exc))
        universe = pd.DataFrame()

    if not universe.empty:
        c1, c2, c3 = st.columns(3)
        min_gap = c1.number_input("Minimum ATH gap", 12, 120, 26, 1)
        batch = c2.selectbox("Universe size", [50, 100, 200, 500], index=1)
        near_pct = c3.slider("Near ATH range %", 1, 10, 5, 1)
        industries = st.multiselect("Industry filter (optional)", sorted(universe["Industry"].dropna().unique()))
        scan_u = universe[universe["Industry"].isin(industries)] if industries else universe
        scan_u = scan_u.head(int(batch)).copy()

        if st.button(f"Run professional scan on {len(scan_u)} stocks", type="primary", use_container_width=True):
            tickers = scan_u["Ticker"].tolist()
            prog = st.progress(0, text="Downloading price history…")
            try:
                raw = download_prices(tickers, period="max")
            except Exception as exc:
                st.error(f"Price download failed: {exc}")
                raw = pd.DataFrame()
            records, monthlies = [], {}
            for i, (_, r) in enumerate(scan_u.iterrows()):
                d = extract_one(raw, r["Ticker"], len(tickers))
                m = to_monthly(d)
                monthlies[r["Ticker"]] = m
                sig = strict_26m_signal(m, int(min_gap))
                if sig:
                    records.append({"Symbol": r["Symbol"], "Company": r["Company Name"], "Industry": r["Industry"], **sig})
                prog.progress((i + 1) / max(1, len(scan_u)), text=f"Scanning {i+1}/{len(scan_u)}: {r['Symbol']}")
            prog.empty()
            st.session_state["pro_scan"] = pd.DataFrame(records)
            st.session_state["pro_monthlies"] = monthlies
            st.session_state["pro_scan_universe"] = scan_u
            st.session_state["pro_gap"] = int(min_gap)

        results = st.session_state.get("pro_scan", pd.DataFrame())
        if not results.empty:
            fresh = results[results["Months Since Signal"].fillna(999) <= 3]
            near = results[(results["Status"] == "Near/No Breakout") & (results["Breakout %"] >= -near_pct) & (results["Months Gap"] >= min_gap)]
            a, b, c, d = st.columns(4)
            a.metric("Records", len(results))
            b.metric("Fresh / <=3M", len(fresh))
            c.metric(f"Near ATH <= {near_pct}%", len(near))
            d.metric("Rule Gap", f">={st.session_state.get('pro_gap', min_gap)}M")
            show = results.sort_values(["Status", "Breakout %"], ascending=[True, False])
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.info("Run the scanner to generate current results.")

with main_tabs[2]:
    st.subheader("Sector rotation from the latest scan")
    results = st.session_state.get("pro_scan", pd.DataFrame())
    monthlies = st.session_state.get("pro_monthlies", {})
    if results.empty:
        st.info("Run the 26M ATH scanner first. Sector rotation is built from the same evaluated universe, so the inputs stay internally consistent.")
    else:
        try:
            bmraw = download_prices(["^NSEI"], period="2y")
            bmd = extract_one(bmraw, "^NSEI", 1)
            bm = to_monthly(bmd)
        except Exception:
            bm = pd.DataFrame()
        sectors = sector_table(results, monthlies, bm)
        st.plotly_chart(rotation_chart(sectors), use_container_width=True)
        st.dataframe(sectors, use_container_width=True, hide_index=True, column_config={
            "Sector Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
            "Breakout Breadth %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        })
        st.caption("LEADING = positive 6M relative strength vs NIFTY + positive 3M momentum; IMPROVING = momentum positive while RS is still weak; WEAKENING/LAGGING are the opposite combinations.")

with main_tabs[3]:
    st.subheader("Fundamental quality check")
    default = "LT"
    scan = st.session_state.get("pro_scan", pd.DataFrame())
    options = scan["Symbol"].dropna().astype(str).tolist() if not scan.empty else []
    if options:
        sym = st.selectbox("Choose scanned stock", options)
    else:
        sym = st.text_input("NSE Symbol", value=default).strip().upper()
    if st.button("Run fundamental analysis", type="primary", use_container_width=True):
        ticker = sym if sym.endswith(".NS") else sym + ".NS"
        with st.spinner("Fetching reported financials…"):
            try:
                fund = fundamental_snapshot(ticker)
                st.session_state["pro_fund"] = fund
            except Exception as exc:
                st.error(f"Fundamental fetch failed: {exc}")
    fund = st.session_state.get("pro_fund")
    if fund:
        a, b, c, d = st.columns(4)
        a.metric("Company", fund["Company"])
        b.metric("Quality Score", "N/A" if pd.isna(fund["Score"]) else f"{fund['Score']:.0f}%")
        c.metric("Passed", f"{fund['Passed']}/{fund['Assessed']}")
        d.metric("Trailing PE", "N/A" if pd.isna(fund["PE"]) else f"{fund['PE']:.1f}x")
        st.caption(f"{fund['Sector']} · {fund['Industry']} · Retrieved {fund['Retrieved']} · Source: {fund['Source']}")
        st.dataframe(fund["Checks"], use_container_width=True, hide_index=True)
        st.info("Promoter holding trend is intentionally not guessed here. Historical promoter/shareholding should be verified from NSE/BSE corporate filings before using it as a pass/fail rule.")

st.divider()
st.caption("Professional Research Lab v1 · Source-aware, availability-aware and mobile-responsive. Scores are research heuristics; they do not predict returns or replace exchange/company filings.")