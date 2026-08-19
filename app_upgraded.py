import io
import time
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from policy_stock_map import render_policy_stock_mapping

st.set_page_config(page_title="Macro + 26M ATH Breakout Dashboard", page_icon="📈", layout="wide")

APP_TITLE = "Macro + 26M ATH Breakout Dashboard"
NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.metric-card {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:14px;}
.small-note {font-size:.86rem; opacity:.75;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=86400, show_spinner=False)
def load_nifty500():
    last_err = None
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in NIFTY500_URLS:
        try:
            r = requests.get(url, timeout=15, headers=headers)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # Common columns: Company Name, Industry, Symbol, Series, ISIN Code
            if "Symbol" not in df.columns:
                continue
            df["Ticker"] = df["Symbol"].astype(str).str.strip() + ".NS"
            if "Industry" not in df.columns:
                df["Industry"] = "Unknown"
            if "Company Name" not in df.columns:
                df["Company Name"] = df["Symbol"]
            return df[["Company Name", "Industry", "Symbol", "Ticker"]].drop_duplicates("Ticker")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"NIFTY 500 constituent list fetch failed: {last_err}")

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, period="max"):
    if not tickers:
        return pd.DataFrame()
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

def extract_one(raw, ticker, n_tickers):
    try:
        if n_tickers == 1:
            df = raw.copy()
        else:
            df = raw[ticker].copy()
        df = df[[c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]]
        df = df.dropna(how="all")
        if "Close" not in df.columns or "High" not in df.columns:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def to_monthly(df):
    if df.empty:
        return df
    x = df.copy()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    agg = {"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}
    agg = {k:v for k,v in agg.items() if k in x.columns}
    m = x.resample("ME").agg(agg).dropna(subset=["Close", "High"])
    return m

def month_diff(later, earlier):
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)

def detect_breakouts(monthly, min_gap=26, lookback_signal_months=12):
    """Breakout = monthly CLOSE > all prior monthly HIGHS, and prior ATH high month is >= min_gap months old."""
    if monthly is None or len(monthly) < min_gap + 2:
        return None
    m = monthly.copy().dropna(subset=["High", "Close"])
    if len(m) < min_gap + 2:
        return None

    signals = []
    highs = m["High"].astype(float)
    closes = m["Close"].astype(float)
    vols = m["Volume"].astype(float) if "Volume" in m.columns else pd.Series(index=m.index, dtype=float)

    for i in range(1, len(m)):
        prior = highs.iloc[:i]
        old_ath = float(prior.max())
        ath_dates = prior.index[np.isclose(prior.values, old_ath, rtol=1e-10, atol=1e-10)]
        old_ath_date = ath_dates[-1] if len(ath_dates) else prior.idxmax()
        current_date = m.index[i]
        gap = month_diff(current_date, old_ath_date)
        close = float(closes.iloc[i])
        if close > old_ath and gap >= min_gap:
            avg12 = float(vols.iloc[max(0, i-12):i].mean()) if len(vols) and i > 0 else np.nan
            curv = float(vols.iloc[i]) if len(vols) else np.nan
            vr = curv / avg12 if avg12 and not np.isnan(avg12) and avg12 > 0 else np.nan
            signals.append({
                "Signal Date": current_date,
                "Old ATH": old_ath,
                "ATH Date": old_ath_date,
                "Months Gap": gap,
                "Monthly Close": close,
                "Breakout %": (close / old_ath - 1) * 100,
                "Volume Ratio": vr,
            })

    if not signals:
        # Near breakout using latest month versus all prior highs
        i = len(m)-1
        prior = highs.iloc[:i]
        if prior.empty:
            return None
        old_ath = float(prior.max())
        old_ath_date = prior.idxmax()
        gap = month_diff(m.index[i], old_ath_date)
        close = float(closes.iloc[i])
        return {
            "Status": "Near/No Breakout",
            "Signal Date": pd.NaT,
            "Old ATH": old_ath,
            "ATH Date": old_ath_date,
            "Months Gap": gap,
            "Monthly Close": close,
            "Breakout %": (close / old_ath - 1) * 100,
            "Volume Ratio": np.nan,
            "Months Since Signal": np.nan,
        }

    latest = signals[-1]
    latest_month = m.index[-1]
    ms = month_diff(latest_month, latest["Signal Date"])
    latest["Months Since Signal"] = ms
    if ms == 0:
        latest["Status"] = "Fresh Breakout"
    elif ms <= 3:
        latest["Status"] = "Breakout ≤3M"
    elif ms <= lookback_signal_months:
        latest["Status"] = f"Breakout ≤{lookback_signal_months}M"
    else:
        latest["Status"] = "Old Breakout"
    return latest

@st.cache_data(ttl=3600, show_spinner=False)
def macro_snapshot():
    assets = {
        "NIFTY 50": "^NSEI",
        "India VIX": "^INDIAVIX",
        "USD/INR": "INR=X",
        "Brent Crude": "BZ=F",
        "Dollar Index": "DX-Y.NYB",
        "US 10Y Yield": "^TNX",
        "Gold": "GC=F",
    }
    rows = []
    for name, ticker in assets.items():
        try:
            d = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
            if d.empty:
                continue
            c = d["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:,0]
            c = c.dropna().astype(float)
            if len(c) < 30:
                continue
            last = float(c.iloc[-1])
            r1m = (last / float(c.iloc[max(0,len(c)-22)]) - 1)*100 if len(c)>22 else np.nan
            r3m = (last / float(c.iloc[max(0,len(c)-66)]) - 1)*100 if len(c)>66 else np.nan
            r12m = (last / float(c.iloc[max(0,len(c)-252)]) - 1)*100 if len(c)>252 else np.nan
            rows.append([name,ticker,last,r1m,r3m,r12m])
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["Indicator","Ticker","Latest","1M %","3M %","12M %"])

def macro_score(mdf):
    if mdf.empty:
        return 5.0, "Neutral"
    s = 0.0
    w = 0.0
    def val(name, col="3M %"):
        r = mdf.loc[mdf["Indicator"]==name, col]
        return float(r.iloc[0]) if len(r) else np.nan
    rules = [
        ("NIFTY 50", +1, 2.0),        # rising equities positive
        ("India VIX", -1, 1.5),       # falling volatility positive
        ("USD/INR", -1, 1.5),         # stable/strong INR positive
        ("Brent Crude", -1, 1.5),     # falling crude positive for India macro
        ("Dollar Index", -1, 1.0),     # falling DXY positive for EM liquidity
        ("US 10Y Yield", -1, 1.0),     # falling yields positive for risk assets
        ("Gold", +1, 0.5),             # small diversification signal, low weight
    ]
    for name, direction, weight in rules:
        x = val(name)
        if np.isnan(x):
            continue
        # tanh avoids one extreme move dominating
        contribution = np.tanh((direction*x)/5.0)
        s += contribution*weight
        w += weight
    if w == 0:
        return 5.0, "Neutral"
    normalized = 5 + 5*(s/w)
    normalized = float(np.clip(normalized,0,10))
    label = "Strong Positive" if normalized>=7.5 else "Positive" if normalized>=6 else "Neutral" if normalized>=4 else "Negative" if normalized>=2.5 else "Strong Negative"
    return normalized, label


def _first_existing(df, labels):
    """Return the first matching statement row as a numeric Series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for label in labels:
        if label in df.index:
            x = pd.to_numeric(df.loc[label], errors="coerce").dropna()
            if not x.empty:
                x.index = pd.to_datetime(x.index)
                return x.sort_index()
    return pd.Series(dtype=float)

def _latest(series):
    return float(series.iloc[-1]) if series is not None and len(series) else np.nan

def _cagr(series, min_years=5):
    if series is None or len(series) < 2:
        return np.nan, 0
    x = series.dropna().sort_index()
    if len(x) < 2 or x.iloc[0] <= 0 or x.iloc[-1] <= 0:
        return np.nan, 0
    years = (x.index[-1] - x.index[0]).days / 365.25
    if years <= 0:
        return np.nan, 0
    return (float(x.iloc[-1] / x.iloc[0]) ** (1 / years) - 1) * 100, years

def _fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}%"

def _fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}"

def _status(condition, available=True):
    if not available:
        return "⚪ Data unavailable"
    return "🟢 PASS" if bool(condition) else "🔴 FAIL"

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetch/derive fundamentals from Yahoo Finance. Missing fields remain explicit N/A."""
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

    revenue = _first_existing(inc, ["Total Revenue", "Operating Revenue"])
    net_income = _first_existing(inc, ["Net Income", "Net Income Common Stockholders"])
    ebit = _first_existing(inc, ["EBIT", "Operating Income"])
    tax = _first_existing(inc, ["Tax Provision"])
    pretax = _first_existing(inc, ["Pretax Income"])

    total_assets = _first_existing(bs, ["Total Assets"])
    current_liab = _first_existing(bs, ["Current Liabilities", "Total Current Liabilities"])
    equity = _first_existing(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    debt = _first_existing(bs, ["Total Debt"])
    cash = _first_existing(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])

    total_debt = info.get("totalDebt", np.nan)
    total_equity = info.get("bookValue", np.nan)
    shares = info.get("sharesOutstanding", np.nan)
    if pd.notna(total_equity) and pd.notna(shares):
        total_equity = float(total_equity) * float(shares)
    else:
        total_equity = _latest(equity)
    if pd.isna(total_debt):
        total_debt = _latest(debt)
    de = float(total_debt / total_equity) if pd.notna(total_debt) and pd.notna(total_equity) and total_equity else np.nan

    latest_ebit = _latest(ebit)
    capital_employed = np.nan
    if len(total_assets) and len(current_liab):
        common = total_assets.index.intersection(current_liab.index)
        if len(common):
            capital_employed = float(total_assets.loc[common[-1]] - current_liab.loc[common[-1]])
    roce = latest_ebit / capital_employed * 100 if pd.notna(latest_ebit) and pd.notna(capital_employed) and capital_employed else np.nan

    latest_tax = _latest(tax)
    latest_pretax = _latest(pretax)
    tax_rate = latest_tax / latest_pretax if pd.notna(latest_tax) and pd.notna(latest_pretax) and latest_pretax > 0 else 0.25
    tax_rate = float(np.clip(tax_rate, 0, 0.5))
    nopat = latest_ebit * (1-tax_rate) if pd.notna(latest_ebit) else np.nan
    invested_capital = np.nan
    latest_eq, latest_debt, latest_cash = _latest(equity), _latest(debt), _latest(cash)
    if pd.notna(latest_eq) and pd.notna(latest_debt):
        invested_capital = latest_eq + latest_debt - (latest_cash if pd.notna(latest_cash) else 0)
    roc = nopat / invested_capital * 100 if pd.notna(nopat) and pd.notna(invested_capital) and invested_capital else np.nan

    sales_cagr, sales_years = _cagr(revenue)
    profit_cagr, profit_years = _cagr(net_income)
    ni = net_income.dropna().sort_index()
    profit_increasing = bool(len(ni) >= 3 and all(ni.iloc[i] >= ni.iloc[i-1] for i in range(1, len(ni))))

    trailing_pe = info.get("trailingPE", np.nan)
    return {
        "Company": info.get("longName", ticker),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Debt/Equity": de,
        "ROCE %": roce,
        "ROC %": roc,
        "Sales CAGR %": sales_cagr,
        "Sales CAGR Years": sales_years,
        "Profit CAGR %": profit_cagr,
        "Profit CAGR Years": profit_years,
        "Net Profit Increasing": profit_increasing if len(ni) >= 3 else None,
        "Profit Years Available": len(ni),
        "Stock PE": trailing_pe,
        "Promoter Trend": None,
    }

@st.cache_data(ttl=21600, show_spinner=False)
def peer_sector_pe(ticker, industry_name, universe_records, max_peers=12):
    """Approximate sector/industry PE with median trailing PE of NIFTY 500 peers."""
    peers = [r["Ticker"] for r in universe_records if r.get("Industry") == industry_name and r.get("Ticker") != ticker][:max_peers]
    pes = []
    for pt in peers:
        try:
            pe = (yf.Ticker(pt).info or {}).get("trailingPE")
            if pe is not None and np.isfinite(float(pe)) and 0 < float(pe) < 300:
                pes.append(float(pe))
        except Exception:
            pass
    return float(np.median(pes)) if pes else np.nan, len(pes)

def fundamental_scorecard(f, sector_pe=np.nan):
    rows = []
    checks = [
        ("Debt to Equity < 0.5", f.get("Debt/Equity"), pd.notna(f.get("Debt/Equity")), lambda x: x < 0.5, _fmt_num),
        ("ROCE > 15%", f.get("ROCE %"), pd.notna(f.get("ROCE %")), lambda x: x > 15, _fmt_pct),
        ("ROC > 15%", f.get("ROC %"), pd.notna(f.get("ROC %")), lambda x: x > 15, _fmt_pct),
        ("Net profit increasing", f.get("Net Profit Increasing"), f.get("Net Profit Increasing") is not None, lambda x: x is True, lambda x: "Yes" if x is True else "No"),
        ("Sales CAGR > 20% for ≥5 years", f.get("Sales CAGR %"), pd.notna(f.get("Sales CAGR %")) and f.get("Sales CAGR Years",0) >= 5, lambda x: x > 20, _fmt_pct),
        ("Profit CAGR > 15% for ≥5 years", f.get("Profit CAGR %"), pd.notna(f.get("Profit CAGR %")) and f.get("Profit CAGR Years",0) >= 5, lambda x: x > 15, _fmt_pct),
        ("Stock PE < sector/industry PE", f.get("Stock PE"), pd.notna(f.get("Stock PE")) and pd.notna(sector_pe), lambda x: x < sector_pe, _fmt_num),
        ("Promoter holding stable/increasing", f.get("Promoter Trend"), False, lambda x: False, lambda x: "N/A"),
    ]
    assessed_pass = assessed_total = 0
    for name, value, available, rule, formatter in checks:
        passed = bool(rule(value)) if available else False
        if available:
            assessed_total += 1
            assessed_pass += int(passed)
        note = ""
        if "Sales CAGR" in name:
            note = f"History: {f.get('Sales CAGR Years',0):.1f} years"
        elif "Profit CAGR" in name:
            note = f"History: {f.get('Profit CAGR Years',0):.1f} years"
        elif name == "Net profit increasing":
            note = f"Annual periods available: {f.get('Profit Years Available',0)}"
        elif "sector/industry PE" in name:
            note = "Peer median from NIFTY 500 industry" if pd.notna(sector_pe) else "Peer PE unavailable"
        elif "Promoter" in name:
            note = "Yahoo Finance does not provide reliable historical Indian promoter-holding trend"
        rows.append({"Criteria": name, "Value": formatter(value), "Status": _status(passed, available), "Note": note})
    score = 100 * assessed_pass / assessed_total if assessed_total else np.nan
    return pd.DataFrame(rows), score, assessed_pass, assessed_total


def make_chart(monthly, signal=None, name="Stock"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=monthly.index, open=monthly["Open"], high=monthly["High"], low=monthly["Low"], close=monthly["Close"], name="Monthly"
    ))
    if signal and pd.notna(signal.get("Old ATH", np.nan)):
        fig.add_hline(y=float(signal["Old ATH"]), line_dash="dash", annotation_text=f"Old ATH {signal['Old ATH']:.2f}")
    if signal and pd.notna(signal.get("Signal Date", pd.NaT)):
        fig.add_vline(x=signal["Signal Date"].timestamp()*1000, line_dash="dot", annotation_text="Breakout")
    fig.update_layout(title=f"{name} — Monthly Chart", xaxis_rangeslider_visible=False, height=520, margin=dict(l=10,r=10,t=50,b=10))
    return fig

st.title(APP_TITLE)
st.caption("Rule: Monthly Close > all prior monthly Highs (old ATH) AND old ATH must be at least 26 months old.")

with st.sidebar:
    st.header("Scanner Settings")
    min_gap = st.number_input("Minimum ATH gap (months)", min_value=12, max_value=120, value=26, step=1)
    signal_window = st.selectbox("Show recent breakouts", [1,3,6,12,24], index=3, format_func=lambda x: f"Last {x} months")
    near_pct = st.slider("Near ATH range", 1, 10, 5, 1)
    batch_size = st.selectbox("Scan size", [50,100,200,500], index=1, help="500 may take longer because history for many stocks must be downloaded.")
    st.divider()
    st.caption("Data: NIFTY Indices constituent list + Yahoo Finance price history. For research use; verify before trading.")

# Macro panel
st.subheader("🌍 Macro Pulse")
macro = macro_snapshot()
mscore, mlabel = macro_score(macro)
mc1, mc2, mc3 = st.columns([1,1,3])
mc1.metric("Macro Score", f"{mscore:.1f}/10")
mc2.metric("Regime", mlabel)
if not macro.empty:
    mc3.dataframe(macro.style.format({"Latest":"{:.2f}","1M %":"{:+.2f}%","3M %":"{:+.2f}%","12M %":"{:+.2f}%"}), use_container_width=True, hide_index=True)
else:
    mc3.warning("Macro market data unavailable right now. Scanner can still run.")

st.divider()
st.subheader("📈 NIFTY 500 — 26M ATH Breakout Scanner")

try:
    universe = load_nifty500()
except Exception as e:
    st.error(str(e))
    st.stop()

industry_options = sorted(universe["Industry"].dropna().unique().tolist())
selected_industries = st.multiselect("Industry filter (optional)", industry_options)
uf = universe[universe["Industry"].isin(selected_industries)] if selected_industries else universe.copy()
uf = uf.head(int(batch_size))

run = st.button(f"Run scan on {len(uf)} stocks", type="primary", use_container_width=True)

if run:
    tickers = uf["Ticker"].tolist()
    progress = st.progress(0, text="Downloading price history…")
    raw = download_prices(tickers, period="max")
    rows = []
    monthlies = {}
    for j, row in uf.reset_index(drop=True).iterrows():
        ticker = row["Ticker"]
        d = extract_one(raw, ticker, len(tickers))
        m = to_monthly(d)
        monthlies[ticker] = m
        sig = detect_breakouts(m, int(min_gap), int(signal_window))
        if sig:
            item = {
                "Symbol": row["Symbol"],
                "Company": row["Company Name"],
                "Industry": row["Industry"],
                **sig,
            }
            # Macro combined score emphasizes technical confirmation but avoids pretending macro is stock-specific.
            tech = 10 if sig["Status"] == "Fresh Breakout" else 8.5 if str(sig["Status"]).startswith("Breakout") else max(0, min(10, 5 + sig["Breakout %"]))
            vol_bonus = 0 if pd.isna(sig.get("Volume Ratio", np.nan)) else min(1.0, max(-1.0, sig["Volume Ratio"]-1))
            combined = 0.65*tech + 0.35*mscore + 0.25*vol_bonus
            item["Combined Score"] = float(np.clip(combined,0,10))
            rows.append(item)
        progress.progress((j+1)/len(uf), text=f"Scanning {j+1}/{len(uf)}: {row['Symbol']}")
    progress.empty()
    res = pd.DataFrame(rows)
    st.session_state["scan_results"] = res
    st.session_state["monthlies"] = monthlies
    st.session_state["scan_universe"] = uf

res = st.session_state.get("scan_results", pd.DataFrame())
monthlies = st.session_state.get("monthlies", {})

if not res.empty:
    recent = res[res["Months Since Signal"].fillna(999) <= signal_window].copy()
    near = res[(res["Status"]=="Near/No Breakout") & (res["Breakout %"] >= -near_pct) & (res["Breakout %"] < 0) & (res["Months Gap"] >= min_gap)].copy()

    a,b,c,d = st.columns(4)
    a.metric("Scanned", len(st.session_state.get("scan_universe", [])))
    b.metric("Recent Breakouts", len(recent))
    c.metric(f"Near ATH (≤{near_pct}%)", len(near))
    d.metric("Macro Score", f"{mscore:.1f}/10")

    tabs = st.tabs(["Confirmed Breakouts", "Near Breakout Watchlist", "All Scan Results", "Stock Chart + Fundamentals"])
    with tabs[0]:
        show = recent.sort_values(["Signal Date","Combined Score"], ascending=[False,False]).copy()
        if show.empty:
            st.info("No stocks in the selected scan have a qualifying breakout in this window.")
        else:
            fmt_cols = ["Old ATH","Monthly Close","Breakout %","Volume Ratio","Combined Score"]
            st.dataframe(show, use_container_width=True, hide_index=True, column_config={
                "Signal Date": st.column_config.DateColumn(format="MMM YYYY"),
                "ATH Date": st.column_config.DateColumn(format="MMM YYYY"),
                "Breakout %": st.column_config.NumberColumn(format="%.2f%%"),
                "Volume Ratio": st.column_config.NumberColumn(format="%.2fx"),
                "Combined Score": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            })
            st.download_button("Download confirmed breakouts CSV", show.to_csv(index=False).encode("utf-8"), "confirmed_26m_ath_breakouts.csv", "text/csv")

    with tabs[1]:
        if near.empty:
            st.info("No near-breakout candidates found with current filters.")
        else:
            near = near.sort_values("Breakout %", ascending=False)
            st.dataframe(near, use_container_width=True, hide_index=True, column_config={
                "Breakout %": st.column_config.NumberColumn("Distance vs old ATH", format="%.2f%%"),
                "ATH Date": st.column_config.DateColumn(format="MMM YYYY"),
                "Combined Score": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            })
            st.download_button("Download near-breakout CSV", near.to_csv(index=False).encode("utf-8"), "near_26m_ath_watchlist.csv", "text/csv")

    with tabs[2]:
        st.dataframe(res.sort_values("Combined Score", ascending=False), use_container_width=True, hide_index=True)

    with tabs[3]:
        choices = sorted(res["Symbol"].unique().tolist())
        sym = st.selectbox("Choose stock", choices)
        rr = res[res["Symbol"]==sym].iloc[0].to_dict()
        ticker = sym + ".NS"
        m = monthlies.get(ticker, pd.DataFrame())
        if not m.empty:
            st.plotly_chart(make_chart(m, rr, sym), use_container_width=True)
            st.write(f"**Old ATH:** ₹{rr['Old ATH']:.2f}  |  **ATH month:** {pd.to_datetime(rr['ATH Date']).strftime('%b %Y')}  |  **Gap:** {int(rr['Months Gap'])} months")
            if pd.notna(rr.get("Signal Date")):
                st.success(f"Breakout confirmed by monthly close in {pd.to_datetime(rr['Signal Date']).strftime('%b %Y')} at ₹{rr['Monthly Close']:.2f} ({rr['Breakout %']:+.2f}% above old ATH).")
            else:
                st.info(f"Latest monthly close is {rr['Breakout %']:.2f}% versus old ATH; not a confirmed breakout yet.")

        st.divider()
        st.subheader("🧾 Fundamental Analysis")
        st.caption("Criteria: D/E <0.5, ROCE & ROC >15%, rising profit, 5Y+ CAGR thresholds, PE below industry peers, promoter trend.")
        if st.button(f"Run Fundamental Analysis — {sym}", type="primary", key=f"fund_{sym}", use_container_width=True):
            with st.spinner(f"Fetching fundamentals for {sym}…"):
                f = fetch_fundamentals(ticker)
                industry_name = rr.get("Industry")
                uni_records = st.session_state.get("scan_universe", pd.DataFrame()).to_dict("records")
                sec_pe, peer_count = peer_sector_pe(ticker, industry_name, uni_records)
                card, fscore, passed, total = fundamental_scorecard(f, sec_pe)
                st.session_state[f"fund_result_{sym}"] = (f, sec_pe, peer_count, card, fscore, passed, total)

        result = st.session_state.get(f"fund_result_{sym}")
        if result:
            f, sec_pe, peer_count, card, fscore, passed, total = result
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Fundamental Score", "N/A" if pd.isna(fscore) else f"{fscore:.0f}%")
            c2.metric("Criteria Passed", f"{passed}/{total}")
            c3.metric("Stock PE", _fmt_num(f.get("Stock PE")))
            c4.metric("Industry Peer PE", _fmt_num(sec_pe), help=f"Median trailing PE from {peer_count} available NIFTY 500 peers in {rr.get('Industry','this industry')}")
            st.dataframe(card, use_container_width=True, hide_index=True)
            if total and passed == total:
                st.success("All currently assessable fundamental criteria pass.")
            elif total:
                st.warning("Some assessable fundamental criteria do not pass. Review the red rows before taking a decision.")
            st.caption("Promoter-holding trend is shown as unavailable when a reliable historical series is not supplied by Yahoo Finance. Verify it from NSE/BSE shareholding filings. 5-year CAGR rules are not marked PASS unless at least 5 years of statement history is available.")
else:
    st.info("Choose scan size and press **Run scan**. Start with 50 or 100 stocks; switch to 500 for the full NIFTY 500 universe.")

st.divider()
with st.expander("How the breakout logic works"):
    st.markdown("""
1. Daily history is converted to **monthly OHLC**.
2. For each month, the app finds the **highest High of all earlier months** = old ATH.
3. It finds the month in which that old ATH was made.
4. A signal is valid only when **current monthly Close > old ATH** and the gap from old ATH month is **≥ 26 months**.
5. Volume Ratio = breakout-month volume / previous 12-month average volume.
6. “Near Breakout” means price is below the old ATH but within your selected % range, while the ATH age condition is already satisfied.

**Important:** A live/current month can change before month-end. For strict confirmation, evaluate the signal after the monthly candle closes.
""")

st.caption("This dashboard is a research tool, not investment advice. Data may be delayed or occasionally unavailable; verify signals with your broker/exchange data before acting.")
