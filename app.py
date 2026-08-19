import io
import time
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import yfinance as yf
from graham import fetch_graham_data
from policy_stock_map import render_policy_stock_mapping
from ui_polish import apply_professional_ui

st.set_page_config(page_title="Macro + 26M ATH Breakout Dashboard", page_icon="📈", layout="wide")
apply_professional_ui()

APP_TITLE = "India Macro + Policy + Sector Rotation + 26M ATH + Value Migration Professional Engine"
NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]

st.markdown("""
<style>
.block-container {padding-top: 5rem !important; padding-bottom: 2rem;}
[data-testid="stMainBlockContainer"] {padding-top: 5rem !important;}
@media (max-width: 900px) {
  .block-container, [data-testid="stMainBlockContainer"] {padding: 4.25rem .8rem 1.5rem !important;}
  [data-testid="stMetric"] {min-width: 0;}
  .stDataFrame {overflow-x: auto;}
}
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
    try:
        return yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
            timeout=20,
        )
    except TypeError:  # compatibility with older yfinance releases
        return yf.download(
            tickers=tickers, period=period, interval="1d", group_by="ticker",
            auto_adjust=False, threads=True, progress=False,
        )

def extract_one(raw, ticker, n_tickers):
    try:
        if n_tickers == 1 and not isinstance(raw.columns, pd.MultiIndex):
            df = raw.copy()
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
            df = raw[ticker].copy()
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(-1):
            df = raw.xs(ticker, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
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
        ath_dates = prior.index[np.isclose(prior.values, old_ath, rtol=1e-10, atol=1e-10)]
        old_ath_date = ath_dates[-1] if len(ath_dates) else prior.idxmax()
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
    try:
        raw = download_prices(list(assets.values()), period="2y")
    except Exception:
        raw = pd.DataFrame()
    for name, ticker in assets.items():
        try:
            d = extract_one(raw, ticker, len(assets))
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



@st.cache_data(ttl=86400, show_spinner=False)
def government_policy_opportunity_report():
    """
    Curated India policy/project snapshot.
    10Y Opportunity Score is a research heuristic (0-10), NOT a return forecast.
    Sources are official Government of India / PIB / ministry pages.
    Research refresh: 14 Aug 2026.
    """
    rows = [
        {
            "Rank": 1,
            "Sector / Theme": "Power Transmission, Grid & Energy Storage",
            "Key Govt Policy / Projects": "National Electricity Plan; ISTS/InSTS expansion; Green Energy Corridors; BESS & Pumped Storage deployment",
            "Project / Target Visibility": "500+ GW RE integration by 2030; transmission network planned to ~6.48 lakh ckm by 2032; ~47 GW BESS considered by 2031-32",
            "Likely Beneficiary Sub-sectors": "Transformers, switchgear, HVDC, cables/conductors, EPC, BESS, power electronics, pumped storage",
            "Policy Strength": 9.8,
            "10Y Opportunity Score": 9.7,
            "Horizon": "2026-2036",
            "Key Risk": "Execution delays, tender competition, valuation, commodity costs",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2243993"
        },
        {
            "Rank": 2,
            "Sector / Theme": "Renewable Energy – Solar, Wind & Hybrid",
            "Key Govt Policy / Projects": "500 GW non-fossil goal; Green Energy Corridors; hybrid/RTC projects; offshore wind; distributed solar",
            "Project / Target Visibility": "RE transmission planning beyond 500 GW by 2030 and 600 GW by 2032",
            "Likely Beneficiary Sub-sectors": "Solar modules, inverters, wind turbines, EPC, renewable developers, cables, structures",
            "Policy Strength": 9.6,
            "10Y Opportunity Score": 9.5,
            "Horizon": "2026-2036",
            "Key Risk": "Module price pressure, China competition, DISCOM/payment risk",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2236994"
        },
        {
            "Rank": 3,
            "Sector / Theme": "Defence, Aerospace, Drones & Electronics",
            "Key Govt Policy / Projects": "Aatmanirbhar defence; DAP 2026; indigenous-content push; defence corridors; AMCA/naval systems; iDEX",
            "Project / Target Visibility": "Defence production target ₹3 lakh crore and exports ₹50,000 crore by 2029; large corridor investments proposed",
            "Likely Beneficiary Sub-sectors": "Missiles, radars, electronics, aerospace structures, drones, ammunition, shipbuilding, defence components",
            "Policy Strength": 9.5,
            "10Y Opportunity Score": 9.4,
            "Horizon": "2026-2036",
            "Key Risk": "Order timing, government-customer concentration, execution, high valuation",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2286045"
        },
        {
            "Rank": 4,
            "Sector / Theme": "Electronics, Components & Semiconductors",
            "Key Govt Policy / Projects": "India Semiconductor Mission; electronics PLI; ECMS/components ecosystem; manufacturing clusters",
            "Project / Target Visibility": "Government push to deepen domestic electronics value-add and semiconductor/design ecosystem",
            "Likely Beneficiary Sub-sectors": "EMS, electronic components, PCB/ATMP/OSAT, semiconductor equipment/materials, industrial electronics",
            "Policy Strength": 9.4,
            "10Y Opportunity Score": 9.3,
            "Horizon": "2026-2036",
            "Key Risk": "Technology cycles, import dependence, execution, global capex cycle",
            "Official Source": "https://ecms.meity.gov.in/"
        },
        {
            "Rank": 5,
            "Sector / Theme": "Railways, High-Speed Rail & Freight",
            "Key Govt Policy / Projects": "Record railway capex; seven high-speed corridors; new Dedicated Freight Corridor; multitracking and safety upgrades",
            "Project / Target Visibility": "~4,000 km proposed high-speed corridors; 2,052 km new DFC; multi-year network/safety capex",
            "Likely Beneficiary Sub-sectors": "Rail EPC, signalling, rolling stock, propulsion, bearings, cables, stations, bridges, logistics",
            "Policy Strength": 9.4,
            "10Y Opportunity Score": 9.2,
            "Horizon": "2026-2036",
            "Key Risk": "Project delays, tender pricing, capex normalization",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2221838"
        },
        {
            "Rank": 6,
            "Sector / Theme": "Green Hydrogen & Green Ammonia",
            "Key Govt Policy / Projects": "National Green Hydrogen Mission; electrolyser manufacturing incentives; hydrogen production awards; green-ammonia demand aggregation",
            "Project / Target Visibility": "5 MMT annual green-hydrogen ambition by 2030; 3,000 MW/yr electrolyser awards; 8.62 lakh TPA hydrogen awards",
            "Likely Beneficiary Sub-sectors": "Electrolysers, renewable power, engineering, compressors, storage, ammonia, ports, industrial gases",
            "Policy Strength": 9.0,
            "10Y Opportunity Score": 9.0,
            "Horizon": "2026-2036",
            "Key Risk": "Economics/cost parity, offtake, technology, project finance",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2244663"
        },
        {
            "Rank": 7,
            "Sector / Theme": "Roads, Expressways & Multimodal Logistics",
            "Key Govt Policy / Projects": "National High-Speed Corridors; Greenfield Expressways; Bharatmala; PM GatiShakti multimodal planning",
            "Project / Target Visibility": "10,389 km high-speed corridors awarded; 5,580 km under implementation as of Jul-2026",
            "Likely Beneficiary Sub-sectors": "Road EPC, cement, steel, construction equipment, logistics, tolling, warehousing",
            "Policy Strength": 8.9,
            "10Y Opportunity Score": 8.8,
            "Horizon": "2026-2036",
            "Key Risk": "Land acquisition, leverage, project delays, competitive bidding",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2287761"
        },
        {
            "Rank": 8,
            "Sector / Theme": "EVs, Charging & Battery Ecosystem",
            "Key Govt Policy / Projects": "PM E-DRIVE; public EV charging infrastructure; e-buses/e-trucks; domestic EV ecosystem",
            "Project / Target Visibility": "₹10,900 crore PM E-DRIVE outlay; ₹2,000 crore for EV public charging infrastructure",
            "Likely Beneficiary Sub-sectors": "EV OEMs, chargers, power electronics, motors, battery packs, cells, bus makers, cables",
            "Policy Strength": 8.8,
            "10Y Opportunity Score": 8.7,
            "Horizon": "2026-2036",
            "Key Risk": "Battery technology shifts, subsidy changes, competition, raw-material dependence",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2290696"
        },
        {
            "Rank": 9,
            "Sector / Theme": "AI, Cloud, Data Centres & Digital Infrastructure",
            "Key Govt Policy / Projects": "IndiaAI Mission; shared AI compute; foundation models, AIKosh, labs and AI ecosystem",
            "Project / Target Visibility": "₹10,371+ crore IndiaAI Mission; 38,000+ GPUs onboarded for common compute by Mar-2026",
            "Likely Beneficiary Sub-sectors": "Data centres, cloud, power/cooling, GPUs/servers, networking, cybersecurity, AI software",
            "Policy Strength": 8.7,
            "10Y Opportunity Score": 8.7,
            "Horizon": "2026-2036",
            "Key Risk": "Very fast technology change, power availability, capex intensity, global competition",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2245069"
        },
        {
            "Rank": 10,
            "Sector / Theme": "Pharma APIs, Complex Drugs & Medical Devices",
            "Key Govt Policy / Projects": "Pharma PLI; Bulk Drugs PLI/Parks; Medical Device Parks; import-substitution drive",
            "Project / Target Visibility": "Bulk-drug parks in AP, Gujarat and HP; multiple greenfield API projects and high-value pharma manufacturing",
            "Likely Beneficiary Sub-sectors": "APIs/KSMs, CDMO, complex generics, medical devices, diagnostics, pharma equipment",
            "Policy Strength": 8.5,
            "10Y Opportunity Score": 8.4,
            "Horizon": "2026-2036",
            "Key Risk": "USFDA/regulatory risk, pricing pressure, R&D failure, global demand",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2239574"
        },
    ]
    return pd.DataFrame(rows)

def build_policy_pdf(policy_df):
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    story = [
        Paragraph("Government Policy + 10-Year Sector Opportunity Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(
            "India policy/project research snapshot — refreshed 14 Aug 2026. "
            "Opportunity scores are research heuristics based on policy visibility, capex duration, import substitution, "
            "demand runway and ecosystem depth; they are not return forecasts.",
            styles["BodyText"]
        ),
        Spacer(1, 12),
    ]
    data = [["Rank", "Sector / Theme", "Policy", "10Y", "Horizon"]]
    for _, r in policy_df.iterrows():
        data.append([
            str(int(r["Rank"])),
            str(r["Sector / Theme"])[:34],
            f'{r["Policy Strength"]:.1f}',
            f'{r["10Y Opportunity Score"]:.1f}',
            str(r["Horizon"]),
        ])
    t = Table(data, colWidths=[35, 260, 55, 55, 80])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    for _, r in policy_df.iterrows():
        story.append(Paragraph(f'<b>{int(r["Rank"])}. {r["Sector / Theme"]}</b>', styles["Heading3"]))
        story.append(Paragraph(f'<b>Policy/Projects:</b> {r["Key Govt Policy / Projects"]}', styles["BodyText"]))
        story.append(Paragraph(f'<b>Visibility:</b> {r["Project / Target Visibility"]}', styles["BodyText"]))
        story.append(Paragraph(f'<b>Beneficiaries:</b> {r["Likely Beneficiary Sub-sectors"]}', styles["BodyText"]))
        story.append(Paragraph(f'<b>Risk:</b> {r["Key Risk"]}', styles["BodyText"]))
        story.append(Spacer(1, 6))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def classify_macro_regime(mscore, mdf):
    """Heuristic research regime from market/macro trends; not an official economic-cycle call."""
    def v(name, col="3M %"):
        r = mdf.loc[mdf["Indicator"]==name, col] if not mdf.empty else pd.Series(dtype=float)
        return float(r.iloc[0]) if len(r) else np.nan
    nifty, vix, crude, y10 = v("NIFTY 50"), v("India VIX"), v("Brent Crude"), v("US 10Y Yield")
    if mscore >= 7.2 and (pd.isna(nifty) or nifty > 0): return "EARLY CYCLE", "Risk-on / recovery"
    if mscore >= 5.5 and (pd.isna(nifty) or nifty >= -2): return "MID CYCLE", "Broad expansion"
    if mscore >= 4.0 or ((not pd.isna(crude) and crude > 8) and (not pd.isna(y10) and y10 > 5)):
        return "LATE CYCLE", "Maturing / inflation-sensitive"
    return "RECESSION / RISK-OFF", "Defensive regime"

@st.cache_data(ttl=3600, show_spinner=False)
def benchmark_monthly():
    """NIFTY 50 is used as a liquid market benchmark for relative-strength context."""
    try:
        d = yf.download("^NSEI", period="2y", interval="1d", auto_adjust=False, progress=False)
        if d.empty:
            return pd.DataFrame()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return to_monthly(d)
    except Exception:
        return pd.DataFrame()

def _period_return(close, months):
    c = close.dropna()
    if len(c) < months + 1:
        return np.nan
    base = float(c.iloc[-(months+1)])
    last = float(c.iloc[-1])
    return (last/base - 1)*100 if base else np.nan

def sector_rotation_table(res, monthlies):
    """
    Sector strength combines:
    - median 3M / 6M / 12M stock momentum,
    - 6M relative strength versus NIFTY 50,
    - breakout breadth / near-ATH breadth.
    This is a research heuristic, not an official sector index calculation.
    """
    rows=[]
    if res.empty:
        return pd.DataFrame()

    bm = benchmark_monthly()
    bm6 = _period_return(bm["Close"], 6) if not bm.empty and "Close" in bm.columns else np.nan

    for ind, g in res.groupby("Industry"):
        r3s, r6s, r12s = [], [], []
        for sym in g["Symbol"].head(30):
            m = monthlies.get(sym+".NS", pd.DataFrame())
            if m.empty or "Close" not in m.columns:
                continue
            c = m["Close"]
            r3 = _period_return(c, 3)
            r6 = _period_return(c, 6)
            r12 = _period_return(c, 12)
            if pd.notna(r3): r3s.append(r3)
            if pd.notna(r6): r6s.append(r6)
            if pd.notna(r12): r12s.append(r12)

        mom3 = float(np.median(r3s)) if r3s else np.nan
        mom6 = float(np.median(r6s)) if r6s else np.nan
        mom12 = float(np.median(r12s)) if r12s else np.nan
        rs6 = mom6 - bm6 if pd.notna(mom6) and pd.notna(bm6) else np.nan

        br = int((g["Months Since Signal"].fillna(999)<=3).sum())
        near = int(((g["Status"]=="Near/No Breakout") & (g["Breakout %"]>=-5)).sum())
        breadth = 100*(br + 0.5*near)/max(1, len(g))

        momentum_component = 0
        weight = 0
        for val, denom, wt in [(mom3, 8, 0.25), (mom6, 12, 0.35), (mom12, 20, 0.20), (rs6, 10, 0.20)]:
            if pd.notna(val):
                momentum_component += np.tanh(val/denom) * wt
                weight += wt
        momentum_component = momentum_component/weight if weight else 0

        score = 5 + 3.2*momentum_component + min(2.0, breadth/25)
        score = float(np.clip(score, 0, 10))
        phase = "LEADING" if score>=7.2 and (pd.isna(rs6) or rs6>=0) else \
                "IMPROVING" if score>=5.5 else \
                "WEAKENING" if score>=4.0 else "LAGGING"

        rows.append({
            "Industry": ind,
            "3M Momentum %": mom3,
            "6M Momentum %": mom6,
            "12M Momentum %": mom12,
            "6M RS vs NIFTY %": rs6,
            "Fresh/≤3M Breakouts": br,
            "Near ATH": near,
            "Breadth %": breadth,
            "Sector Score": score,
            "Rotation Phase": phase
        })
    return pd.DataFrame(rows).sort_values("Sector Score", ascending=False)

def make_sector_rotation_map(sector_df):
    """
    4-quadrant sector-rotation map.
    X = 6M relative strength vs NIFTY.
    Y = 3M momentum (recent acceleration/deceleration).
    Bubble size = Sector Score.
    """
    if sector_df is None or sector_df.empty:
        return go.Figure()

    d = sector_df.copy()
    d = d.dropna(subset=["6M RS vs NIFTY %", "3M Momentum %"])
    if d.empty:
        return go.Figure()

    def quadrant(row):
        x = float(row["6M RS vs NIFTY %"])
        y = float(row["3M Momentum %"])
        if x >= 0 and y >= 0:
            return "LEADING"
        if x < 0 and y >= 0:
            return "IMPROVING"
        if x >= 0 and y < 0:
            return "WEAKENING"
        return "LAGGING"

    d["Rotation Quadrant"] = d.apply(quadrant, axis=1)
    sizes = 16 + 5 * d["Sector Score"].fillna(5).clip(0, 10)

    fig = go.Figure()
    for q in ["LEADING", "IMPROVING", "WEAKENING", "LAGGING"]:
        qd = d[d["Rotation Quadrant"] == q]
        if qd.empty:
            continue
        fig.add_trace(go.Scatter(
            x=qd["6M RS vs NIFTY %"],
            y=qd["3M Momentum %"],
            mode="markers+text",
            text=qd["Industry"],
            textposition="top center",
            name=q,
            marker=dict(
                size=sizes.loc[qd.index],
                opacity=0.72,
                line=dict(width=1)
            ),
            customdata=np.stack([
                qd["Sector Score"].round(2),
                qd["6M Momentum %"].round(2),
                qd["12M Momentum %"].round(2),
                qd["Breadth %"].round(2),
            ], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Quadrant: " + q + "<br>"
                "6M RS vs NIFTY: %{x:.2f}%<br>"
                "3M Momentum: %{y:.2f}%<br>"
                "Sector Score: %{customdata[0]:.1f}/10<br>"
                "6M Momentum: %{customdata[1]:.2f}%<br>"
                "12M Momentum: %{customdata[2]:.2f}%<br>"
                "Breadth: %{customdata[3]:.1f}%<extra></extra>"
            )
        ))

    xabs = max(10.0, float(np.nanmax(np.abs(d["6M RS vs NIFTY %"]))) * 1.20)
    yabs = max(10.0, float(np.nanmax(np.abs(d["3M Momentum %"]))) * 1.25)

    fig.add_vline(x=0, line_dash="dash")
    fig.add_hline(y=0, line_dash="dash")
    fig.add_annotation(x=xabs*0.72, y=yabs*0.90, text="LEADING<br>Strong + Accelerating", showarrow=False)
    fig.add_annotation(x=-xabs*0.72, y=yabs*0.90, text="IMPROVING<br>Weak RS + Accelerating", showarrow=False)
    fig.add_annotation(x=xabs*0.72, y=-yabs*0.90, text="WEAKENING<br>Strong RS + Slowing", showarrow=False)
    fig.add_annotation(x=-xabs*0.72, y=-yabs*0.90, text="LAGGING<br>Weak + Slowing", showarrow=False)

    fig.update_layout(
        title="4-Quadrant Sector Rotation Map",
        xaxis_title="6M Relative Strength vs NIFTY (%) →",
        yaxis_title="3M Momentum / Acceleration (%) →",
        xaxis=dict(range=[-xabs, xabs], zeroline=False),
        yaxis=dict(range=[-yabs, yabs], zeroline=False),
        height=680,
        legend_title="Rotation Quadrant",
        margin=dict(l=20, r=20, t=70, b=30),
    )
    return fig

def add_rotation_quadrant(sector_df):
    if sector_df is None or sector_df.empty:
        return sector_df
    d = sector_df.copy()
    def quadrant(row):
        x = row.get("6M RS vs NIFTY %", np.nan)
        y = row.get("3M Momentum %", np.nan)
        if pd.isna(x) or pd.isna(y):
            return "N/A"
        if x >= 0 and y >= 0:
            return "LEADING"
        if x < 0 and y >= 0:
            return "IMPROVING"
        if x >= 0 and y < 0:
            return "WEAKENING"
        return "LAGGING"
    d["Rotation Quadrant"] = d.apply(quadrant, axis=1)
    order = {"LEADING": 0, "IMPROVING": 1, "WEAKENING": 2, "LAGGING": 3, "N/A": 4}
    d["_q"] = d["Rotation Quadrant"].map(order).fillna(4)
    d = d.sort_values(["_q", "Sector Score"], ascending=[True, False]).drop(columns="_q")
    return d

def policy_support_for_industry(industry_name):
    """Map NIFTY industry labels to broad government-policy themes."""
    s = str(industry_name).lower()
    mapping = [
        (["power", "electrical", "transformer", "cable", "utilities", "energy"], 9.4, "Grid / Energy"),
        (["renewable", "solar", "wind"], 9.5, "Renewable Energy"),
        (["defence", "aerospace", "shipbuilding"], 9.4, "Defence / Aerospace"),
        (["electronic", "semiconductor", "telecom", "computer hardware"], 9.2, "Electronics / Semiconductor"),
        (["rail", "transport infrastructure"], 9.1, "Railways / Infrastructure"),
        (["auto", "automobile", "battery", "ev"], 8.7, "EV / Battery"),
        (["construction", "cement", "logistics", "road", "port"], 8.7, "Infrastructure / Logistics"),
        (["pharma", "health", "hospital", "medical"], 8.4, "Pharma / Healthcare"),
        (["software", "it ", "information technology", "cyber"], 8.5, "AI / Digital"),
        (["chemical", "industrial gas", "fertilizer"], 8.2, "Green Hydrogen / Chemicals"),
    ]
    for keys, score, theme in mapping:
        if any(k in s for k in keys):
            return score, theme
    return 5.0, "Broad / Neutral"

def auto_fundamental_lite(ticker):
    """
    Fast fundamental-quality score used for batch ranking.
    It intentionally excludes peer PE and promoter trend to keep batch scanning practical.
    """
    f = fetch_fundamentals(ticker)
    checks = []
    vals = [
        (f.get("Debt/Equity"), lambda x: x < 0.5),
        (f.get("ROCE %"), lambda x: x > 15),
        (f.get("ROC %"), lambda x: x > 15),
        (f.get("Net Profit Increasing"), lambda x: x is True),
        (f.get("Sales CAGR %") if f.get("Sales CAGR Years",0) >= 5 else np.nan, lambda x: x > 20),
        (f.get("Profit CAGR %") if f.get("Profit CAGR Years",0) >= 5 else np.nan, lambda x: x > 15),
    ]
    for v, rule in vals:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        try:
            checks.append(1 if rule(v) else 0)
        except Exception:
            pass
    score = 100*sum(checks)/len(checks) if checks else np.nan
    return score, len(checks), f

def build_pro_opportunity_flags(row):
    return {
        "Macro": "✅" if row.get("Macro Score", 0) >= 6 else "⚠️" if row.get("Macro Score",0)>=4 else "❌",
        "Policy": "✅" if row.get("Policy Score", 0) >= 8 else "⚠️" if row.get("Policy Score",0)>=6 else "➖",
        "Sector": "✅" if row.get("Sector Score", 0) >= 7 else "⚠️" if row.get("Sector Score",0)>=5.5 else "❌",
        "26M": "✅" if str(row.get("Status","")).startswith("Fresh") or str(row.get("Status","")).startswith("Breakout") else "👀",
        "Fundamental": "✅" if row.get("Fundamental Score %", np.nan) >= 70 else "⚠️" if row.get("Fundamental Score %", np.nan) >= 50 else "❌" if pd.notna(row.get("Fundamental Score %",np.nan)) else "N/A",
    }


def build_pdf_report(regime, mscore, sector_df, ranked_df):
    buf=io.BytesIO(); styles=getSampleStyleSheet(); doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=28,leftMargin=28,topMargin=28,bottomMargin=28)
    story=[Paragraph("India Macro Cycle + Sector Rotation + 26M ATH Report",styles["Title"]),Spacer(1,10),Paragraph(f"Macro regime: <b>{regime}</b> | Macro score: <b>{mscore:.1f}/10</b>",styles["BodyText"]),Spacer(1,12)]
    if not sector_df.empty:
        story += [Paragraph("Top Sectors",styles["Heading2"])]
        data=[["Industry","Score","Phase"]]+[[str(r.Industry)[:28],f"{r['Sector Score']:.1f}",r['Rotation Phase']] for _,r in sector_df.head(10).iterrows()]
        t=Table(data,colWidths=[260,70,100]); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('FONTSIZE',(0,0),(-1,-1),8)])); story += [t,Spacer(1,12)]
    if not ranked_df.empty:
        story += [Paragraph("Top Ranked Stocks",styles["Heading2"])]
        data=[["Symbol","Industry","Final","Status"]]+[[str(r.Symbol),str(r.Industry)[:22],f"{r['Final Stock Score']:.1f}",str(r.Status)] for _,r in ranked_df.head(20).iterrows()]
        t=Table(data,colWidths=[65,190,55,100]); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('FONTSIZE',(0,0),(-1,-1),7)])); story += [t]
    story += [Spacer(1,12),Paragraph("Research tool only. Verify exchange/broker data and company filings before acting.",styles["BodyText"])]
    doc.build(story); return buf.getvalue()



# ---------------- VALUE MIGRATION MODULE (V6) ----------------

VALUE_MIGRATION_REFRESH = "15 Aug 2026"

# Representative NSE baskets are used only as live market-confirmation proxies.
# They do not imply that every company is a pure-play beneficiary.
VALUE_MIGRATION_BASKETS = {
    "Power Grid, Transformers & Transmission": [
        "POWERGRID.NS","ABB.NS","SIEMENS.NS","CGPOWER.NS","APARINDS.NS",
        "HITACHIENER.NS","GEVERNOVA.NS","KEC.NS","KPIL.NS","POLYCAB.NS"
    ],
    "AI Data Centre Infrastructure": [
        "NETWEB.NS","ANANTRAJ.NS","ABB.NS","SIEMENS.NS","CUMMINSIND.NS",
        "BLUESTARCO.NS","VOLTAS.NS","POLYCAB.NS","KEI.NS","TECHM.NS"
    ],
    "Battery Energy Storage (BESS) & Power Electronics": [
        "TATAPOWER.NS","JSWENERGY.NS","EXIDEIND.NS","AMARAJABAT.NS",
        "WAAREEENER.NS","ABB.NS","SIEMENS.NS","CGPOWER.NS"
    ],
    "Electronics Components / EMS / Semiconductor Ecosystem": [
        "DIXON.NS","KAYNES.NS","SYRMA.NS","AMBER.NS","PGEL.NS",
        "BEL.NS","NETWEB.NS","MOSCHIP.NS"
    ],
    "Defence Indigenisation & Component Suppliers": [
        "HAL.NS","BEL.NS","BDL.NS","MAZDOCK.NS","COCHINSHIP.NS",
        "GRSE.NS","DATAPATTNS.NS","PARAS.NS"
    ],
    "Grain / Flexible-feed Ethanol": [
        "BALRAMCHIN.NS","TRIVENI.NS","GLOBUSSPR.NS","RENUKA.NS",
        "EIDPARRY.NS","BAJAJHIND.NS"
    ],
}

def _vm_policy_rows():
    """Transparent policy evidence. Scores are anchored to quantified official targets/capex."""
    return [
        {
            "Theme": "Power Grid, Transformers & Transmission",
            "Old Value Pool": "Generation-led power capex",
            "New Value Pool": "Grid expansion, HV equipment, transmission, automation",
            "Primary Catalysts": "RE integration + EV load + data centres + manufacturing electrification",
            "Bottleneck / Picks & Shovels": "Transformers, switchgear, HVDC, conductors, cables, substations, grid automation",
            "Policy Evidence": "National Electricity Plan: transmission network ~4.98 lakh ckm (Nov-2025) → ~6.48 lakh ckm by 2032; transformation capacity 1,398 → 2,345 GVA; plan cost ~₹9.16 lakh Cr.",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2215187&lang=1&reg=1",
            "Policy / Capex": 9.9, "5-6Y Runway": 9.7, "Bottleneck Intensity": 9.8,
            "Early-stage Score": 8.3, "Stage": "Early-to-Mid",
            "Key Risk": "Rich valuations, execution delays, commodity costs, tender competition",
        },
        {
            "Theme": "AI Data Centre Infrastructure",
            "Old Value Pool": "Traditional enterprise IT / servers",
            "New Value Pool": "AI compute + data centres + electrical/cooling infrastructure",
            "Primary Catalysts": "AI adoption, cloud growth, sovereign data, hyperscaler capex",
            "Bottleneck / Picks & Shovels": "Power distribution, transformers, UPS, cooling, cables, backup power, racks",
            "Policy Evidence": "India data-centre buildout is primarily private-capex led; dashboard therefore gives this theme a lower policy weight and relies more on live market confirmation.",
            "Official Source": "https://www.meity.gov.in/",
            "Policy / Capex": 7.8, "5-6Y Runway": 9.8, "Bottleneck Intensity": 9.7,
            "Early-stage Score": 9.0, "Stage": "Early",
            "Key Risk": "Capex concentration, power/water constraints, technology changes, rich valuations",
        },
        {
            "Theme": "Battery Energy Storage (BESS) & Power Electronics",
            "Old Value Pool": "Renewable generation without storage",
            "New Value Pool": "Renewables + storage + dispatchable clean power",
            "Primary Catalysts": "Variable solar/wind, peak demand, grid balancing, storage tenders",
            "Bottleneck / Picks & Shovels": "Cells/packs, BMS, PCS/inverters, EMS, grid integration, storage EPC",
            "Policy Evidence": "NEP projects BESS requirement of 47.24 GW / 236 GWh by 2031-32 with estimated investment ~₹3.49 lakh Cr; VGF schemes target large storage additions.",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2290015&lang=1&reg=3",
            "Policy / Capex": 9.8, "5-6Y Runway": 9.8, "Bottleneck Intensity": 9.5,
            "Early-stage Score": 9.3, "Stage": "Early",
            "Key Risk": "Battery price compression, imports, chemistry shifts, tender economics",
        },
        {
            "Theme": "Electronics Components / EMS / Semiconductor Ecosystem",
            "Old Value Pool": "Imported electronics and low-value assembly",
            "New Value Pool": "Indian components, high-value EMS, packaging and power electronics",
            "Primary Catalysts": "China+1, localisation, ECMS, export manufacturing",
            "Bottleneck / Picks & Shovels": "PCB/PCBA, components, enclosures, connectors, OSAT/ATMP, capital equipment",
            "Policy Evidence": "ECMS approvals: 46 proposals had ~₹54,567 Cr projected investment; a further 29 proposals added ~₹7,104 Cr in Mar-2026.",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2247040&lang=1&reg=3",
            "Policy / Capex": 9.7, "5-6Y Runway": 9.5, "Bottleneck Intensity": 8.9,
            "Early-stage Score": 8.7, "Stage": "Early-to-Mid",
            "Key Risk": "Customer concentration, fast tech cycles, valuation, import dependence",
        },
        {
            "Theme": "Defence Indigenisation & Component Suppliers",
            "Old Value Pool": "Imported defence platforms/components",
            "New Value Pool": "Domestic platforms, electronics, precision components and exports",
            "Primary Catalysts": "Indigenisation lists, domestic procurement, export push",
            "Bottleneck / Picks & Shovels": "Radar/electronics, propulsion parts, precision engineering, drones, defence materials",
            "Policy Evidence": "Structural policy support remains strong, but many listed defence names have already re-rated; live market breadth is used to detect whether leadership is still broadening.",
            "Official Source": "https://www.mod.gov.in/",
            "Policy / Capex": 9.5, "5-6Y Runway": 9.0, "Bottleneck Intensity": 8.7,
            "Early-stage Score": 6.8, "Stage": "Mid",
            "Key Risk": "Valuation, order timing, customer concentration, execution",
        },
        {
            "Theme": "Grain / Flexible-feed Ethanol",
            "Old Value Pool": "Pure petrol + sugar-heavy ethanol feedstock",
            "New Value Pool": "E20 + grain/flexible-feed ethanol supply",
            "Primary Catalysts": "E20 demand, flexible feedstock, OMC procurement",
            "Bottleneck / Picks & Shovels": "Flexible-feed distilleries, grain logistics, enzymes, DDGS integration",
            "Policy Evidence": "E20 is already a mature policy milestone; upside now depends more on feedstock economics, utilisation and company-level margins than on a new blending step-up.",
            "Official Source": "https://mopng.gov.in/",
            "Policy / Capex": 7.8, "5-6Y Runway": 6.5, "Bottleneck Intensity": 6.6,
            "Early-stage Score": 5.2, "Stage": "Mid-to-Late",
            "Key Risk": "E20 ceiling, oversupply, feedstock prices, policy allocation",
        },
    ]

@st.cache_data(ttl=1800, show_spinner=False)
def live_value_migration_market_signals():
    """Live market confirmation from representative NSE baskets (Yahoo Finance)."""
    rows = []
    for theme, tickers in VALUE_MIGRATION_BASKETS.items():
        rets = {"1M": [], "3M": [], "6M": [], "12M": []}
        available = 0
        positive6 = 0
        try:
            raw = yf.download(
                tickers=tickers, period="15mo", interval="1d",
                group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
        except Exception:
            raw = pd.DataFrame()

        for ticker in tickers:
            try:
                if raw.empty:
                    continue
                if len(tickers) == 1:
                    px = raw["Close"]
                else:
                    px = raw[ticker]["Close"]
                px = pd.to_numeric(px, errors="coerce").dropna()
                if len(px) < 40:
                    continue
                available += 1
                last = float(px.iloc[-1])
                def ret(days):
                    if len(px) <= days:
                        return np.nan
                    return (last / float(px.iloc[-days-1]) - 1) * 100
                r1, r3, r6, r12 = ret(21), ret(63), ret(126), ret(252)
                for k, v in [("1M",r1),("3M",r3),("6M",r6),("12M",r12)]:
                    if pd.notna(v):
                        rets[k].append(v)
                if pd.notna(r6) and r6 > 0:
                    positive6 += 1
            except Exception:
                continue

        med = {k: (float(np.nanmedian(v)) if v else np.nan) for k,v in rets.items()}
        breadth = 100 * positive6 / available if available else np.nan

        # Live score rewards medium-term trend and breadth, while capping extreme moves.
        parts = []
        if pd.notna(med["3M"]): parts.append(5 + 5*np.tanh(med["3M"]/20))
        if pd.notna(med["6M"]): parts.append(5 + 5*np.tanh(med["6M"]/35))
        if pd.notna(med["12M"]): parts.append(5 + 5*np.tanh(med["12M"]/60))
        if pd.notna(breadth): parts.append(np.clip(breadth/10,0,10))
        live_score = float(np.mean(parts)) if parts else np.nan

        rows.append({
            "Theme": theme, "Basket Stocks": available,
            "Median 1M %": med["1M"], "Median 3M %": med["3M"],
            "Median 6M %": med["6M"], "Median 12M %": med["12M"],
            "6M Positive Breadth %": breadth, "Live Market Score": live_score,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=False)
def value_migration_themes():
    """Policy + structural runway + LIVE market confirmation."""
    df = pd.DataFrame(_vm_policy_rows())
    live = live_value_migration_market_signals()
    df = df.merge(live, on="Theme", how="left")

    # Live score carries 30% of the final result; if unavailable, use neutral 5/10.
    live_component = df["Live Market Score"].fillna(5.0)
    df["Value Migration Score"] = (
        0.25*df["Policy / Capex"] +
        0.20*df["5-6Y Runway"] +
        0.15*df["Bottleneck Intensity"] +
        0.10*df["Early-stage Score"] +
        0.30*live_component
    ) * 10

    df["Rank"] = df["Value Migration Score"].rank(method="first", ascending=False).astype(int)
    return df.sort_values("Value Migration Score", ascending=False).reset_index(drop=True)

def _theme_industry_match(theme, industry):
    s = str(industry).lower()
    mapping = {
        "Power Grid, Transformers & Transmission": ["power", "electric", "electrical", "cable", "transform", "transmission", "capital goods", "engineering"],
        "AI Data Centre Infrastructure": ["software", "it ", "telecom", "electrical", "power", "cooling", "air condition", "capital goods", "engineering"],
        "Battery Energy Storage (BESS) & Power Electronics": ["battery", "storage", "power", "electrical", "renewable", "energy", "electronics"],
        "Electronics Components / EMS / Semiconductor Ecosystem": ["electronics", "electronic", "semiconductor", "telecom", "consumer durables", "capital goods"],
        "Defence Indigenisation & Component Suppliers": ["defence", "aerospace", "engineering", "electronics", "ship", "capital goods"],
        "Grain / Flexible-feed Ethanol": ["sugar", "distiller", "alcohol", "beverage", "agri", "food products"],
    }
    return any(k in s for k in mapping.get(theme, []))

def value_migration_candidate_score(theme_score, live_score, breadth, stock_score=np.nan,
                                    fundamental_score=np.nan):
    """Availability-aware candidate heuristic; inputs are normalized to 0–100."""
    components = [(theme_score, 0.30), (live_score * 10, 0.15), (breadth, 0.10)]
    if pd.notna(stock_score):
        components.append((stock_score * 10, 0.25))
    if pd.notna(fundamental_score):
        components.append((fundamental_score, 0.20))
    available = [(float(v), w) for v, w in components if pd.notna(v)]
    if not available:
        return np.nan, 0
    weight = sum(w for _, w in available)
    return float(np.clip(sum(v * w for v, w in available) / weight, 0, 100)), len(available)

def render_value_migration_page():
    st.title("🚀 Value Migration → Real-Data Multibagger Hunting Engine")
    st.caption(
        "Final theme score now combines quantified policy/capex evidence with LIVE NSE basket momentum and breadth. "
        "A high score identifies a strong migration setup; it does not predict or guarantee 10x/40x returns."
    )

    if st.button("🔄 Refresh Live Theme Data", use_container_width=True):
        live_value_migration_market_signals.clear()
        value_migration_themes.clear()
        st.rerun()

    vm = value_migration_themes()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Top Theme", vm.iloc[0]["Theme"])
    c2.metric("Top VM Score", f"{vm.iloc[0]['Value Migration Score']:.0f}/100")
    c3.metric("Themes Tracked", len(vm))
    c4.metric("Live Data TTL", "30 min")

    st.markdown("### 1️⃣ Real-Data Value Migration Ranking")
    display = vm[[
        "Rank","Theme","Stage","Policy / Capex","5-6Y Runway",
        "Live Market Score","Median 6M %","6M Positive Breadth %",
        "Value Migration Score","Key Risk"
    ]].copy()
    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Policy / Capex": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            "5-6Y Runway": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            "Live Market Score": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            "6M Positive Breadth %": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f%%"),
            "Value Migration Score": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
        }
    )

    fig = go.Figure(go.Bar(
        x=vm["Value Migration Score"], y=vm["Theme"], orientation="h",
        text=vm["Value Migration Score"].round(0), textposition="auto"
    ))
    fig.update_layout(
        title="Policy + Runway + Live Market Confirmation",
        xaxis_title="Dynamic Score / 100", yaxis_title="", height=430,
        yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=50,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2️⃣ Migration Chain + Live Confirmation")
    selected_theme = st.selectbox("Choose Value Migration theme", vm["Theme"].tolist())
    r = vm[vm["Theme"]==selected_theme].iloc[0]

    a,b,c,d = st.columns(4)
    a.metric("Policy / Capex", f"{r['Policy / Capex']:.1f}/10")
    b.metric("5–6Y Runway", f"{r['5-6Y Runway']:.1f}/10")
    c.metric("Live Market", "N/A" if pd.isna(r["Live Market Score"]) else f"{r['Live Market Score']:.1f}/10")
    d.metric("6M Breadth", "N/A" if pd.isna(r["6M Positive Breadth %"]) else f"{r['6M Positive Breadth %']:.0f}%")

    st.info(
        f"**OLD VALUE POOL** → {r['Old Value Pool']}\n\n"
        f"**CATALYST** → {r['Primary Catalysts']}\n\n"
        f"**NEW VALUE POOL** → {r['New Value Pool']}\n\n"
        f"**BOTTLENECK / PICKS & SHOVELS** → {r['Bottleneck / Picks & Shovels']}"
    )
    st.markdown(f"**Policy evidence:** {r['Policy Evidence']}")
    st.link_button("🏛️ Open Official Source", r["Official Source"], use_container_width=True)
    st.warning(f"Key risk: {r['Key Risk']}")

    st.markdown("### 3️⃣ Live Theme Market Diagnostics")
    diag_cols = ["Theme","Basket Stocks","Median 1M %","Median 3M %","Median 6M %","Median 12M %","6M Positive Breadth %","Live Market Score"]
    st.dataframe(vm[diag_cols], use_container_width=True, hide_index=True)

    st.markdown("### 4️⃣ What Can Create a 10x/40x Candidate?")
    st.markdown("""
**Sector migration alone is not enough.** Prefer companies where several conditions overlap:
- Addressable opportunity is large relative to the company's current revenue/market-cap base.
- Capacity/order book can multiply revenue over several years.
- Earnings growth is sustained and ROCE stays healthy/improves.
- Debt and cash-flow remain manageable.
- The company supplies a genuine bottleneck / picks-and-shovels product.
- Valuation still leaves room for earnings-led compounding.
- The theme has broad market confirmation, not only one speculative stock.
- Price confirmation: **monthly close above an ≥26-month-old ATH**.
""")

    st.markdown("### 5️⃣ Candidate Discovery — NIFTY 500")
    st.caption("Candidate classification is then linked to your latest 26M breakout/fundamental scan.")
    try:
        uni = load_nifty500()
        cand = uni[uni["Industry"].apply(lambda x: _theme_industry_match(selected_theme, x))].copy()
        cand = cand.rename(columns={"Company Name":"Company"})
        st.metric("Theme-linked NIFTY 500 universe", len(cand))

        latest_ranked = st.session_state.get("latest_ranked", pd.DataFrame())
        if not latest_ranked.empty:
            cols = [c for c in ["Symbol","Status","Sector Score","Policy Score","Fundamental Score %","Pro Final Score"] if c in latest_ranked.columns]
            joined = cand.merge(latest_ranked[cols], on="Symbol", how="left")
            joined["Migration Theme Score"] = float(r["Value Migration Score"])
            scored = joined.apply(
                lambda x: value_migration_candidate_score(
                    r["Value Migration Score"], r["Live Market Score"],
                    r["6M Positive Breadth %"], x.get("Pro Final Score", np.nan),
                    x.get("Fundamental Score %", np.nan)), axis=1
            )
            joined["Candidate Heuristic %"] = scored.apply(lambda x: x[0])
            joined["Inputs Available"] = scored.apply(lambda x: f"{x[1]}/5")
            joined = joined.sort_values(["Candidate Heuristic %","Pro Final Score"], ascending=False, na_position="last")
            st.success("Latest main-dashboard scan is linked to this Value Migration page.")
            st.dataframe(joined, use_container_width=True, hide_index=True)
        else:
            st.dataframe(cand[["Symbol","Company","Industry"]], use_container_width=True, hide_index=True)
            st.info("Run the Main Dashboard scan once; then return here to merge 26M breakout + fundamentals into the migration ranking.")
    except Exception as e:
        st.warning(f"Candidate universe unavailable right now: {e}")

    st.markdown("### 6️⃣ Dynamic Scoring Formula")
    st.code(
        "Value Migration Score = 25% Policy/Capex + 20% 5–6Y Runway + "
        "15% Bottleneck + 10% Early-stage + 30% LIVE Market Confirmation"
    )
    st.caption(
        "Candidate Heuristic = 30% theme + 15% live basket confirmation + 10% breadth + "
        "25% stock confirmation + 20% fundamentals. Missing stock/fundamental inputs are excluded "
        "and the remaining weights are normalized; Inputs Available makes that coverage explicit."
    )
    st.success(
        "MACRO CHANGE → POLICY/CAPEX → VALUE MIGRATION → BOTTLENECK → LIVE MARKET BREADTH → "
        "SMALL/MID-CAP BENEFICIARY → FUNDAMENTALS → VALUATION → 26M ATH BREAKOUT → WATCHLIST"
    )
    st.caption("Research tool only. Live market data comes from Yahoo Finance and can be temporarily unavailable. Official policy links are included for verification.")

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



@st.cache_data(ttl=900, show_spinner=False)
def graham_value_data(ticker):
    # A short cache keeps CMP reasonably fresh while avoiding repeated calls to
    # the quote and historical fundamentals endpoints on each Streamlit rerun.
    return fetch_graham_data(ticker)

def render_graham_page():
    st.title("🧮 Graham Value Formula Scorecard")
    st.caption("Ten Graham tests calculated from Yahoo Finance live quotes, reported annual statements, fundamentals history and dividend events. Missing observations remain N/A.")
    sym = st.text_input("NSE Symbol", value="BSE", help="Examples: BSE, BALRAMCHIN, LT, BEL").strip().upper()
    if not st.button("🔎 Calculate Graham Score", type="primary", use_container_width=True):
        st.info("Enter an NSE symbol and press **Calculate Graham Score**.")
        return
    try:
        with st.spinner("Fetching live quote and reported financial history…"):
            d = graham_value_data(sym)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Financial data could not be fetched: {error}")
        return

    tests = [
        ("1", "Adequate Size", "Sales > ₹500 Cr", d["sales"] / 1e7 if pd.notna(d["sales"]) else np.nan,
         None if pd.isna(d["sales"]) else d["sales"] > 500e7, "₹ Cr"),
        ("2", "Current Ratio", "Current Assets / Current Liabilities > 2", d["current_ratio"],
         None if pd.isna(d["current_ratio"]) else d["current_ratio"] > 2, "x"),
        ("3", "Debt Check", "Long-term Debt < Net Working Capital", d["long_debt"] / 1e7 if pd.notna(d["long_debt"]) else np.nan,
         None if pd.isna(d["long_debt"]) or pd.isna(d["nwc"]) else d["long_debt"] < d["nwc"], "₹ Cr LT debt"),
        ("4", "Earnings Stability", "Positive EPS in each of last 10 annual reports", np.nan, d["eps_positive_10y"], ""),
        ("5", "Dividend Record", "Dividend paid in each of 20 completed years", np.nan, d["dividend20"], ""),
        ("6", "Earnings Growth", "Latest 3-year avg EPS ≥ 133% of first 3-year avg", np.nan, d["eps_growth_10y"], ""),
        ("7", "3-year Average P/E", "CMP / 3-year average EPS < 15", d["pe3"],
         None if pd.isna(d["pe3"]) else d["pe3"] < 15, "x"),
        ("8", "P/B Ratio", "CMP / book value per share < 1.5", d["pb"],
         None if pd.isna(d["pb"]) else d["pb"] < 1.5, "x"),
        ("9", "Combined Test", "3-year P/E × P/B < 22.5", d["combined"],
         None if pd.isna(d["combined"]) else d["combined"] < 22.5, "x"),
        ("10", "Graham Number", "√(22.5 × 3-year avg EPS × BVPS) > CMP", d["graham_no"],
         None if pd.isna(d["graham_no"]) or pd.isna(d["price"]) else d["graham_no"] > d["price"], "₹"),
    ]
    rows, passed, assessed = [], 0, 0
    for no, name, formula, value, result, unit in tests:
        if result is None:
            status = "⚪ N/A"
        else:
            assessed += 1
            passed += int(bool(result))
            status = "🟢 PASS" if result else "🔴 FAIL"
        if pd.isna(value):
            display = "N/A"
        elif unit == "₹ Cr":
            display = f"₹{value:,.0f} Cr"
        elif unit == "₹ Cr LT debt":
            nwc = f"; NWC ₹{d['nwc'] / 1e7:,.0f} Cr" if pd.notna(d["nwc"]) else "; NWC N/A"
            display = f"₹{value:,.0f} Cr{nwc}"
        elif unit == "₹":
            display = f"₹{value:,.2f}"
        else:
            display = f"{value:.2f}x"
        rows.append({"#": no, "Criterion": name, "Formula / threshold": formula, "Reported value": display, "Result": status})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Company", d["company"])
    m2.metric("CMP", "N/A" if pd.isna(d["price"]) else f"₹{d['price']:,.2f}")
    m3.metric("Graham Score", f"{passed}/{assessed} evaluable")
    m4.metric("Unavailable", f"{10 - assessed}/10")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Ticker: {d['symbol']} · Retrieved: {d['as_of']} · Source: {d['sources']}. Annual EPS periods found: {d['eps_years']}. Score percentage is {100 * passed / assessed:.0f}% of evaluable criteria." if assessed else f"Ticker: {d['symbol']} · Retrieved: {d['as_of']} · No criteria were evaluable.")
    st.info("Sales, debt and working capital are displayed in ₹ crore (₹1 crore = ₹10,000,000); EPS, CMP and BVPS are per share in rupees. The current calendar year is excluded from the 20-year dividend test.")
    if pd.isna(d["price"]):
        st.error("Live CMP is unavailable. Price-dependent criteria remain N/A rather than becoming FAIL.")
    st.warning("A Graham score is a conservative value screen, not a buy signal. Financial-sector balance sheets may not report classified current assets/current liabilities, in which case those tests correctly remain N/A.")


# Sidebar navigation — permanently visible and never clipped by Streamlit's top header.
if "app_page" not in st.session_state:
    st.session_state["app_page"] = "main"

with st.sidebar:
    st.header("🧭 Dashboard Menu")

    if st.button(
        "📊 Main Dashboard",
        use_container_width=True,
        type="primary" if st.session_state["app_page"] == "main" else "secondary",
    ):
        st.session_state["app_page"] = "main"
        st.rerun()

    if st.button(
        "🚀 Value Migration / 10x–40x Hunt",
        use_container_width=True,
        type="primary" if st.session_state["app_page"] == "value_migration" else "secondary",
    ):
        st.session_state["app_page"] = "value_migration"
        st.rerun()

    if st.button(
        "🧮 Graham Value Formula",
        use_container_width=True,
        type="primary" if st.session_state["app_page"] == "graham" else "secondary",
    ):
        st.session_state["app_page"] = "graham"
        st.rerun()

    st.divider()

if st.session_state["app_page"] == "value_migration":
    render_value_migration_page()
    st.stop()

if st.session_state["app_page"] == "graham":
    render_graham_page()
    st.stop()

st.title(APP_TITLE)
st.caption("Rule: Monthly Close > all prior monthly Highs (old ATH) AND old ATH must be at least 26 months old.")

with st.sidebar:
    st.header("Scanner Settings")
    min_gap = st.number_input("Minimum ATH gap (months)", min_value=12, max_value=120, value=26, step=1)
    signal_window = st.selectbox("Show recent breakouts", [1,3,6,12,24], index=3, format_func=lambda x: f"Last {x} months")
    near_pct = st.slider("Near ATH range", 1, 10, 5, 1)
    batch_size = st.selectbox("Scan size", [50,100,200,500], index=1, help="500 may take longer because history for many stocks must be downloaded.")
    auto_fund_n = st.selectbox("Auto-fundamental top stocks", [5,10,20,30], index=1, help="Runs fundamentals only when you press the batch fundamental button.")
    st.divider()
    st.caption("Data: NIFTY Indices constituent list + Yahoo Finance price history. For research use; verify before trading.")

# Macro panel
st.subheader("🌍 Macro Pulse")
macro = macro_snapshot()
mscore, mlabel = macro_score(macro)
mc1, mc2, mc3 = st.columns([1,1,3])
mc1.metric("Macro Score", f"{mscore:.1f}/10")
regime, regime_note = classify_macro_regime(mscore, macro)
mc2.metric("Macro Regime", regime)
st.caption(f"Cycle interpretation: {regime_note}. Heuristic research classification, not an official economic-cycle designation.")
if not macro.empty:
    mc3.dataframe(macro.style.format({"Latest":"{:.2f}","1M %":"{:+.2f}%","3M %":"{:+.2f}%","12M %":"{:+.2f}%"}), use_container_width=True, hide_index=True)
else:
    mc3.warning("Macro market data unavailable right now. Scanner can still run.")


st.divider()
policy_btn = st.button("🏛️ Government Policy + 10Y Opportunity Report", use_container_width=True)
if policy_btn:
    st.session_state["show_policy_report"] = not st.session_state.get("show_policy_report", False)

if st.session_state.get("show_policy_report", False):
    policy_df = government_policy_opportunity_report()
    st.subheader("🏛️ India Government Policy → Next 10-Year Sector Opportunity")
    st.caption(
        "Research refresh: 14 Aug 2026. 10Y Opportunity Score is a policy/project-runway heuristic, "
        "not a stock-return forecast. Official source links are included for verification."
    )
    p1, p2, p3 = st.columns(3)
    p1.metric("Highest 10Y Theme", policy_df.iloc[0]["Sector / Theme"])
    p2.metric("Top Opportunity Score", f'{policy_df.iloc[0]["10Y Opportunity Score"]:.1f}/10')
    p3.metric("Themes Tracked", len(policy_df))

    st.dataframe(
        policy_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Policy Strength": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
            "10Y Opportunity Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
            "Official Source": st.column_config.LinkColumn("Official Govt Source", display_text="Open source"),
        },
    )

    top5 = policy_df.nlargest(5, "10Y Opportunity Score")[["Sector / Theme", "10Y Opportunity Score", "Likely Beneficiary Sub-sectors"]]
    st.markdown("#### 🔭 Highest-visibility 10-year themes")
    st.dataframe(top5, use_container_width=True, hide_index=True)

    st.markdown("---")
    render_policy_stock_mapping(policy_df)

    st.warning(
        "Policy support alone does not guarantee stock returns. Valuation, execution, competition, balance sheet, "
        "earnings growth and the 26M breakout/fundamental filters should still be checked."
    )
    st.download_button(
        "📄 Download Government Policy + 10Y Opportunity PDF",
        build_policy_pdf(policy_df),
        "india_government_policy_10y_opportunity_report.pdf",
        "application/pdf",
        use_container_width=True,
    )

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
    st.session_state["scan_settings"] = {
        "min_gap": int(min_gap), "signal_window": int(signal_window),
        "batch_size": int(batch_size), "industries": tuple(selected_industries),
    }

res = st.session_state.get("scan_results", pd.DataFrame())
monthlies = st.session_state.get("monthlies", {})

saved_settings = st.session_state.get("scan_settings")
current_settings = {
    "min_gap": int(min_gap), "signal_window": int(signal_window),
    "batch_size": int(batch_size), "industries": tuple(selected_industries),
}
if not res.empty and saved_settings and saved_settings != current_settings:
    st.warning("Scanner settings changed after the saved run. Run the scan again before relying on filtered results or rankings.")

if not res.empty:
    recent = res[res["Months Since Signal"].fillna(999) <= signal_window].copy()
    near = res[(res["Status"]=="Near/No Breakout") & (res["Breakout %"] >= -near_pct) & (res["Breakout %"] < 0) & (res["Months Gap"] >= min_gap)].copy()
    sector_df = sector_rotation_table(res, monthlies)
    sector_map = sector_df.set_index("Industry")["Sector Score"].to_dict() if not sector_df.empty else {}
    res["Sector Score"] = res["Industry"].map(sector_map).fillna(5.0)

    policy_vals = res["Industry"].apply(policy_support_for_industry)
    res["Policy Score"] = policy_vals.apply(lambda x: x[0])
    res["Policy Theme"] = policy_vals.apply(lambda x: x[1])
    res["Macro Score"] = mscore

    # Professional base score before fundamentals:
    # Technical 40% + Sector 25% + Macro 15% + Policy 20%
    res["Base Pro Score"] = np.clip(
        0.40*res["Combined Score"] +
        0.25*res["Sector Score"] +
        0.15*mscore +
        0.20*res["Policy Score"], 0, 10
    )

    # Carry any previously batch-scanned fundamentals into current ranking.
    auto_funds = st.session_state.get("auto_fund_scores", {})
    res["Fundamental Score %"] = res["Symbol"].map(auto_funds)
    res["Pro Final Score"] = np.where(
        res["Fundamental Score %"].notna(),
        np.clip(
            0.35*res["Combined Score"] +
            0.20*res["Sector Score"] +
            0.15*mscore +
            0.10*res["Policy Score"] +
            0.20*(res["Fundamental Score %"]/10), 0, 10
        ),
        res["Base Pro Score"]
    )

    ranked = res.sort_values(["Pro Final Score","Signal Date"], ascending=[False,False]).copy()
    st.session_state["latest_ranked"] = ranked.copy()

    a,b,c,d,e = st.columns(5)
    a.metric("Scanned", len(st.session_state.get("scan_universe", [])))
    b.metric("Recent Breakouts", len(recent))
    c.metric(f"Near ATH (≤{near_pct}%)", len(near))
    d.metric("Macro Score", f"{mscore:.1f}/10")
    e.metric("Regime", regime)

    tabs = st.tabs([
        "⭐ Top Opportunities",
        "Macro Cycle",
        "Sector Rotation",
        "Final Ranking",
        "Confirmed Breakouts",
        "Near Breakout",
        "Stock Detail + Fundamentals",
        "PDF Report"
    ])

    with tabs[0]:
        st.subheader("⭐ Professional Top Opportunities")
        st.caption(
            "Flow: Macro → Government Policy → Sector Relative Strength → 26M ATH → Fundamental Quality → Final Score. "
            "Run batch fundamentals below to convert the ranking from base score to quality-adjusted score."
        )

        b1,b2 = st.columns([2,1])
        with b1:
            st.info(
                f"Current regime: **{regime}** | Macro: **{mscore:.1f}/10**. "
                "Policy score is a curated theme-support heuristic; Sector Score includes momentum, relative strength and breakout breadth."
            )
        with b2:
            run_auto = st.button(
                f"🧪 Auto Fundamental Scan — Top {auto_fund_n}",
                type="primary",
                use_container_width=True
            )

        if run_auto:
            targets = ranked.head(int(auto_fund_n)).copy()
            prog = st.progress(0, text="Running fundamental quality scan…")
            scores = dict(st.session_state.get("auto_fund_scores", {}))
            details = dict(st.session_state.get("auto_fund_details", {}))
            for i, (_, r) in enumerate(targets.iterrows()):
                sym = r["Symbol"]
                ticker = sym + ".NS"
                try:
                    fs, assessed, fdetail = auto_fundamental_lite(ticker)
                    scores[sym] = fs
                    details[sym] = fdetail
                except Exception:
                    scores[sym] = np.nan
                prog.progress((i+1)/len(targets), text=f"Fundamentals {i+1}/{len(targets)}: {sym}")
            prog.empty()
            st.session_state["auto_fund_scores"] = scores
            st.session_state["auto_fund_details"] = details
            st.rerun()

        top = ranked.head(20).copy()
        flags = top.apply(build_pro_opportunity_flags, axis=1)
        top["Macro"] = flags.apply(lambda x: x["Macro"])
        top["Policy"] = flags.apply(lambda x: x["Policy"])
        top["Sector"] = flags.apply(lambda x: x["Sector"])
        top["26M"] = flags.apply(lambda x: x["26M"])
        top["Fundamental"] = flags.apply(lambda x: x["Fundamental"])

        display_cols = [
            "Symbol","Company","Industry","Policy Theme",
            "Macro","Policy","Sector","26M","Fundamental",
            "Status","Sector Score","Policy Score","Fundamental Score %","Pro Final Score"
        ]
        st.dataframe(
            top[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sector Score": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
                "Policy Score": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
                "Fundamental Score %": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f%%"),
                "Pro Final Score": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            }
        )
        if top["Fundamental Score %"].isna().all():
            st.warning("Fundamental column is N/A until you run the Auto Fundamental Scan.")
        else:
            st.success("Quality-adjusted ranking is active for stocks whose fundamentals were scanned.")

    with tabs[1]:
        st.subheader(f"Macro Regime: {regime}")
        st.metric("Macro Score", f"{mscore:.1f}/10")
        st.dataframe(macro, use_container_width=True, hide_index=True)
        st.info("Macro regime is a heuristic research classification using market, VIX, currency, crude, dollar, yields and gold trends.")

    with tabs[2]:
        st.subheader("Sector Rotation, Relative Strength & Breadth")
        st.caption(
            "Rotation Map: X-axis = 6M relative strength vs NIFTY; Y-axis = 3M momentum. "
            "This creates a transparent 4-quadrant research view: LEADING, IMPROVING, WEAKENING and LAGGING."
        )

        sector_view = add_rotation_quadrant(sector_df)
        st.plotly_chart(make_sector_rotation_map(sector_view), use_container_width=True)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("🟢 Leading", int((sector_view["Rotation Quadrant"]=="LEADING").sum()) if not sector_view.empty else 0)
        q2.metric("🔵 Improving", int((sector_view["Rotation Quadrant"]=="IMPROVING").sum()) if not sector_view.empty else 0)
        q3.metric("🟠 Weakening", int((sector_view["Rotation Quadrant"]=="WEAKENING").sum()) if not sector_view.empty else 0)
        q4.metric("🔴 Lagging", int((sector_view["Rotation Quadrant"]=="LAGGING").sum()) if not sector_view.empty else 0)

        st.info(
            "How to read it: LEADING = benchmark outperformance + positive recent momentum; "
            "IMPROVING = relative strength is still weak but momentum has turned positive; "
            "WEAKENING = still outperforming but recent momentum is negative; "
            "LAGGING = underperforming and recent momentum is negative. "
            "The map is a research heuristic, not a guarantee of future returns."
        )

        st.dataframe(
            sector_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sector Score":st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
                "Breadth %":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.1f%%")
            }
        )

    with tabs[3]:
        st.subheader("Final Stock Ranking")
        st.caption(
            "Without fundamentals: Technical 40% + Sector 25% + Macro 15% + Policy 20%. "
            "After batch fundamentals: Technical 35% + Sector 20% + Macro 15% + Policy 10% + Fundamentals 20%."
        )
        st.dataframe(
            ranked,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pro Final Score":st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
                "Base Pro Score":st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
                "Fundamental Score %":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f%%"),
                "Sector Score":st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
                "Policy Score":st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            }
        )

    with tabs[4]:
        show = recent.sort_values(["Combined Score","Signal Date"], ascending=[False,False]).copy()
        if show.empty:
            st.info("No qualifying recent breakout in this scan window.")
        else:
            st.dataframe(show,use_container_width=True,hide_index=True)

    with tabs[5]:
        if near.empty:
            st.info("No near-breakout candidates found.")
        else:
            st.dataframe(near.sort_values("Breakout %",ascending=False),use_container_width=True,hide_index=True)

    with tabs[6]:
        choices=ranked["Symbol"].tolist()
        sym=st.selectbox("Choose stock",choices)
        rr=ranked[ranked["Symbol"]==sym].iloc[0].to_dict()
        ticker=sym+".NS"
        m=monthlies.get(ticker,pd.DataFrame())

        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Pro Final",f"{rr['Pro Final Score']:.1f}/10")
        c2.metric("Sector",f"{rr['Sector Score']:.1f}/10")
        c3.metric("Policy",f"{rr['Policy Score']:.1f}/10")
        c4.metric("Macro",f"{mscore:.1f}/10")
        c5.metric("26M Status",str(rr['Status']))

        st.caption(f"Policy theme mapping: **{rr['Policy Theme']}**")
        if not m.empty:
            st.plotly_chart(make_chart(m,rr,sym),use_container_width=True)

        st.subheader("Fundamental Quality")
        if st.button(f"Run Detailed Fundamental Analysis — {sym}",type="primary",key=f"fund_{sym}",use_container_width=True):
            with st.spinner(f"Fetching fundamentals for {sym}…"):
                f=fetch_fundamentals(ticker)
                uni_records=st.session_state.get("scan_universe",pd.DataFrame()).to_dict("records")
                sec_pe,peer_count=peer_sector_pe(ticker,rr.get("Industry"),uni_records)
                card,fscore,passed,total=fundamental_scorecard(f,sec_pe)
                st.session_state[f"fund_result_{sym}"]=(f,sec_pe,peer_count,card,fscore,passed,total)

                auto_scores = dict(st.session_state.get("auto_fund_scores", {}))
                auto_scores[sym] = fscore
                st.session_state["auto_fund_scores"] = auto_scores

        result=st.session_state.get(f"fund_result_{sym}")
        if result:
            f,sec_pe,peer_count,card,fscore,passed,total=result
            adjusted=np.clip(
                0.35*rr['Combined Score'] +
                0.20*rr['Sector Score'] +
                0.15*mscore +
                0.10*rr['Policy Score'] +
                0.20*((fscore if pd.notna(fscore) else 50)/10), 0, 10
            )
            x1,x2,x3,x4=st.columns(4)
            x1.metric("Fundamental Score","N/A" if pd.isna(fscore) else f"{fscore:.0f}%")
            x2.metric("Passed",f"{passed}/{total}")
            x3.metric("Peer PE Count",str(peer_count))
            x4.metric("Quality-adjusted Final",f"{adjusted:.1f}/10")
            st.dataframe(card,use_container_width=True,hide_index=True)
            st.caption("Promoter trend remains N/A where reliable historical shareholding data is unavailable; verify NSE/BSE filings.")

    with tabs[7]:
        st.subheader("Research PDF Report")
        pdf=build_pdf_report(regime,mscore,sector_df,ranked.rename(columns={"Pro Final Score":"Final Stock Score"}))
        st.download_button(
            "Download PDF Report",
            pdf,
            "macro_policy_sector_26m_professional_report.pdf",
            "application/pdf",
            use_container_width=True
        )

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

st.caption("Professional research dashboard only — not investment advice. Policy mapping and macro/sector scores are heuristics. Verify exchange prices, company filings, valuations and official policy documents before acting.")
