from datetime import datetime, timezone
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from graham import fetch_graham_data

st.set_page_config(page_title="VALUE Stock Engine", page_icon="💎", layout="wide")


def text_metric(container, label, value):
    """Responsive text card for long categorical values; avoids Streamlit metric ellipsis."""
    safe_label = str(label).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    safe_value = str(value).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    container.markdown(
        f"""
        <div style="min-height:118px;padding:16px 18px;border:1px solid rgba(148,163,184,.18);
                    border-radius:16px;background:linear-gradient(145deg,rgba(19,31,54,.96),rgba(10,20,38,.96));
                    box-shadow:0 10px 28px rgba(0,0,0,.16);display:flex;flex-direction:column;justify-content:center;">
          <div style="color:#cbd5e1;font-size:.86rem;font-weight:650;margin-bottom:9px;">{safe_label}</div>
          <div style="color:#f8fafc;font-size:clamp(1.18rem,1.55vw,1.75rem);font-weight:500;
                      line-height:1.16;white-space:normal;overflow-wrap:anywhere;word-break:normal;">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1650px;}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.20); border-radius:12px; padding:10px; min-width:0;}
[data-testid="stMetricValue"] {white-space:normal; overflow:visible; text-overflow:clip;}
.value-card {border:1px solid rgba(128,128,128,.22); border-radius:14px; padding:14px; margin-bottom:8px;}
@media (max-width: 900px) {
  .block-container {padding:1.1rem .65rem 1.5rem !important;}
  h1 {font-size:1.7rem !important;}
  h2 {font-size:1.35rem !important;}
  [data-testid="stMetricValue"] {font-size:1.28rem;}
  .stDataFrame {overflow-x:auto;}
}
</style>
""", unsafe_allow_html=True)

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

VALUE_MIGRATION_MAP = [
    (["power","electric","cable","transform","transmission","energy"], 94, "Grid / Power Equipment"),
    (["renewable","solar","wind"], 95, "Renewable Energy"),
    (["defence","aerospace","ship"], 94, "Defence / Aerospace"),
    (["electronic","semiconductor","telecom","computer hardware"], 92, "Electronics / Semiconductor"),
    (["rail","transport infrastructure"], 91, "Railways / Infrastructure"),
    (["auto","automobile","battery","ev"], 87, "EV / Battery"),
    (["construction","cement","logistics","road","port"], 87, "Infrastructure / Logistics"),
    (["pharma","health","hospital","medical"], 84, "Pharma / Healthcare"),
    (["software","information technology","cyber"], 85, "AI / Digital"),
    (["chemical","industrial gas","fertilizer"], 82, "Green Hydrogen / Chemicals"),
]

VALUE_MIGRATION_BASKETS = {
    "Power Grid, Transformers & Transmission": ["POWERGRID.NS","ABB.NS","SIEMENS.NS","CGPOWER.NS","APARINDS.NS","HITACHIENER.NS","GEVERNOVA.NS","KEC.NS","KPIL.NS","POLYCAB.NS"],
    "AI Data Centre Infrastructure": ["NETWEB.NS","ANANTRAJ.NS","ABB.NS","SIEMENS.NS","CUMMINSIND.NS","BLUESTARCO.NS","VOLTAS.NS","POLYCAB.NS","KEI.NS","TECHM.NS"],
    "Battery Energy Storage (BESS) & Power Electronics": ["TATAPOWER.NS","JSWENERGY.NS","EXIDEIND.NS","AMARAJABAT.NS","WAAREEENER.NS","ABB.NS","SIEMENS.NS","CGPOWER.NS"],
    "Electronics Components / EMS / Semiconductor Ecosystem": ["DIXON.NS","KAYNES.NS","SYRMA.NS","AMBER.NS","PGEL.NS","BEL.NS","NETWEB.NS","MOSCHIP.NS"],
    "Defence Indigenisation & Component Suppliers": ["HAL.NS","BEL.NS","BDL.NS","MAZDOCK.NS","COCHINSHIP.NS","GRSE.NS","DATAPATTNS.NS","PARAS.NS"],
    "Grain / Flexible-feed Ethanol": ["BALRAMCHIN.NS","TRIVENI.NS","GLOBUSSPR.NS","RENUKA.NS","EIDPARRY.NS","BAJAJHIND.NS"],
}

VALUE_MIGRATION_ROLES = {
    "AI Data Centre Infrastructure": {
        "NETWEB.NS":"AI servers, HPC & storage", "ANANTRAJ.NS":"Data-centre operator/developer",
        "ABB.NS":"Electrical distribution & automation", "SIEMENS.NS":"Electrification & automation",
        "CUMMINSIND.NS":"Backup power systems", "BLUESTARCO.NS":"Precision cooling & HVAC",
        "VOLTAS.NS":"Cooling & HVAC", "POLYCAB.NS":"Power and data cables",
        "KEI.NS":"Power cables", "TECHM.NS":"Cloud & digital services (indirect)",
    }
}


def safe_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def normalize_ticker(sym):
    s = str(sym).strip().upper()
    return s if s.endswith(".NS") else s + ".NS"


@st.cache_data(ttl=86400, show_spinner=False)
def load_nifty500():
    errors = []
    headers = {"User-Agent":"Mozilla/5.0"}
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
            return df[["Company Name","Industry","Symbol","Ticker"]].drop_duplicates("Ticker")
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("NIFTY 500 constituent list unavailable: " + " | ".join(errors[-2:]))


@st.cache_data(ttl=1200, show_spinner=False)
def download_prices(tickers, period="10y", interval="1wk"):
    if not tickers:
        return pd.DataFrame()
    kwargs = dict(
        tickers=tickers, period=period, interval=interval, group_by="ticker",
        auto_adjust=False, threads=True, progress=False
    )
    try:
        return yf.download(timeout=30, **kwargs)
    except TypeError:
        return yf.download(**kwargs)


def extract_one(raw, ticker, total):
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
        cols = [c for c in ["Open","High","Low","Close","Volume"] if c in d.columns]
        return d[cols].dropna(how="all")
    except Exception:
        return pd.DataFrame()


def pct_return(close, periods):
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) <= periods:
        return np.nan
    b = safe_float(c.iloc[-periods-1]); e = safe_float(c.iloc[-1])
    return (e/b - 1)*100 if pd.notna(b) and b else np.nan


def value_migration_score(industry):
    text = str(industry).lower()
    for keys, score, theme in VALUE_MIGRATION_MAP:
        if any(k in text for k in keys):
            return score, theme
    return 50, "Broad / Neutral"


@st.cache_data(ttl=1200, show_spinner=False)
def macro_support():
    tickers = list(MACRO_ASSETS.values())
    try:
        raw = download_prices(tickers, period="2y", interval="1d")
    except Exception:
        raw = pd.DataFrame()
    specs = [
        ("NIFTY 50", +1, 2.2, 8.0), ("India VIX", -1, 1.7, 15.0),
        ("USD/INR", -1, 1.5, 4.0), ("Brent Crude", -1, 1.5, 12.0),
        ("Dollar Index", -1, 1.0, 6.0), ("US 10Y Yield", -1, 1.0, 8.0),
        ("Gold", -1, .6, 12.0),
    ]
    rows=[]; weighted=0.; weights=0.
    for name, direction, weight, scale in specs:
        d = extract_one(raw, MACRO_ASSETS[name], len(tickers))
        if d.empty or "Close" not in d.columns:
            continue
        c = pd.to_numeric(d["Close"], errors="coerce").dropna()
        if len(c)<65: continue
        last=safe_float(c.iloc[-1])
        r1=(last/safe_float(c.iloc[-22])-1)*100 if len(c)>22 else np.nan
        r3=(last/safe_float(c.iloc[-64])-1)*100
        impulse=.75*r3+.25*(r1 if pd.notna(r1) else r3)
        sig=float(np.tanh(direction*impulse/scale))
        weighted += sig*weight; weights += weight
        rows.append({"Driver":name,"Latest":last,"1M %":r1,"3M %":r3,
                     "Signal":"Supportive" if sig>.15 else "Adverse" if sig<-.15 else "Neutral",
                     "Contribution":sig*weight,
                     "As Of":pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d")})
    total=sum(s[2] for s in specs)
    score=float(np.clip(5+5*weighted/weights,0,10)) if weights else 5.
    coverage=int(round(100*weights/total)) if total else 0
    df=pd.DataFrame(rows)
    def v(name, col="3M %"):
        r=df[df["Driver"]==name] if not df.empty else pd.DataFrame()
        return safe_float(r.iloc[0][col]) if not r.empty else np.nan
    nifty=v("NIFTY 50"); crude=v("Brent Crude"); fx=v("USD/INR")
    vixrow=df[df["Driver"]=="India VIX"] if not df.empty else pd.DataFrame()
    vix=safe_float(vixrow.iloc[0]["Latest"]) if not vixrow.empty else np.nan
    inflation=int(pd.notna(crude) and crude>10)+int(pd.notna(fx) and fx>3)
    stress=int(pd.notna(nifty) and nifty<-5)+int(pd.notna(vix) and vix>20)
    if score>=7.6 and (pd.isna(nifty) or nifty>0) and stress==0: regime="EARLY / RISK-ON"
    elif score>=6.8 and stress==0: regime="MID CYCLE / EXPANSION"
    elif inflation>=1 and score>=3.8: regime="LATE CYCLE / INFLATION-SENSITIVE"
    elif score<3.8 or stress>=2: regime="RISK-OFF / CONTRACTION"
    else: regime="MID-TO-LATE / MIXED"
    return score, regime, coverage, df


def ath_breakout_signal(d, lookback_min_weeks=104):
    if d.empty or len(d)<lookback_min_weeks+4 or "High" not in d or "Close" not in d:
        return False, 0, ""
    x=d.dropna(subset=["High","Close"]).copy()
    start=max(lookback_min_weeks, len(x)-8)
    best=None
    for i in range(start, len(x)):
        prior=x.iloc[:i]
        ath=safe_float(prior["High"].max())
        ath_date=prior["High"].idxmax()
        age=(x.index[i]-ath_date).days/7
        close=safe_float(x["Close"].iloc[i])
        if pd.notna(close) and pd.notna(ath) and close>ath and age>=lookback_min_weeks:
            strength=min(100, 75 + max(0,(close/ath-1)*250))
            best=(True, strength, f"Weekly close broke old ATH after {age/52:.1f} years")
    return best or (False, 0, "")


def multi_year_breakout(d, years=3):
    if d.empty or len(d)<years*52+12:
        return None
    x=d.dropna(subset=["Open","High","Low","Close"]).copy()
    signals=[]
    for i in range(max(years*52, len(x)-8), len(x)):
        pre=x.iloc[max(0,i-years*52):i]
        if len(pre)<104: continue
        resistance=safe_float(pre["High"].quantile(.98))
        close=safe_float(x["Close"].iloc[i])
        if pd.isna(resistance) or pd.isna(close) or close<=resistance*1.005:
            continue
        c=pd.to_numeric(pre["Close"], errors="coerce").dropna()
        if len(c)<50: continue
        t=np.arange(len(c))
        slope=np.polyfit(t, c.values, 1)[0]
        trend_move=abs(slope*len(c))/max(1,float(c.mean()))
        width=(safe_float(c.quantile(.90))-safe_float(c.quantile(.10)))/max(1,float(c.median()))
        sideways=(trend_move<=0.30 and width<=0.55)
        typ="Multi-year Sideways Type-1" if sideways else "Multi-year Non-sideways Type-2"
        score=88 if sideways else 78
        signals.append((typ,score,f"{years}Y resistance breakout; trend={trend_move:.0%}, base width={width:.0%}"))
    return signals[-1] if signals else None


def classic_bottom_breakout(d):
    if d.empty or len(d)<90:
        return False, 0, ""
    x=d.dropna(subset=["Open","High","Low","Close"]).copy()
    if "Volume" not in x.columns:
        return False, 0, ""
    best=None
    for i in range(max(60,len(x)-8), len(x)):
        for base_weeks in (16,20,26,32,40):
            if i < base_weeks+26: continue
            base=x.iloc[i-base_weeks:i]
            prior=x.iloc[max(0,i-base_weeks-52):i-base_weeks]
            if len(base)<16 or len(prior)<20: continue
            base_high=safe_float(base["High"].max()); base_low=safe_float(base["Low"].min())
            base_mid=(base_high+base_low)/2 if pd.notna(base_high) and pd.notna(base_low) else np.nan
            width=(base_high-base_low)/base_mid if pd.notna(base_mid) and base_mid else np.nan
            prior_high=safe_float(prior["High"].max())
            drawdown=(prior_high-base_low)/prior_high if pd.notna(prior_high) and prior_high else 0
            close=safe_float(x["Close"].iloc[i]); op=safe_float(x["Open"].iloc[i]); vol=safe_float(x["Volume"].iloc[i])
            avgv=safe_float(base["Volume"].tail(min(13,len(base))).mean())
            vr=vol/avgv if pd.notna(vol) and pd.notna(avgv) and avgv>0 else np.nan
            green=pd.notna(close) and pd.notna(op) and close>op
            breakout=pd.notna(close) and pd.notna(base_high) and close>base_high*1.005
            compact=pd.notna(width) and width<=0.38
            prior_decline=drawdown>=0.30
            high_green_volume=green and pd.notna(vr) and vr>=1.35
            if breakout and compact and prior_decline and high_green_volume:
                score=float(np.clip(80 + (vr-1.35)*10 + (0.38-width)*30,0,100))
                best=(True,score,f"{base_weeks}W bottom base; prior drawdown {drawdown:.0%}; green volume {vr:.2f}x avg")
    return best or (False,0,"")


def classify_b_factor(d):
    ath, ath_score, ath_reason=ath_breakout_signal(d)
    my=multi_year_breakout(d, years=3)
    cb, cb_score, cb_reason=classic_bottom_breakout(d)
    candidates=[]
    if ath: candidates.append(("All Time High Breakout",ath_score,ath_reason))
    if my: candidates.append(my)
    if cb: candidates.append(("Classic Bottom + High Green Volume",cb_score,cb_reason))
    if not candidates:
        return "No qualifying B-Factor",0,"No current/recent match"
    return sorted(candidates,key=lambda z:z[1],reverse=True)[0]


@st.cache_data(ttl=3600, show_spinner=False)
def p_factor_snapshot(ticker):
    t=yf.Ticker(ticker)
    try: info=t.info or {}
    except Exception: info={}
    try: inc=t.financials.copy()
    except Exception: inc=pd.DataFrame()
    try: bs=t.balance_sheet.copy()
    except Exception: bs=pd.DataFrame()

    market_cap=safe_float(info.get("marketCap"))
    market_cap_cr=market_cap/1e7 if pd.notna(market_cap) else np.nan

    def row(df,names):
        if df is None or df.empty: return pd.Series(dtype=float)
        labels={str(i).lower():i for i in df.index}
        for name in names:
            if name.lower() in labels:
                s=pd.to_numeric(df.loc[labels[name.lower()]],errors="coerce").dropna()
                s.index=pd.to_datetime(s.index,errors="coerce")
                return s[s.index.notna()].sort_index()
        return pd.Series(dtype=float)

    equity=row(bs,["Stockholders Equity","Total Equity Gross Minority Interest"])
    debt=row(bs,["Total Debt"])
    profit=row(inc,["Net Income","Net Income Common Stockholders"])
    eq=safe_float(equity.iloc[-1]) if len(equity) else np.nan
    td=safe_float(info.get("totalDebt"))
    if pd.isna(td) and len(debt): td=safe_float(debt.iloc[-1])
    de=td/eq if pd.notna(td) and pd.notna(eq) and eq else np.nan

    p=profit.dropna().sort_index()
    yoy=np.nan; breakout=False
    if len(p)>=2 and p.iloc[-2] != 0:
        yoy=(p.iloc[-1]/p.iloc[-2]-1)*100
    if len(p)>=4:
        breakout=bool(p.iloc[-1] > p.iloc[:-1].tail(3).max() and pd.notna(yoy) and yoy>=20)
    elif len(p)>=2:
        breakout=bool(pd.notna(yoy) and yoy>=20)

    c1=pd.notna(market_cap_cr) and 5000<=market_cap_cr<=25000
    c2=pd.notna(de) and de<0.5
    c3=breakout
    available=sum([pd.notna(market_cap_cr),pd.notna(de),len(p)>=2])
    passed=sum([bool(c1),bool(c2),bool(c3)])
    score=100*passed/available if available else np.nan
    return {
        "Company":info.get("longName",ticker),
        "Market Cap Cr":market_cap_cr,
        "Valuation Pass":c1 if pd.notna(market_cap_cr) else None,
        "Debt/Equity":de,
        "Debt Pass":c2 if pd.notna(de) else None,
        "Profit Growth %":yoy,
        "Profit Breakout":c3 if len(p)>=2 else None,
        "P Factor Score":score,
        "P Passed":passed,
        "P Available":available,
        "Trailing PE":safe_float(info.get("trailingPE")),
    }


def sector_leadership(scan, weeklies):
    if scan.empty: return pd.DataFrame()
    try:
        bmraw=download_prices(["^NSEI"],period="2y",interval="1wk")
        bm=extract_one(bmraw,"^NSEI",1)
        bm6=pct_return(bm["Close"],26) if not bm.empty else np.nan
        bm12=pct_return(bm["Close"],52) if not bm.empty else np.nan
    except Exception:
        bm6=bm12=np.nan
    rows=[]
    for ind,g in scan.groupby("Industry"):
        r6=[];r12=[]
        for sym in g["Symbol"]:
            d=weeklies.get(sym+".NS",pd.DataFrame())
            if d.empty: continue
            a=pct_return(d["Close"],26); b=pct_return(d["Close"],52)
            if pd.notna(a):r6.append(a)
            if pd.notna(b):r12.append(b)
        m6=float(np.median(r6)) if r6 else np.nan
        m12=float(np.median(r12)) if r12 else np.nan
        rs6=m6-bm6 if pd.notna(m6) and pd.notna(bm6) else np.nan
        rs12=m12-bm12 if pd.notna(m12) and pd.notna(bm12) else np.nan
        raw=np.nanmean([x for x in [rs6,rs12] if pd.notna(x)]) if any(pd.notna(x) for x in [rs6,rs12]) else np.nan
        score=float(np.clip(50+50*np.tanh(raw/20),0,100)) if pd.notna(raw) else 50
        rows.append({"Industry":ind,"6M Return %":m6,"12M Return %":m12,"6M RS vs NIFTY %":rs6,"12M RS vs NIFTY %":rs12,"Leader Score":score})
    sec=pd.DataFrame(rows).sort_values("Leader Score",ascending=False)
    if not sec.empty:
        sec["Sector Rank"]=np.arange(1,len(sec)+1)
        sec["Leader Status"]=np.where(sec["Leader Score"]>=65,"LEADER",np.where(sec["Leader Score"]>=50,"NEUTRAL","LAGGARD"))
    return sec


def build_final(scan, leader_df, macro_score, pfactors):
    if scan.empty: return pd.DataFrame()
    d=scan.copy()
    lmap=leader_df.set_index("Industry")["Leader Score"].to_dict() if not leader_df.empty else {}
    rmap=leader_df.set_index("Industry")["Sector Rank"].to_dict() if not leader_df.empty else {}
    d["Leader Score"]=d["Industry"].map(lmap).fillna(50)
    d["Sector Rank"]=d["Industry"].map(rmap)
    d["Macro Support"]=macro_score*10
    vm=d["Industry"].apply(value_migration_score)
    fallback_score=vm.apply(lambda x:x[0]); fallback_theme=vm.apply(lambda x:x[1])
    d["Value Migration Score"]=pd.to_numeric(d.get("Selected VM Score", fallback_score),errors="coerce").fillna(fallback_score)
    d["Value Migration Theme"]=d.get("Selected VM Theme", fallback_theme).replace("",np.nan).fillna(fallback_theme)
    d["P Factor Score"]=d["Symbol"].map({k:v.get("P Factor Score",np.nan) for k,v in pfactors.items()})
    def score_row(r):
        comps=[(r["B Factor Score"],.30),(r["Leader Score"],.20),(r["Macro Support"],.10),(r["Value Migration Score"],.15)]
        if pd.notna(r["P Factor Score"]): comps.append((r["P Factor Score"],.25))
        w=sum(w for _,w in comps); return sum(v*w for v,w in comps)/w if w else np.nan
    d["VALUE Score"]=d.apply(score_row,axis=1)
    d["Coverage"]=np.where(d["P Factor Score"].notna(),"5/6 + Graham on-demand","4/6 + P/Graham pending")
    d["VALUE Status"]=np.where(d["VALUE Score"]>=80,"STRONG VALUE SETUP",np.where(d["VALUE Score"]>=65,"BUY-WATCH",np.where(d["VALUE Score"]>=50,"WATCH","LOW")))
    return d.sort_values(["VALUE Score","B Factor Score"],ascending=False)


def graham_quick(symbol):
    try:
        g=fetch_graham_data(symbol)
    except Exception as exc:
        return None,str(exc)
    tests=[
        None if pd.isna(g["sales"]) else g["sales"]>500e7,
        None if pd.isna(g["current_ratio"]) else g["current_ratio"]>2,
        None if pd.isna(g["long_debt"]) or pd.isna(g["nwc"]) else g["long_debt"]<g["nwc"],
        g["eps_positive_10y"], g["dividend20"], g["eps_growth_10y"],
        None if pd.isna(g["pe3"]) else g["pe3"]<15,
        None if pd.isna(g["pb"]) else g["pb"]<1.5,
        None if pd.isna(g["combined"]) else g["combined"]<22.5,
        None if pd.isna(g["graham_no"]) or pd.isna(g["price"]) else g["graham_no"]>g["price"],
    ]
    evaluable=[x for x in tests if x is not None]
    return {"score":100*sum(bool(x) for x in evaluable)/len(evaluable) if evaluable else np.nan,
            "passed":sum(bool(x) for x in evaluable),"assessed":len(evaluable),"raw":g},None


st.title("💎 VALUE Stock Engine")
st.caption("Your framework: Value Migration → Macro Support → P Factor → Leader → Graham Value → B Factor. Photo-derived B-Factor rules are implemented as transparent heuristics, not as guaranteed chart-pattern recognition.")

macro_score, macro_regime, macro_cov, macro_df = macro_support()

top1,top2,top3,top4=st.columns(4)
top1.metric("Macro Support",f"{macro_score*10:.0f}/100")
text_metric(top2,"Macro Regime",macro_regime)
top3.metric("Macro Coverage",f"{macro_cov}%")
text_metric(top4,"Engine","6-Factor VALUE")

with st.expander("📸 Rules taken from your notebook + chart photos", expanded=False):
    st.markdown("""
**P Factor**
- Company valuation / market-cap preference: **₹5,000 Cr to ₹25,000 Cr**.
- **Debt / Equity < 0.5**.
- **Profit Breakout:** latest annual net profit growth at least **20%** and preferably a new 3-year profit high / sudden earnings jump.

**B Factor**
- **6.1 All-Time-High Breakout** — breakout of old lifetime resistance.
- **6.2 Multi-year Breakout, Type-1** — multi-year horizontal/sideways base followed by breakout.
- **6.3 Multi-year Breakout, Type-2** — multi-year resistance breakout where the pre-breakout structure is not clean sideways.
- **6.4 Classic Bottom Breakout** — severe prior decline, roughly **4–10 month compact bottom**, no large swings, breakout on **green volume above average**.

The scanner checks recent weekly candles so a breakout remains visible for several weeks after the exact breakout candle.
""")

nav1,nav2=st.columns(2)
with nav1:
    if st.button("🚀 Open existing Value Migration module",use_container_width=True):
        st.switch_page("pages/6_Value_Migration.py")
with nav2:
    if st.button("🧮 Open existing Graham Value Formula",use_container_width=True):
        st.session_state["app_page"]="graham"
        st.switch_page("pages/5_VALUE_4_of_6_Qualification.py")

tabs=st.tabs(["🔎 VALUE Finder","🅿️ P Factor","🅱️ B Factor","🏆 Leader","🌍 Macro Support","🧮 Graham Deep Check"])

with tabs[0]:
    st.subheader("VALUE Finder — NIFTY 500")
    st.caption("First pass scans price patterns + sector leadership. P Factor fundamentals can then be batch-added to the highest-ranked setups.")
    try:
        universe=load_nifty500()
    except Exception as exc:
        st.error(str(exc)); universe=pd.DataFrame()
    if not universe.empty:
        source_options=["NIFTY 500","Value Migration Theme Stocks"]
        source_index=1 if st.session_state.get("value_universe_source")=="Value Migration Theme Stocks" else 0
        source=st.radio("Analysis universe",source_options,index=source_index,horizontal=True,key="value_universe_source")
        selected_vm_theme=""
        selected_vm_score=np.nan
        if source=="Value Migration Theme Stocks":
            themes=list(VALUE_MIGRATION_BASKETS)
            preferred=st.session_state.get("vm_selected_theme",themes[0])
            idx=themes.index(preferred) if preferred in themes else 0
            selected_vm_theme=st.selectbox("Value Migration theme",themes,index=idx,key="value_engine_vm_theme")
            st.session_state["vm_selected_theme"]=selected_vm_theme
            selected_vm_score=safe_float(st.session_state.get("vm_selected_score",np.nan))
            tickerset=set(VALUE_MIGRATION_BASKETS[selected_vm_theme])
            u=universe[universe["Ticker"].isin(tickerset)].copy()
            roles=VALUE_MIGRATION_ROLES.get(selected_vm_theme,{})
            u["Theme Role"]=u["Ticker"].map(roles).fillna("Theme beneficiary")
            st.success(f"{selected_vm_theme}: {len(u)} NIFTY 500 stocks linked for VALUE analysis.")
        else:
            u=universe.copy()
        c1,c2,c3=st.columns(3)
        batch=c1.selectbox("Universe size",[10,25,50,100,200,500],index=3,key="value_batch")
        period=c2.selectbox("Price history",["5y","10y","max"],index=1,key="value_period")
        industry_sel=c3.multiselect("Industry filter",sorted(u["Industry"].dropna().unique()),key="value_ind")
        u=u[u["Industry"].isin(industry_sel)] if industry_sel else u
        u=u.head(int(batch)).copy()
        if st.button(f"💎 Run VALUE price scan on {len(u)} stocks",type="primary",use_container_width=True):
            tickers=u["Ticker"].tolist()
            prog=st.progress(0,text="Downloading weekly price history…")
            try: raw=download_prices(tickers,period=period,interval="1wk")
            except Exception as exc:
                st.error(f"Price download failed: {exc}"); raw=pd.DataFrame()
            rec=[]; weeklies={}
            for i,(_,r) in enumerate(u.iterrows()):
                d=extract_one(raw,r["Ticker"],len(tickers)); weeklies[r["Ticker"]]=d
                typ,bscore,reason=classify_b_factor(d)
                rec.append({"Symbol":r["Symbol"],"Company":r["Company Name"],"Industry":r["Industry"],
                            "Theme Role":r.get("Theme Role",""),"Selected VM Theme":selected_vm_theme,
                            "Selected VM Score":selected_vm_score,
                            "B Factor":typ,"B Factor Score":bscore,"B Reason":reason})
                prog.progress((i+1)/max(1,len(u)),text=f"Scanning {i+1}/{len(u)}: {r['Symbol']}")
            prog.empty()
            scan=pd.DataFrame(rec)
            leaders=sector_leadership(scan,weeklies)
            st.session_state["value_scan"]=scan
            st.session_state["value_weeklies"]=weeklies
            st.session_state["value_leaders"]=leaders
            st.session_state["value_pfactors"]={}
            st.success("VALUE price-pattern + leadership scan completed.")

    scan=st.session_state.get("value_scan",pd.DataFrame())
    weeklies=st.session_state.get("value_weeklies",{})
    leaders=st.session_state.get("value_leaders",pd.DataFrame())
    pfactors=dict(st.session_state.get("value_pfactors",{}))
    if not scan.empty:
        ranking=build_final(scan,leaders,macro_score,pfactors)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Stocks scanned",len(ranking))
        m2.metric("B-Factor matches",int((ranking["B Factor Score"]>0).sum()))
        m3.metric("Sector leaders",int((leaders["Leader Status"]=="LEADER").sum()) if not leaders.empty else 0)
        m4.metric("P Factor covered",f"{ranking['P Factor Score'].notna().sum()}/{len(ranking)}")

        a,b=st.columns([2,1])
        with a:
            st.markdown("#### Ranked VALUE candidates")
        with b:
            n=st.selectbox("P-Factor batch",[5,10,20,30,50],index=1,key="pf_batch")
            runp=st.button(f"🅿️ Add P Factor — Top {n}",use_container_width=True)
        if runp:
            targets=ranking.head(int(n))
            prog=st.progress(0,text="Fetching market-cap, debt and profit history…")
            for i,(_,r) in enumerate(targets.iterrows()):
                sym=str(r["Symbol"])
                try: pfactors[sym]=p_factor_snapshot(normalize_ticker(sym))
                except Exception: pfactors[sym]={"P Factor Score":np.nan}
                prog.progress((i+1)/len(targets),text=f"P Factor {i+1}/{len(targets)}: {sym}")
            prog.empty()
            st.session_state["value_pfactors"]=pfactors
            st.rerun()

        ranking=build_final(scan,leaders,macro_score,pfactors)
        showcols=["Symbol","Company","Industry","Theme Role","VALUE Status","VALUE Score","B Factor","B Factor Score",
                  "P Factor Score","Leader Score","Sector Rank","Macro Support","Value Migration Theme",
                  "Value Migration Score","Coverage","B Reason"]
        showcols=[c for c in showcols if c in ranking.columns]
        st.dataframe(ranking[showcols].head(100),use_container_width=True,hide_index=True,
            column_config={
                "VALUE Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
                "B Factor Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
                "P Factor Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
                "Leader Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
                "Macro Support":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
                "Value Migration Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
            })
        st.info("VALUE Score currently combines B Factor 30% + P Factor 25% + Leader 20% + Value Migration 15% + Macro 10%. If P Factor is missing, available weights are normalized. Graham remains a separate deep-value confirmation.")
    else:
        st.info("Run the VALUE price scan to create the first candidate list.")

with tabs[1]:
    st.subheader("🅿️ P Factor — your 3-point filter")
    scan=st.session_state.get("value_scan",pd.DataFrame())
    opts=scan["Symbol"].tolist() if not scan.empty else []
    sym=st.selectbox("Choose scanned stock",opts,key="pstock") if opts else st.text_input("NSE Symbol",value="LT",key="pmanual").upper().strip()
    if st.button("Run P Factor analysis",type="primary",use_container_width=True,key="prun"):
        try:
            p=p_factor_snapshot(normalize_ticker(sym)); st.session_state["p_single"]=p
        except Exception as exc: st.error(str(exc))
    p=st.session_state.get("p_single")
    if p:
        a,b,c,d=st.columns(4)
        a.metric("Market Cap","N/A" if pd.isna(p["Market Cap Cr"]) else f"₹{p['Market Cap Cr']:,.0f} Cr")
        b.metric("Debt / Equity","N/A" if pd.isna(p["Debt/Equity"]) else f"{p['Debt/Equity']:.2f}")
        c.metric("Profit Growth","N/A" if pd.isna(p["Profit Growth %"]) else f"{p['Profit Growth %']:+.1f}%")
        d.metric("P Factor Score","N/A" if pd.isna(p["P Factor Score"]) else f"{p['P Factor Score']:.0f}/100")
        rows=[
            {"P Factor Rule":"3.1 Company valuation / market cap ₹5,000–25,000 Cr","Value":"N/A" if pd.isna(p["Market Cap Cr"]) else f"₹{p['Market Cap Cr']:,.0f} Cr","Result":"N/A" if p["Valuation Pass"] is None else "PASS" if p["Valuation Pass"] else "FAIL"},
            {"P Factor Rule":"3.2 Debt / Equity < 0.5","Value":"N/A" if pd.isna(p["Debt/Equity"]) else f"{p['Debt/Equity']:.2f}","Result":"N/A" if p["Debt Pass"] is None else "PASS" if p["Debt Pass"] else "FAIL"},
            {"P Factor Rule":"3.3 Profit Breakout: ≥20% YoY + new recent profit high","Value":"N/A" if pd.isna(p["Profit Growth %"]) else f"{p['Profit Growth %']:+.1f}%","Result":"N/A" if p["Profit Breakout"] is None else "PASS" if p["Profit Breakout"] else "FAIL"},
        ]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.caption("Your note says a breakout year / sudden net-profit increase is preferred. The engine therefore requires ≥20% YoY growth and a fresh recent profit high when enough annual history is available.")

with tabs[2]:
    st.subheader("🅱️ B Factor — photo-derived breakout engine")
    st.markdown("""
**6.1 ATH Breakout** → weekly close above old lifetime high, old high roughly ≥2 years old.  
**6.2 Type-1 Multi-year** → 3-year resistance breakout after a relatively horizontal / sideways structure.  
**6.3 Type-2 Multi-year** → 3-year resistance breakout but the prior structure is directional / not clean sideways.  
**6.4 Classic Bottom** → severe prior decline + compact 4–10 month bottom + breakout + green volume ≥1.35× recent average.
""")
    scan=st.session_state.get("value_scan",pd.DataFrame())
    if scan.empty:
        st.info("Run VALUE Finder first.")
    else:
        only=scan[scan["B Factor Score"]>0].sort_values("B Factor Score",ascending=False)
        st.dataframe(only,use_container_width=True,hide_index=True,
                     column_config={"B Factor Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f")})
        if not only.empty:
            sym=st.selectbox("Chart a B-Factor match",only["Symbol"].tolist(),key="bchart")
            d=st.session_state.get("value_weeklies",{}).get(sym+".NS",pd.DataFrame())
            if not d.empty:
                fig=go.Figure(go.Candlestick(x=d.index,open=d["Open"],high=d["High"],low=d["Low"],close=d["Close"],name=sym))
                fig.update_layout(height=560,xaxis_rangeslider_visible=False,title=f"{sym} — Weekly B-Factor review")
                st.plotly_chart(fig,use_container_width=True)

with tabs[3]:
    st.subheader("🏆 Leader — sector vs NIFTY 50 and among all sectors")
    leaders=st.session_state.get("value_leaders",pd.DataFrame())
    if leaders.empty:
        st.info("Run VALUE Finder first. Leader ranking is calculated from the same scanned universe.")
    else:
        st.dataframe(leaders,use_container_width=True,hide_index=True,
            column_config={"Leader Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f")})
        st.caption("Leader Score uses median 6M and 12M sector returns relative to NIFTY 50. Sector Rank compares every evaluated sector with the others.")

with tabs[4]:
    st.subheader("🌍 Macro Support")
    a,b,c=st.columns(3)
    a.metric("Macro Support",f"{macro_score*10:.0f}/100")
    text_metric(b,"Regime",macro_regime)
    c.metric("Coverage",f"{macro_cov}%")
    if not macro_df.empty:
        st.dataframe(macro_df.sort_values("Contribution",ascending=False),use_container_width=True,hide_index=True)
    st.caption("This is dynamic, not hard-coded. It changes with NIFTY, VIX, INR, crude, DXY, US 10Y and gold.")

with tabs[5]:
    st.subheader("🧮 Graham Deep Value confirmation")
    scan=st.session_state.get("value_scan",pd.DataFrame())
    opts=scan["Symbol"].tolist() if not scan.empty else []
    sym=st.selectbox("Stock",opts,key="gstock") if opts else st.text_input("NSE Symbol",value="LT",key="gmanual").upper().strip()
    if st.button("Run Graham deep check",type="primary",use_container_width=True,key="grun"):
        result,error=graham_quick(sym)
        if error: st.error(error)
        else: st.session_state["value_graham"]=result
    gr=st.session_state.get("value_graham")
    if gr:
        a,b,c=st.columns(3)
        a.metric("Graham Score","N/A" if pd.isna(gr["score"]) else f"{gr['score']:.0f}%")
        b.metric("Passed",f"{gr['passed']}/{gr['assessed']}")
        raw=gr["raw"]
        c.metric("CMP","N/A" if pd.isna(raw["price"]) else f"₹{raw['price']:,.2f}")
        st.caption(f"Source: {raw['sources']} · Retrieved {raw['as_of']}. Use the existing Graham page for the full 10-rule table.")

st.divider()
st.caption("VALUE Stock Engine v1 · Your photo/notebook rules have been converted into explicit, testable heuristics. Pattern scores are research filters—not guaranteed chart recognition, buy/sell advice, or return forecasts. Verify exchange prices, company filings, promoter/shareholding and valuation before acting.")
