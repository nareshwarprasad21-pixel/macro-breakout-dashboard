import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Professional Research Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;padding-bottom:2rem;max-width:1600px}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.20);border-radius:12px;padding:10px}
.hero-card{padding:22px 24px;border-radius:20px;border:1px solid rgba(120,120,120,.20);box-shadow:0 8px 24px rgba(0,0,0,.08);margin-bottom:14px}
.hero-green{background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(34,197,94,.06));border-left:6px solid #16a34a}
.hero-yellow{background:linear-gradient(135deg,rgba(250,204,21,.20),rgba(245,158,11,.06));border-left:6px solid #eab308}
.hero-orange{background:linear-gradient(135deg,rgba(251,146,60,.20),rgba(249,115,22,.06));border-left:6px solid #f97316}
.hero-red{background:linear-gradient(135deg,rgba(248,113,113,.20),rgba(220,38,38,.06));border-left:6px solid #dc2626}
.regime-box{padding:14px 16px;border-radius:14px;border:1px solid rgba(120,120,120,.18);min-height:135px}
.green-box{background:rgba(34,197,94,.10)}.yellow-box{background:rgba(234,179,8,.12)}.orange-box{background:rgba(249,115,22,.12)}.red-box{background:rgba(220,38,38,.10)}
.support-box{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.28);padding:16px;border-radius:14px;min-height:180px}
.negative-box{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.28);padding:16px;border-radius:14px;min-height:180px}
.confirm-box{background:rgba(234,179,8,.09);border:1px solid rgba(234,179,8,.30);padding:16px;border-radius:14px;min-height:180px}
.flow-card{padding:15px 12px;border-radius:14px;border:1px solid rgba(120,120,120,.18);text-align:center;min-height:110px;background:rgba(99,102,241,.06)}
.swing-card{padding:18px;border-radius:16px;border:1px solid rgba(59,130,246,.22);background:linear-gradient(135deg,rgba(59,130,246,.10),rgba(168,85,247,.06));min-height:145px}
.small-note{font-size:.88rem;opacity:.78}
@media(max-width:900px){.block-container{padding:1rem .7rem 1.5rem!important}.hero-card{padding:16px}.regime-box,.support-box,.negative-box,.confirm-box{min-height:auto}.stDataFrame{overflow-x:auto}h1{font-size:1.7rem!important}h2{font-size:1.35rem!important}}
</style>
""", unsafe_allow_html=True)

NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
MACRO_ASSETS = {
    "NIFTY 50": "^NSEI", "India VIX": "^INDIAVIX", "USD/INR": "INR=X",
    "Brent Crude": "BZ=F", "Dollar Index": "DX-Y.NYB", "US 10Y Yield": "^TNX", "Gold": "GC=F",
}


def safe_float(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def month_diff(later, earlier):
    return (later.year-earlier.year)*12 + (later.month-earlier.month)


def period_return(close, periods):
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) <= periods:
        return np.nan
    base, last = safe_float(c.iloc[-periods-1]), safe_float(c.iloc[-1])
    return (last/base-1)*100 if pd.notna(base) and base != 0 else np.nan


@st.cache_data(ttl=86400, show_spinner=False)
def load_nifty500():
    headers = {"User-Agent": "Mozilla/5.0"}
    errors = []
    for url in NIFTY500_URLS:
        try:
            r = requests.get(url, timeout=15, headers=headers); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if "Symbol" not in df.columns: continue
            if "Industry" not in df.columns: df["Industry"] = "Unknown"
            if "Company Name" not in df.columns: df["Company Name"] = df["Symbol"]
            df["Ticker"] = df["Symbol"].astype(str).str.strip()+".NS"
            return df[["Company Name","Industry","Symbol","Ticker"]].drop_duplicates("Ticker")
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("NIFTY 500 list unavailable: " + " | ".join(errors[-2:]))


@st.cache_data(ttl=1200, show_spinner=False)
def download_prices(tickers, period="2y", auto_adjust=False):
    if not tickers: return pd.DataFrame()
    kwargs = dict(tickers=tickers, period=period, interval="1d", group_by="ticker", auto_adjust=auto_adjust, threads=True, progress=False)
    try: return yf.download(timeout=25, **kwargs)
    except TypeError: return yf.download(**kwargs)


def extract_one(raw, ticker, total):
    try:
        if raw.empty: return pd.DataFrame()
        if total == 1 and not isinstance(raw.columns, pd.MultiIndex): d = raw.copy()
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0): d = raw[ticker].copy()
        elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(-1): d = raw.xs(ticker, axis=1, level=-1).copy()
        else: return pd.DataFrame()
        cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in d.columns]
        return d[cols].dropna(how="all")
    except Exception:
        return pd.DataFrame()


def to_monthly(d):
    if d is None or d.empty or "Close" not in d.columns or "High" not in d.columns: return pd.DataFrame()
    x = d.copy(); x.index = pd.to_datetime(x.index).tz_localize(None)
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    return x.resample("ME").agg({k:v for k,v in agg.items() if k in x.columns}).dropna(subset=["High","Close"])


def strict_26m_signal(monthly, min_gap=26):
    if monthly is None or len(monthly) < min_gap+2: return None
    m = monthly.dropna(subset=["High","Close"]).copy(); highs = pd.to_numeric(m["High"], errors="coerce"); closes = pd.to_numeric(m["Close"], errors="coerce")
    signals = []
    for i in range(1, len(m)):
        prior = highs.iloc[:i].dropna()
        if prior.empty: continue
        old_ath = float(prior.max()); dates = prior.index[np.isclose(prior.values, old_ath, rtol=1e-10, atol=1e-10)]; old_date = dates[-1] if len(dates) else prior.idxmax(); gap = month_diff(m.index[i], old_date); close = safe_float(closes.iloc[i])
        if pd.notna(close) and close > old_ath and gap >= min_gap:
            avg12 = safe_float(m["Volume"].iloc[max(0,i-12):i].mean()) if "Volume" in m.columns else np.nan; vol = safe_float(m["Volume"].iloc[i]) if "Volume" in m.columns else np.nan
            signals.append({"Signal Date":m.index[i],"Old ATH":old_ath,"ATH Date":old_date,"Months Gap":gap,"Monthly Close":close,"Breakout %":(close/old_ath-1)*100,"Volume Ratio":vol/avg12 if pd.notna(vol) and pd.notna(avg12) and avg12>0 else np.nan})
    if signals:
        sig = signals[-1]; age = month_diff(m.index[-1], sig["Signal Date"]); sig["Months Since Signal"] = age; sig["Status"] = "Fresh Breakout" if age==0 else "Breakout <=3M" if age<=3 else "Older Breakout"; return sig
    prior = highs.iloc[:-1].dropna()
    if prior.empty: return None
    old_ath = float(prior.max()); dates = prior.index[np.isclose(prior.values,old_ath,rtol=1e-10,atol=1e-10)]; old_date = dates[-1] if len(dates) else prior.idxmax(); close = safe_float(closes.iloc[-1])
    return {"Status":"Near/No Breakout","Signal Date":pd.NaT,"Old ATH":old_ath,"ATH Date":old_date,"Months Gap":month_diff(m.index[-1],old_date),"Monthly Close":close,"Breakout %":(close/old_ath-1)*100 if pd.notna(close) else np.nan,"Volume Ratio":np.nan,"Months Since Signal":np.nan}


@st.cache_data(ttl=1200, show_spinner=False)
def macro_snapshot():
    tickers = list(MACRO_ASSETS.values())
    try: raw = download_prices(tickers, period="2y")
    except Exception: raw = pd.DataFrame()
    rows=[]
    for name,ticker in MACRO_ASSETS.items():
        d=extract_one(raw,ticker,len(tickers))
        if d.empty or "Close" not in d.columns: continue
        c=pd.to_numeric(d["Close"],errors="coerce").dropna()
        if len(c)<25: continue
        rows.append({"Indicator":name,"Ticker":ticker,"Latest":safe_float(c.iloc[-1]),"1M %":period_return(c,21),"3M %":period_return(c,63),"12M %":period_return(c,252),"As Of":pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


def long_term_macro_engine(mdf):
    specs=[
        ("NIFTY 50",+1,2.2,10.0,"Domestic equity trend"),("India VIX",-1,1.6,18.0,"Volatility / fear"),("USD/INR",-1,1.5,6.0,"Rupee / imported inflation pressure"),
        ("Brent Crude",-1,1.5,18.0,"India import-cost pressure"),("Dollar Index",-1,1.0,8.0,"Global dollar liquidity"),("US 10Y Yield",-1,1.0,10.0,"Global cost of money"),("Gold",-1,0.6,18.0,"Defensive / uncertainty proxy")]
    if mdf is None or mdf.empty: return 5.0,"🟡 MIXED / SELECTIVE",0,pd.DataFrame(),"Insufficient live inputs"
    total=sum(x[2] for x in specs); weighted=0.0; used=0.0; rows=[]
    for name,direction,weight,scale,meaning in specs:
        r=mdf[mdf["Indicator"]==name]
        if r.empty: continue
        r1,r3,r12=[safe_float(r.iloc[0][c]) for c in ["1M %","3M %","12M %"]]
        parts=[]
        if pd.notna(r1): parts.append((r1,.15))
        if pd.notna(r3): parts.append((r3,.40))
        if pd.notna(r12): parts.append((r12,.45))
        if not parts: continue
        move=sum(v*w for v,w in parts)/sum(w for _,w in parts); raw=float(np.tanh(direction*move/scale)); weighted+=raw*weight; used+=weight
        signal="🟢 Supportive" if raw>.15 else "🔴 Negative" if raw<-.15 else "🟡 Neutral/Mixed"
        rows.append({"Driver":name,"Meaning":meaning,"1M %":r1,"3M %":r3,"12M %":r12,"Contribution":raw*weight,"Signal":signal})
    if used==0: return 5.0,"🟡 MIXED / SELECTIVE",0,pd.DataFrame(rows),"No evaluable inputs"
    score=float(np.clip(5+5*weighted/used,0,10)); coverage=int(round(100*used/total))
    regime="🟢 SUPPORTIVE / RISK-ON" if score>=7.5 else "🟡 MIXED / SELECTIVE" if score>=5 else "🟠 CAUTIOUS" if score>=3 else "🔴 RISK-OFF"
    return score,regime,coverage,pd.DataFrame(rows).sort_values("Contribution",ascending=False),f"{coverage}% weighted input coverage"


def regime_strategy(score):
    if score>=7.5: return "🟢","SUPPORTIVE / RISK-ON","More opportunities can be explored; still require sector leadership, quality and valid breakout structure.","hero-green"
    if score>=5: return "🟡","MIXED / SELECTIVE","Only sector leaders + strong fundamentals + confirmed breakouts. Avoid broad-market buying.","hero-yellow"
    if score>=3: return "🟠","CAUTIOUS","Reduce position size, demand stronger confirmation and stay highly selective.","hero-orange"
    return "🔴","RISK-OFF","Avoid aggressive fresh buying; prioritize capital protection and only exceptional setups.","hero-red"


def sector_table(results, monthlies, benchmark_monthly):
    if results is None or results.empty: return pd.DataFrame()
    bm6=period_return(benchmark_monthly["Close"],6) if not benchmark_monthly.empty else np.nan; rows=[]
    for industry,group in results.groupby("Industry"):
        r3s=[];r6s=[];r12s=[]
        for sym in group["Symbol"].head(40):
            m=monthlies.get(sym+".NS",pd.DataFrame())
            if m.empty: continue
            r3s.append(period_return(m["Close"],3));r6s.append(period_return(m["Close"],6));r12s.append(period_return(m["Close"],12))
        clean=lambda xs:[x for x in xs if pd.notna(x)];r3s,r6s,r12s=clean(r3s),clean(r6s),clean(r12s)
        m3=float(np.median(r3s)) if r3s else np.nan;m6=float(np.median(r6s)) if r6s else np.nan;m12=float(np.median(r12s)) if r12s else np.nan;rs6=m6-bm6 if pd.notna(m6) and pd.notna(bm6) else np.nan
        recent=int((group["Months Since Signal"].fillna(999)<=3).sum());near=int(((group["Status"]=="Near/No Breakout")&(group["Breakout %"]>=-5)).sum());breadth=100*(recent+.5*near)/max(1,len(group))
        comps=[(np.tanh(v/s),w) for v,s,w in [(m3,8,.25),(m6,12,.30),(m12,22,.20),(rs6,10,.25)] if pd.notna(v)];mom=sum(v*w for v,w in comps)/sum(w for _,w in comps) if comps else 0;score=float(np.clip(5+3.4*mom+min(1.8,breadth/30),0,10))
        quadrant="LEADING" if pd.notna(rs6) and rs6>=0 and pd.notna(m3) and m3>=0 else "IMPROVING" if pd.notna(m3) and m3>=0 else "WEAKENING" if pd.notna(rs6) and rs6>=0 else "LAGGING"
        rows.append({"Industry":industry,"3M Momentum %":m3,"6M Momentum %":m6,"12M Momentum %":m12,"6M RS vs NIFTY %":rs6,"Breakout Breadth %":breadth,"Sector Score":score,"Rotation Quadrant":quadrant,"Stocks Evaluated":len(group)})
    return pd.DataFrame(rows).sort_values("Sector Score",ascending=False)


def rotation_chart(df):
    fig=go.Figure()
    if df is None or df.empty:return fig
    d=df.dropna(subset=["6M RS vs NIFTY %","3M Momentum %"]).copy()
    colors={"LEADING":"#22c55e","IMPROVING":"#3b82f6","WEAKENING":"#f59e0b","LAGGING":"#ef4444"}
    for q in ["LEADING","IMPROVING","WEAKENING","LAGGING"]:
        x=d[d["Rotation Quadrant"]==q]
        if x.empty:continue
        fig.add_trace(go.Scatter(x=x["6M RS vs NIFTY %"],y=x["3M Momentum %"],mode="markers+text",text=x["Industry"],textposition="top center",name=q,marker={"size":14+4*x["Sector Score"].clip(0,10),"opacity":.75,"color":colors[q]},customdata=np.stack([x["Sector Score"],x["Breakout Breadth %"]],axis=-1),hovertemplate="<b>%{text}</b><br>RS6: %{x:.2f}%<br>M3: %{y:.2f}%<br>Score: %{customdata[0]:.1f}/10<br>Breadth: %{customdata[1]:.1f}%<extra></extra>"))
    fig.add_vline(x=0,line_dash="dash");fig.add_hline(y=0,line_dash="dash");fig.update_layout(height=560,xaxis_title="6M Relative Strength vs NIFTY (%)",yaxis_title="3M Momentum (%)",margin=dict(l=10,r=10,t=35,b=10));return fig


@st.cache_data(ttl=3600, show_spinner=False)
def fundamental_snapshot(ticker):
    stock=yf.Ticker(ticker)
    try: info=stock.info or {}
    except Exception: info={}
    try: inc=stock.financials.copy()
    except Exception: inc=pd.DataFrame()
    try: bs=stock.balance_sheet.copy()
    except Exception: bs=pd.DataFrame()
    def row(df,names):
        if df is None or df.empty:return pd.Series(dtype=float)
        labels={str(i).lower():i for i in df.index}
        for name in names:
            if name.lower() in labels:
                s=pd.to_numeric(df.loc[labels[name.lower()]],errors="coerce").dropna();s.index=pd.to_datetime(s.index,errors="coerce");return s[s.index.notna()].sort_index()
        return pd.Series(dtype=float)
    revenue=row(inc,["Total Revenue","Operating Revenue"]);profit=row(inc,["Net Income","Net Income Common Stockholders"]);ebit=row(inc,["EBIT","Operating Income"]);assets=row(bs,["Total Assets"]);cl=row(bs,["Current Liabilities","Total Current Liabilities"]);equity=row(bs,["Stockholders Equity","Total Equity Gross Minority Interest"]);debt=row(bs,["Total Debt"])
    def cagr(s):
        s=s.dropna().sort_index()
        if len(s)<2 or s.iloc[0]<=0 or s.iloc[-1]<=0:return np.nan,0
        years=(s.index[-1]-s.index[0]).days/365.25;return ((s.iloc[-1]/s.iloc[0])**(1/years)-1)*100 if years>0 else np.nan,years
    sales_cagr,sales_years=cagr(revenue);profit_cagr,profit_years=cagr(profit);eq=safe_float(equity.iloc[-1]) if len(equity) else np.nan;total_debt=safe_float(info.get("totalDebt"));total_debt=safe_float(debt.iloc[-1]) if pd.isna(total_debt) and len(debt) else total_debt;de=total_debt/eq if pd.notna(total_debt) and pd.notna(eq) and eq!=0 else np.nan;latest_ebit=safe_float(ebit.iloc[-1]) if len(ebit) else np.nan;common=assets.index.intersection(cl.index);cap=safe_float(assets.loc[common[-1]]-cl.loc[common[-1]]) if len(common) else np.nan;roce=latest_ebit/cap*100 if pd.notna(latest_ebit) and pd.notna(cap) and cap!=0 else np.nan;p=profit.dropna().sort_index();increasing=bool(len(p)>=3 and all(p.iloc[i]>=p.iloc[i-1] for i in range(1,len(p)))) if len(p)>=3 else None;pe=safe_float(info.get("trailingPE"))
    checks=[("Debt / Equity < 0.5",de,pd.notna(de),lambda x:x<.5),("ROCE > 15%",roce,pd.notna(roce),lambda x:x>15),("Net profit increasing",increasing,increasing is not None,lambda x:x is True),("Sales CAGR > 20% for >=5Y",sales_cagr,pd.notna(sales_cagr) and sales_years>=5,lambda x:x>20),("Profit CAGR > 15% for >=5Y",profit_cagr,pd.notna(profit_cagr) and profit_years>=5,lambda x:x>15)]
    rows=[];passed=0;assessed=0
    for name,value,available,rule in checks:
        ok=bool(rule(value)) if available else None
        if ok is not None:assessed+=1;passed+=int(ok)
        rows.append({"Criterion":name,"Value":"N/A" if not available else ("Yes" if value is True else "No" if value is False else f"{value:.2f}"),"Result":"N/A" if ok is None else "PASS" if ok else "FAIL"})
    return {"Company":info.get("longName") or ticker,"Sector":info.get("sector") or "N/A","Industry":info.get("industry") or "N/A","PE":pe,"Score":100*passed/assessed if assessed else np.nan,"Passed":passed,"Assessed":assessed,"Checks":pd.DataFrame(rows),"Source":"Yahoo Finance quote + reported financial statements","Retrieved":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}


st.title("🧠 Professional Investment Research Lab")
st.caption("One integrated workspace for positional macro regime, swing roadmap, strict 26M ATH, sector rotation and fundamentals. Research tool, not investment advice.")

macro=macro_snapshot();score,regime,coverage,drivers,regime_note=long_term_macro_engine(macro);icon,regime_name,strategy,hero_class=regime_strategy(score)

m1,m2,m3,m4=st.columns(4);m1.metric("Macro Score",f"{score:.1f}/10");m2.metric("Macro Regime",regime_name);m3.metric("Macro Coverage",f"{coverage}%");m4.metric("Live Data TTL","20 min")

with st.expander("🔎 Data provenance & freshness", expanded=False):
    st.markdown("**Universe:** official NIFTY Indices constituent CSV. **Market prices / macro / fundamentals:** Yahoo Finance via `yfinance`. These are research feeds, not an exchange-certified real-time terminal.")
    if not macro.empty: st.caption(f"Macro observation dates: {macro['As Of'].min()} to {macro['As Of'].max()}. {regime_note}")
    st.warning("Before order execution, verify price and filings on NSE/BSE/company filings or your broker terminal.")

main_tabs=st.tabs(["🌐 Long-Term / Positional","⚡ Swing Trading","📈 26M ATH Scanner","🔄 Sector Rotation","🧾 Fundamentals"])

with main_tabs[0]:
    st.markdown(f'<div class="hero-card {hero_class}"><div style="font-size:.92rem;font-weight:700;letter-spacing:.06em;opacity:.75">CURRENT MACRO REGIME</div><div style="font-size:2rem;font-weight:800;margin:4px 0 2px">{icon} {regime_name}</div><div style="font-size:1.18rem;font-weight:700">Macro Score: {score:.1f} / 10 &nbsp; | &nbsp; Coverage: {coverage}%</div><div style="margin-top:10px;font-size:1.02rem"><b>Recommended Strategy:</b> {strategy}</div></div>',unsafe_allow_html=True)
    st.markdown("### 🎚️ Permanent Macro Regime Scale")
    c1,c2,c3,c4=st.columns(4)
    c1.markdown('<div class="regime-box green-box"><b>🟢 7.5–10<br>SUPPORTIVE / RISK-ON</b><br><br>Macro equities के favour में.<br><b>Action:</b> ज्यादा opportunities खोजें.</div>',unsafe_allow_html=True)
    c2.markdown('<div class="regime-box yellow-box"><b>🟡 5.0–7.49<br>MIXED / SELECTIVE</b><br><br>Positive + negative signals दोनों.<br><b>Action:</b> sector leaders + quality breakouts.</div>',unsafe_allow_html=True)
    c3.markdown('<div class="regime-box orange-box"><b>🟠 3.0–4.99<br>CAUTIOUS</b><br><br>Macro headwinds बढ़ रहे हैं.<br><b>Action:</b> smaller size + highly selective.</div>',unsafe_allow_html=True)
    c4.markdown('<div class="regime-box red-box"><b>🔴 0–2.99<br>RISK-OFF</b><br><br>Macro environment adverse.<br><b>Action:</b> aggressive fresh buying avoid.</div>',unsafe_allow_html=True)
    st.markdown("### 🔍 Why this regime?")
    supportive=drivers[drivers["Signal"].str.contains("Supportive",na=False)] if not drivers.empty else pd.DataFrame();negative=drivers[drivers["Signal"].str.contains("Negative",na=False)] if not drivers.empty else pd.DataFrame()
    a,b,c=st.columns(3)
    with a:
        items="".join([f"<li><b>{r['Driver']}</b> — {r['Meaning']}</li>" for _,r in supportive.iterrows()]) or "<li>No strong supportive driver</li>";st.markdown(f'<div class="support-box"><b>🟢 SUPPORTIVE DRIVERS</b><ul>{items}</ul></div>',unsafe_allow_html=True)
    with b:
        items="".join([f"<li><b>{r['Driver']}</b> — {r['Meaning']}</li>" for _,r in negative.iterrows()]) or "<li>No strong negative driver</li>";st.markdown(f'<div class="negative-box"><b>🔴 NEGATIVE DRIVERS</b><ul>{items}</ul></div>',unsafe_allow_html=True)
    with c:
        n=macro[macro["Indicator"]=="NIFTY 50"] if not macro.empty else pd.DataFrame();conf="Nifty confirmation unavailable" if n.empty else f"Nifty: 1M {safe_float(n.iloc[0]['1M %']):.2f}% · 3M {safe_float(n.iloc[0]['3M %']):.2f}% · 12M {safe_float(n.iloc[0]['12M %']):.2f}%";st.markdown(f'<div class="confirm-box"><b>🟡 MARKET CONFIRMATION</b><br><br>{conf}<br><br>{regime_note}</div>',unsafe_allow_html=True)
    st.markdown("### 🧠 Positional Decision Chain")
    cols=st.columns(6);steps=[("🌐","MACRO REGIME","Background filter"),("🔄","SECTOR ROTATION","Where money is moving"),("🥇","WEEKLY LEADERS","Strongest sectors"),("📈","26M ATH","Major breakout"),("🧾","FUNDAMENTALS","Quality confirmation"),("⭐","FINAL OPPORTUNITY","Rank + decision")]
    for col,(em,title,desc) in zip(cols,steps): col.markdown(f'<div class="flow-card"><div style="font-size:1.5rem">{em}</div><b>{title}</b><br><span class="small-note">{desc}</span></div>',unsafe_allow_html=True)
    st.markdown("### 📊 Macro Regime Diagnostics")
    if macro.empty: st.warning("Macro feed unavailable right now.")
    else: st.dataframe(macro,use_container_width=True,hide_index=True);st.markdown("#### Driver Attribution");st.dataframe(drivers,use_container_width=True,hide_index=True)
    st.info("Positional rule: 1M gets low weight; 3M and 12M get higher weight. Macro is a background filter, not a direct buy/sell signal.")

with main_tabs[1]:
    st.markdown("### ⚡ Swing Trading Roadmap")
    st.caption("Swing में Macro का weight positional से कम रहेगा. Actual trade selection = sector leadership + strong stock + daily technical setup.")
    cols=st.columns(6);steps=[("📍","MARKET CONDITION","Nifty daily/weekly trend"),("🥇","SECTOR LEADER","Weekly outperformance"),("💪","STRONG STOCK","Stock vs sector/Nifty RS"),("📈","TECHNICAL SETUP","Breakout / pullback / base"),("🛑","ENTRY + SL","Structure based risk"),("🎯","TARGET / TRAIL","R:R + trailing stop")]
    for col,(em,title,desc) in zip(cols,steps): col.markdown(f'<div class="swing-card"><div style="font-size:1.55rem">{em}</div><b>{title}</b><br><span class="small-note">{desc}</span></div>',unsafe_allow_html=True)
    st.markdown("### 🎯 Proposed Swing Score")
    swing_df=pd.DataFrame([("Weekly Sector Leadership",20,"Leader / Improving sector"),("Stock Relative Strength",20,"Stock outperforming sector + Nifty"),("Daily Trend",15,"Higher highs/lows; price above key averages"),("Breakout / Pullback Setup",20,"Confirmed daily setup"),("Volume Confirmation",10,"Healthy expansion / participation"),("Macro Background",5,"Low weight for swing"),("Risk : Reward",10,"Prefer >= 2:1 where structure supports")],columns=["Component","Weight %","What to check"])
    st.dataframe(swing_df,use_container_width=True,hide_index=True,column_config={"Weight %":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%d%%")})
    st.success("Swing decision rule: Macro Mixed होने पर भी Leader Sector + Strong Stock + Confirmed Daily Setup मिल रहा हो तो trade candidate हो सकता है. Risk-Off में confirmation और position sizing अधिक strict रखें.")
    st.info("यह tab अभी roadmap/decision framework है. अगला upgrade live Swing Scanner होगा जो leader sector से stocks scan करके technical score, Entry, SL और target candidates निकालेगा.")

with main_tabs[2]:
    st.subheader("Strict monthly 26M ATH scanner")
    st.caption("Signal = monthly CLOSE above every prior monthly HIGH and old ATH month >= selected gap. Unfinished month can change before month-end.")
    try: universe=load_nifty500()
    except Exception as exc: st.error(str(exc));universe=pd.DataFrame()
    if not universe.empty:
        c1,c2,c3=st.columns(3);min_gap=c1.number_input("Minimum ATH gap",12,120,26,1);batch=c2.selectbox("Universe size",[50,100,200,500],index=1);near_pct=c3.slider("Near ATH range %",1,10,5,1);industries=st.multiselect("Industry filter (optional)",sorted(universe["Industry"].dropna().unique()));scan_u=universe[universe["Industry"].isin(industries)] if industries else universe;scan_u=scan_u.head(int(batch)).copy()
        if st.button(f"Run professional scan on {len(scan_u)} stocks",type="primary",use_container_width=True):
            tickers=scan_u["Ticker"].tolist();prog=st.progress(0,text="Downloading price history…")
            try: raw=download_prices(tickers,period="max")
            except Exception as exc: st.error(f"Price download failed: {exc}");raw=pd.DataFrame()
            records=[];monthlies={}
            for i,(_,r) in enumerate(scan_u.iterrows()):
                d=extract_one(raw,r["Ticker"],len(tickers));m=to_monthly(d);monthlies[r["Ticker"]]=m;sig=strict_26m_signal(m,int(min_gap))
                if sig: records.append({"Symbol":r["Symbol"],"Company":r["Company Name"],"Industry":r["Industry"],**sig})
                prog.progress((i+1)/max(1,len(scan_u)),text=f"Scanning {i+1}/{len(scan_u)}: {r['Symbol']}")
            prog.empty();st.session_state["pro_scan"]=pd.DataFrame(records);st.session_state["pro_monthlies"]=monthlies;st.session_state["pro_scan_universe"]=scan_u;st.session_state["pro_gap"]=int(min_gap)
        results=st.session_state.get("pro_scan",pd.DataFrame())
        if not results.empty:
            fresh=results[results["Months Since Signal"].fillna(999)<=3];near=results[(results["Status"]=="Near/No Breakout")&(results["Breakout %"]>=-near_pct)&(results["Months Gap"]>=min_gap)];a,b,c,d=st.columns(4);a.metric("Records",len(results));b.metric("Fresh / <=3M",len(fresh));c.metric(f"Near ATH <= {near_pct}%",len(near));d.metric("Rule Gap",f">={st.session_state.get('pro_gap',min_gap)}M");st.dataframe(results.sort_values(["Status","Breakout %"],ascending=[True,False]),use_container_width=True,hide_index=True)
        else: st.info("Run the scanner to generate current results.")

with main_tabs[3]:
    st.subheader("Sector rotation from the latest scan")
    results=st.session_state.get("pro_scan",pd.DataFrame());monthlies=st.session_state.get("pro_monthlies",{})
    if results.empty: st.info("Run the 26M ATH scanner first. Sector rotation will use the same evaluated universe.")
    else:
        try: bmraw=download_prices(["^NSEI"],period="2y");bm=to_monthly(extract_one(bmraw,"^NSEI",1))
        except Exception: bm=pd.DataFrame()
        sectors=sector_table(results,monthlies,bm);st.plotly_chart(rotation_chart(sectors),use_container_width=True);st.dataframe(sectors,use_container_width=True,hide_index=True,column_config={"Sector Score":st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),"Breakout Breadth %":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.1f%%")});st.caption("LEADING = positive 6M RS vs NIFTY + positive 3M momentum; IMPROVING = momentum positive while RS is still weak.")

with main_tabs[4]:
    st.subheader("Fundamental quality check")
    scan=st.session_state.get("pro_scan",pd.DataFrame());options=scan["Symbol"].dropna().astype(str).tolist() if not scan.empty else [];sym=st.selectbox("Choose scanned stock",options) if options else st.text_input("NSE Symbol",value="LT").strip().upper()
    if st.button("Run fundamental analysis",type="primary",use_container_width=True):
        ticker=sym if sym.endswith(".NS") else sym+".NS"
        with st.spinner("Fetching reported financials…"):
            try: st.session_state["pro_fund"]=fundamental_snapshot(ticker)
            except Exception as exc: st.error(f"Fundamental fetch failed: {exc}")
    fund=st.session_state.get("pro_fund")
    if fund:
        a,b,c,d=st.columns(4);a.metric("Company",fund["Company"]);b.metric("Quality Score","N/A" if pd.isna(fund["Score"]) else f"{fund['Score']:.0f}%");c.metric("Passed",f"{fund['Passed']}/{fund['Assessed']}");d.metric("Trailing PE","N/A" if pd.isna(fund["PE"]) else f"{fund['PE']:.1f}x");st.caption(f"{fund['Sector']} · {fund['Industry']} · Retrieved {fund['Retrieved']} · Source: {fund['Source']}");st.dataframe(fund["Checks"],use_container_width=True,hide_index=True);st.info("Promoter holding trend is not guessed here; verify historical promoter/shareholding from NSE/BSE corporate filings.")

st.divider();st.caption("Professional Research Lab v2 · Integrated positional + swing framework · Scores are research heuristics; they do not predict returns.")