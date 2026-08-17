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
.live-card{padding:16px;border-radius:16px;border:1px solid rgba(120,120,120,.20);min-height:160px;box-shadow:0 5px 16px rgba(0,0,0,.05)}
.live-good{background:linear-gradient(135deg,rgba(34,197,94,.14),rgba(34,197,94,.04));border-left:5px solid #22c55e}
.live-mid{background:linear-gradient(135deg,rgba(234,179,8,.14),rgba(234,179,8,.04));border-left:5px solid #eab308}
.live-bad{background:linear-gradient(135deg,rgba(239,68,68,.13),rgba(239,68,68,.04));border-left:5px solid #ef4444}
.small-note{font-size:.88rem;opacity:.78}
@media(max-width:900px){.block-container{padding:1rem .7rem 1.5rem!important}.hero-card{padding:16px}.regime-box,.support-box,.negative-box,.confirm-box,.live-card{min-height:auto}.stDataFrame{overflow-x:auto}h1{font-size:1.7rem!important}h2{font-size:1.35rem!important}}
</style>
""", unsafe_allow_html=True)

NIFTY500_URLS=["https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv","https://niftyindices.com/IndexConstituent/ind_nifty500list.csv"]
MACRO_ASSETS={"NIFTY 50":"^NSEI","India VIX":"^INDIAVIX","USD/INR":"INR=X","Brent Crude":"BZ=F","Dollar Index":"DX-Y.NYB","US 10Y Yield":"^TNX","Gold":"GC=F"}
SECTORS={"NIFTY Auto":"^CNXAUTO","NIFTY Bank":"^NSEBANK","NIFTY Financial Services":"NIFTY_FIN_SERVICE.NS","NIFTY FMCG":"^CNXFMCG","NIFTY IT":"^CNXIT","NIFTY Media":"^CNXMEDIA","NIFTY Metal":"^CNXMETAL","NIFTY Pharma":"^CNXPHARMA","NIFTY PSU Bank":"^CNXPSUBANK","NIFTY Realty":"^CNXREALTY","NIFTY Energy":"^CNXENERGY","NIFTY Infrastructure":"^CNXINFRA","NIFTY Commodities":"^CNXCMDT","NIFTY Consumption":"^CNXCONSUM","NIFTY Services Sector":"^CNXSERVICE"}
SECTOR_HINTS={"technology":"NIFTY IT","financial services":"NIFTY Financial Services","financial":"NIFTY Financial Services","bank":"NIFTY Bank","healthcare":"NIFTY Pharma","basic materials":"NIFTY Metal","consumer cyclical":"NIFTY Auto","consumer defensive":"NIFTY FMCG","real estate":"NIFTY Realty","energy":"NIFTY Energy","communication services":"NIFTY Media","industrials":"NIFTY Infrastructure","utilities":"NIFTY Energy"}

def safe_float(v):
    try:
        v=float(v);return v if np.isfinite(v) else np.nan
    except Exception:return np.nan

def month_diff(later,earlier):return (later.year-earlier.year)*12+(later.month-earlier.month)

def period_return(close,periods):
    c=pd.to_numeric(close,errors="coerce").dropna()
    if len(c)<=periods:return np.nan
    base,last=safe_float(c.iloc[-periods-1]),safe_float(c.iloc[-1]);return (last/base-1)*100 if pd.notna(base) and base!=0 else np.nan

def extract_one(raw,ticker,total):
    try:
        if raw.empty:return pd.DataFrame()
        if total==1 and not isinstance(raw.columns,pd.MultiIndex):d=raw.copy()
        elif isinstance(raw.columns,pd.MultiIndex) and ticker in raw.columns.get_level_values(0):d=raw[ticker].copy()
        elif isinstance(raw.columns,pd.MultiIndex) and ticker in raw.columns.get_level_values(-1):d=raw.xs(ticker,axis=1,level=-1).copy()
        else:return pd.DataFrame()
        cols=[c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in d.columns];return d[cols].dropna(how="all")
    except Exception:return pd.DataFrame()

@st.cache_data(ttl=86400,show_spinner=False)
def load_nifty500():
    headers={"User-Agent":"Mozilla/5.0"};errors=[]
    for url in NIFTY500_URLS:
        try:
            r=requests.get(url,timeout=15,headers=headers);r.raise_for_status();df=pd.read_csv(io.StringIO(r.text))
            if "Symbol" not in df.columns:continue
            if "Industry" not in df.columns:df["Industry"]="Unknown"
            if "Company Name" not in df.columns:df["Company Name"]=df["Symbol"]
            df["Ticker"]=df["Symbol"].astype(str).str.strip()+".NS";return df[["Company Name","Industry","Symbol","Ticker"]].drop_duplicates("Ticker")
        except Exception as exc:errors.append(str(exc))
    raise RuntimeError("NIFTY 500 list unavailable: "+" | ".join(errors[-2:]))

@st.cache_data(ttl=1200,show_spinner=False)
def download_prices(tickers,period="2y",interval="1d",auto_adjust=False):
    if not tickers:return pd.DataFrame()
    kwargs=dict(tickers=tickers,period=period,interval=interval,group_by="ticker",auto_adjust=auto_adjust,threads=True,progress=False)
    try:return yf.download(timeout=25,**kwargs)
    except TypeError:return yf.download(**kwargs)

def to_monthly(d):
    if d is None or d.empty or "Close" not in d.columns or "High" not in d.columns:return pd.DataFrame()
    x=d.copy();x.index=pd.to_datetime(x.index).tz_localize(None);agg={"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"};return x.resample("ME").agg({k:v for k,v in agg.items() if k in x.columns}).dropna(subset=["High","Close"])

def strict_26m_signal(monthly,min_gap=26):
    if monthly is None or len(monthly)<min_gap+2:return None
    m=monthly.dropna(subset=["High","Close"]).copy();highs=pd.to_numeric(m["High"],errors="coerce");closes=pd.to_numeric(m["Close"],errors="coerce");signals=[]
    for i in range(1,len(m)):
        prior=highs.iloc[:i].dropna()
        if prior.empty:continue
        old_ath=float(prior.max());dates=prior.index[np.isclose(prior.values,old_ath,rtol=1e-10,atol=1e-10)];old_date=dates[-1] if len(dates) else prior.idxmax();gap=month_diff(m.index[i],old_date);close=safe_float(closes.iloc[i])
        if pd.notna(close) and close>old_ath and gap>=min_gap:
            avg12=safe_float(m["Volume"].iloc[max(0,i-12):i].mean()) if "Volume" in m.columns else np.nan;vol=safe_float(m["Volume"].iloc[i]) if "Volume" in m.columns else np.nan;signals.append({"Signal Date":m.index[i],"Old ATH":old_ath,"ATH Date":old_date,"Months Gap":gap,"Monthly Close":close,"Breakout %":(close/old_ath-1)*100,"Volume Ratio":vol/avg12 if pd.notna(vol) and pd.notna(avg12) and avg12>0 else np.nan})
    if signals:
        sig=signals[-1];age=month_diff(m.index[-1],sig["Signal Date"]);sig["Months Since Signal"]=age;sig["Status"]="Fresh Breakout" if age==0 else "Breakout <=3M" if age<=3 else "Older Breakout";return sig
    prior=highs.iloc[:-1].dropna()
    if prior.empty:return None
    old_ath=float(prior.max());dates=prior.index[np.isclose(prior.values,old_ath,rtol=1e-10,atol=1e-10)];old_date=dates[-1] if len(dates) else prior.idxmax();close=safe_float(closes.iloc[-1]);return {"Status":"Near/No Breakout","Signal Date":pd.NaT,"Old ATH":old_ath,"ATH Date":old_date,"Months Gap":month_diff(m.index[-1],old_date),"Monthly Close":close,"Breakout %":(close/old_ath-1)*100 if pd.notna(close) else np.nan,"Volume Ratio":np.nan,"Months Since Signal":np.nan}

@st.cache_data(ttl=1200,show_spinner=False)
def macro_snapshot():
    tickers=list(MACRO_ASSETS.values())
    try:raw=download_prices(tickers,period="2y")
    except Exception:raw=pd.DataFrame()
    rows=[]
    for name,ticker in MACRO_ASSETS.items():
        d=extract_one(raw,ticker,len(tickers))
        if d.empty or "Close" not in d.columns:continue
        c=pd.to_numeric(d["Close"],errors="coerce").dropna()
        if len(c)<25:continue
        rows.append({"Indicator":name,"Ticker":ticker,"Latest":safe_float(c.iloc[-1]),"1M %":period_return(c,21),"3M %":period_return(c,63),"12M %":period_return(c,252),"As Of":pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)

def long_term_macro_engine(mdf):
    specs=[("NIFTY 50",+1,2.2,10.0,"Domestic equity trend"),("India VIX",-1,1.6,18.0,"Volatility / fear"),("USD/INR",-1,1.5,6.0,"Rupee / imported inflation pressure"),("Brent Crude",-1,1.5,18.0,"India import-cost pressure"),("Dollar Index",-1,1.0,8.0,"Global dollar liquidity"),("US 10Y Yield",-1,1.0,10.0,"Global cost of money"),("Gold",-1,0.6,18.0,"Defensive / uncertainty proxy")]
    if mdf is None or mdf.empty:return 5.0,"🟡 MIXED / SELECTIVE",0,pd.DataFrame(),"Insufficient live inputs"
    total=sum(x[2] for x in specs);weighted=0.0;used=0.0;rows=[]
    for name,direction,weight,scale,meaning in specs:
        r=mdf[mdf["Indicator"]==name]
        if r.empty:continue
        r1,r3,r12=[safe_float(r.iloc[0][c]) for c in ["1M %","3M %","12M %"]];parts=[]
        if pd.notna(r1):parts.append((r1,.15))
        if pd.notna(r3):parts.append((r3,.40))
        if pd.notna(r12):parts.append((r12,.45))
        if not parts:continue
        move=sum(v*w for v,w in parts)/sum(w for _,w in parts);raw=float(np.tanh(direction*move/scale));weighted+=raw*weight;used+=weight;signal="🟢 Supportive" if raw>.15 else "🔴 Negative" if raw<-.15 else "🟡 Neutral/Mixed";rows.append({"Driver":name,"Meaning":meaning,"1M %":r1,"3M %":r3,"12M %":r12,"Contribution":raw*weight,"Signal":signal})
    if used==0:return 5.0,"🟡 MIXED / SELECTIVE",0,pd.DataFrame(rows),"No evaluable inputs"
    score=float(np.clip(5+5*weighted/used,0,10));coverage=int(round(100*used/total));regime="🟢 SUPPORTIVE / RISK-ON" if score>=7.5 else "🟡 MIXED / SELECTIVE" if score>=5 else "🟠 CAUTIOUS" if score>=3 else "🔴 RISK-OFF";return score,regime,coverage,pd.DataFrame(rows).sort_values("Contribution",ascending=False),f"{coverage}% weighted input coverage"

def regime_strategy(score):
    if score>=7.5:return "🟢","SUPPORTIVE / RISK-ON","More opportunities can be explored; still require sector leadership, quality and valid breakout structure.","hero-green"
    if score>=5:return "🟡","MIXED / SELECTIVE","Only sector leaders + strong fundamentals + confirmed breakouts. Avoid broad-market buying.","hero-yellow"
    if score>=3:return "🟠","CAUTIOUS","Reduce position size, demand stronger confirmation and stay highly selective.","hero-orange"
    return "🔴","RISK-OFF","Avoid aggressive fresh buying; prioritize capital protection and only exceptional setups.","hero-red"

def auto_sector_for_stock(ticker):
    try:info=yf.Ticker(ticker).info or {};sec=(info.get("sector") or "").strip().lower()
    except Exception:sec=""
    for hint,sector in SECTOR_HINTS.items():
        if hint in sec:return sector,sec.title() if sec else "N/A"
    return "NIFTY Services Sector",sec.title() if sec else "N/A"

def rsi14(close):
    c=pd.to_numeric(close,errors="coerce").dropna()
    if len(c)<16:return np.nan
    delta=c.diff();up=delta.clip(lower=0).rolling(14).mean();down=(-delta.clip(upper=0)).rolling(14).mean();rs=up/down.replace(0,np.nan);return safe_float((100-(100/(1+rs))).iloc[-1])

def atr14(d):
    if d is None or d.empty or not all(x in d.columns for x in ["High","Low","Close"]):return np.nan
    h=pd.to_numeric(d["High"],errors="coerce");l=pd.to_numeric(d["Low"],errors="coerce");c=pd.to_numeric(d["Close"],errors="coerce");tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);return safe_float(tr.rolling(14).mean().iloc[-1])

def live_swing_analysis(symbol,macro_score):
    ticker=symbol if symbol.endswith(".NS") else symbol+".NS";sector_name,yahoo_sector=auto_sector_for_stock(ticker);sector_ticker=SECTORS.get(sector_name,"^CNXSERVICE");tickers=[ticker,"^NSEI",sector_ticker];raw=download_prices(tickers,period="1y",interval="1d",auto_adjust=False);stock=extract_one(raw,ticker,len(tickers));nifty=extract_one(raw,"^NSEI",len(tickers));sector=extract_one(raw,sector_ticker,len(tickers))
    if stock.empty or nifty.empty or "Close" not in stock.columns:return None
    sc=pd.to_numeric(stock["Close"],errors="coerce").dropna();nc=pd.to_numeric(nifty["Close"],errors="coerce").dropna();sec=pd.to_numeric(sector["Close"],errors="coerce").dropna() if not sector.empty else pd.Series(dtype=float);last=safe_float(sc.iloc[-1]);dma20=safe_float(sc.rolling(20).mean().iloc[-1]);dma50=safe_float(sc.rolling(50).mean().iloc[-1]);dma200=safe_float(sc.rolling(200).mean().iloc[-1]) if len(sc)>=200 else np.nan;n20=safe_float(nc.rolling(20).mean().iloc[-1]);n50=safe_float(nc.rolling(50).mean().iloc[-1]);nlast=safe_float(nc.iloc[-1]);market_points=sum([int(pd.notna(n20) and nlast>n20),int(pd.notna(n50) and nlast>n50),int(period_return(nc,20)>0)]);market_s=100*market_points/3;sector_r4=period_return(sec,20)-period_return(nc,20) if not sec.empty else np.nan;sector_r13=period_return(sec,65)-period_return(nc,65) if not sec.empty else np.nan;sector_s=float(np.clip(50+8*(0 if pd.isna(sector_r4) else sector_r4)+4*(0 if pd.isna(sector_r13) else sector_r13),0,100));rs_nifty=period_return(sc,20)-period_return(nc,20);rs_sector=period_return(sc,20)-period_return(sec,20) if not sec.empty else np.nan;rs_s=float(np.clip(50+8*rs_nifty+6*(0 if pd.isna(rs_sector) else rs_sector),0,100));trend_points=sum([int(pd.notna(dma20) and last>dma20),int(pd.notna(dma50) and last>dma50),int(pd.isna(dma200) or last>dma200),int(pd.notna(dma20) and pd.notna(dma50) and dma20>dma50)]);trend_s=100*trend_points/4;high20=safe_float(pd.to_numeric(stock["High"],errors="coerce").iloc[-21:-1].max()) if len(stock)>21 else np.nan;low20=safe_float(pd.to_numeric(stock["Low"],errors="coerce").iloc[-20:].min()) if len(stock)>=20 else np.nan;breakout_pct=(last/high20-1)*100 if pd.notna(high20) and high20 else np.nan;pullback_pct=(last/dma20-1)*100 if pd.notna(dma20) and dma20 else np.nan;breakout_ok=pd.notna(high20) and last>high20;pullback_ok=pd.notna(pullback_pct) and -2.5<=pullback_pct<=2.5 and pd.notna(dma50) and last>dma50;setup_s=95 if breakout_ok else 80 if pullback_ok else 55 if pd.notna(dma20) and last>dma20 else 30;vol=pd.to_numeric(stock["Volume"],errors="coerce").dropna() if "Volume" in stock.columns else pd.Series(dtype=float);vol_ratio=safe_float(vol.iloc[-1]/vol.iloc[-21:-1].mean()) if len(vol)>21 and vol.iloc[-21:-1].mean()>0 else np.nan;volume_s=float(np.clip(50+35*((0 if pd.isna(vol_ratio) else vol_ratio)-1),0,100));atr=atr14(stock);rsi=rsi14(sc);structure_stop=min(x for x in [dma20,low20] if pd.notna(x)) if any(pd.notna(x) for x in [dma20,low20]) else last-(2*atr if pd.notna(atr) else last*.04);atr_stop=last-1.5*atr if pd.notna(atr) else structure_stop;stop=max(structure_stop,atr_stop) if pd.notna(structure_stop) and pd.notna(atr_stop) else structure_stop
    if stop>=last:stop=last-(1.5*atr if pd.notna(atr) else last*.03)
    risk=last-stop if pd.notna(stop) else np.nan;target=last+2*risk if pd.notna(risk) and risk>0 else np.nan;trail=dma20 if pd.notna(dma20) else stop;rr=2.0 if pd.notna(target) and pd.notna(risk) and risk>0 else np.nan;rr_s=100 if pd.notna(rr) and rr>=2 else 50;macro_s=float(np.clip(macro_score*10,0,100));weighted=market_s*.10+sector_s*.20+rs_s*.20+trend_s*.15+setup_s*.20+volume_s*.10+macro_s*.05;final_label="🟢 HIGH-QUALITY SWING CANDIDATE" if weighted>=75 else "🟡 WATCH / SELECTIVE" if weighted>=60 else "🟠 WEAK SETUP" if weighted>=45 else "🔴 AVOID / WAIT"
    return {"Ticker":ticker,"Sector":sector_name,"Price":last,"As Of":pd.to_datetime(sc.index[-1]).strftime("%Y-%m-%d"),"Market Score":market_s,"Market Detail":f"Nifty {nlast:.1f} | 20DMA {n20:.1f} | 50DMA {n50:.1f}","Sector Score":sector_s,"Sector Detail":f"{sector_name}: 1M RS {sector_r4:.2f}% | ~3M RS {sector_r13:.2f}%","RS Score":rs_s,"RS Detail":f"Stock vs Nifty 1M {rs_nifty:.2f}% | vs Sector {rs_sector:.2f}%" if pd.notna(rs_sector) else f"Stock vs Nifty 1M {rs_nifty:.2f}%","Trend Score":trend_s,"Trend Detail":f"Price {last:.2f} | 20DMA {dma20:.2f} | 50DMA {dma50:.2f}"+(f" | 200DMA {dma200:.2f}" if pd.notna(dma200) else ""),"Setup Score":setup_s,"Setup Detail":f"{'Breakout' if breakout_ok else 'Pullback near 20DMA' if pullback_ok else 'No clean trigger'} | 20D high {high20:.2f} | Distance {breakout_pct:.2f}%","Volume Score":volume_s,"Volume Detail":f"Volume ratio vs 20D avg: {vol_ratio:.2f}x" if pd.notna(vol_ratio) else "Volume data N/A","Macro Score":macro_s,"Macro Detail":f"Macro regime score {macro_score:.1f}/10 (low weight in swing)","RR Score":rr_s,"RR Detail":f"Entry {last:.2f} | SL {stop:.2f} | Target {target:.2f} | R:R ~{rr:.1f}:1" if pd.notna(target) else "R:R unavailable","Entry":last,"Stop Loss":stop,"Target 2R":target,"Trail":trail,"RSI14":rsi,"Final Score":weighted,"Final Label":final_label}

def weekly_sector_scores():
    tickers=["^NSEI"]+list(SECTORS.values());raw=download_prices(tickers,period="2y",interval="1wk",auto_adjust=True);closes={}
    for t in tickers:
        d=extract_one(raw,t,len(tickers))
        if not d.empty and "Close" in d.columns:closes[t]=pd.to_numeric(d["Close"],errors="coerce").dropna()
    b=closes.get("^NSEI",pd.Series(dtype=float))
    if b.empty:return pd.DataFrame()
    rows=[]
    for name,ticker in SECTORS.items():
        s=closes.get(ticker,pd.Series(dtype=float))
        if s.empty:continue
        r4=period_return(s,4)-period_return(b,4);r13=period_return(s,13)-period_return(b,13);r26=period_return(s,26)-period_return(b,26);r52=period_return(s,52)-period_return(b,52);accel=r4-r13;rs=np.nanmean([r13,r26,r52]);status="🟢 LEADER" if pd.notna(rs) and rs>0 and r13>0 and r26>0 else "🔵 IMPROVING" if pd.notna(r4) and r4>0 and pd.notna(accel) and accel>0 else "🟠 WEAKENING" if pd.notna(rs) and rs>0 else "🔴 LAGGARD";rows.append({"Sector":name,"Status":status,"4W RS":r4,"13W RS":r13,"26W RS":r26,"52W RS":r52,"Acceleration":accel,"RS Score":rs})
    return pd.DataFrame(rows).sort_values(["RS Score","Acceleration"],ascending=False)

def sector_table(results,monthlies,benchmark_monthly):
    if results is None or results.empty:return pd.DataFrame()
    bm6=period_return(benchmark_monthly["Close"],6) if not benchmark_monthly.empty else np.nan;rows=[]
    for industry,group in results.groupby("Industry"):
        r3s=[];r6s=[];r12s=[]
        for sym in group["Symbol"].head(40):
            m=monthlies.get(sym+".NS",pd.DataFrame())
            if m.empty:continue
            r3s.append(period_return(m["Close"],3));r6s.append(period_return(m["Close"],6));r12s.append(period_return(m["Close"],12))
        clean=lambda xs:[x for x in xs if pd.notna(x)];r3s,r6s,r12s=clean(r3s),clean(r6s),clean(r12s);m3=float(np.median(r3s)) if r3s else np.nan;m6=float(np.median(r6s)) if r6s else np.nan;m12=float(np.median(r12s)) if r12s else np.nan;rs6=m6-bm6 if pd.notna(m6) and pd.notna(bm6) else np.nan;recent=int((group["Months Since Signal"].fillna(999)<=3).sum());near=int(((group["Status"]=="Near/No Breakout")&(group["Breakout %"]>=-5)).sum());breadth=100*(recent+.5*near)/max(1,len(group));comps=[(np.tanh(v/s),w) for v,s,w in [(m3,8,.25),(m6,12,.30),(m12,22,.20),(rs6,10,.25)] if pd.notna(v)];mom=sum(v*w for v,w in comps)/sum(w for _,w in comps) if comps else 0;s=float(np.clip(5+3.4*mom+min(1.8,breadth/30),0,10));quadrant="LEADING" if pd.notna(rs6) and rs6>=0 and pd.notna(m3) and m3>=0 else "IMPROVING" if pd.notna(m3) and m3>=0 else "WEAKENING" if pd.notna(rs6) and rs6>=0 else "LAGGING";rows.append({"Industry":industry,"3M Momentum %":m3,"6M Momentum %":m6,"12M Momentum %":m12,"6M RS vs NIFTY %":rs6,"Breakout Breadth %":breadth,"Sector Score":s,"Rotation Quadrant":quadrant,"Stocks Evaluated":len(group)})
    return pd.DataFrame(rows).sort_values("Sector Score",ascending=False)

def rotation_chart(df):
    fig=go.Figure()
    if df is None or df.empty:return fig
    d=df.dropna(subset=["6M RS vs NIFTY %","3M Momentum %"]).copy();colors={"LEADING":"#22c55e","IMPROVING":"#3b82f6","WEAKENING":"#f59e0b","LAGGING":"#ef4444"}
    for q in ["LEADING","IMPROVING","WEAKENING","LAGGING"]:
        x=d[d["Rotation Quadrant"]==q]
        if x.empty:continue
        fig.add_trace(go.Scatter(x=x["6M RS vs NIFTY %"],y=x["3M Momentum %"],mode="markers+text",text=x["Industry"],textposition="top center",name=q,marker={"size":14+4*x["Sector Score"].clip(0,10),"opacity":.75,"color":colors[q]}))
    fig.add_vline(x=0,line_dash="dash");fig.add_hline(y=0,line_dash="dash");fig.update_layout(height=560,xaxis_title="6M Relative Strength vs NIFTY (%)",yaxis_title="3M Momentum (%)",margin=dict(l=10,r=10,t=35,b=10));return fig

@st.cache_data(ttl=3600,show_spinner=False)
def fundamental_snapshot(ticker):
    stock=yf.Ticker(ticker)
    try:info=stock.info or {}
    except Exception:info={}
    try:inc=stock.financials.copy()
    except Exception:inc=pd.DataFrame()
    try:bs=stock.balance_sheet.copy()
    except Exception:bs=pd.DataFrame()
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
    sales_cagr,sales_years=cagr(revenue);profit_cagr,profit_years=cagr(profit);eq=safe_float(equity.iloc[-1]) if len(equity) else np.nan;total_debt=safe_float(info.get("totalDebt"));total_debt=safe_float(debt.iloc[-1]) if pd.isna(total_debt) and len(debt) else total_debt;de=total_debt/eq if pd.notna(total_debt) and pd.notna(eq) and eq!=0 else np.nan;latest_ebit=safe_float(ebit.iloc[-1]) if len(ebit) else np.nan;common=assets.index.intersection(cl.index);cap=safe_float(assets.loc[common[-1]]-cl.loc[common[-1]]) if len(common) else np.nan;roce=latest_ebit/cap*100 if pd.notna(latest_ebit) and pd.notna(cap) and cap!=0 else np.nan;p=profit.dropna().sort_index();increasing=bool(len(p)>=3 and all(p.iloc[i]>=p.iloc[i-1] for i in range(1,len(p)))) if len(p)>=3 else None;pe=safe_float(info.get("trailingPE"));checks=[("Debt / Equity < 0.5",de,pd.notna(de),lambda x:x<.5),("ROCE > 15%",roce,pd.notna(roce),lambda x:x>15),("Net profit increasing",increasing,increasing is not None,lambda x:x is True),("Sales CAGR > 20% for >=5Y",sales_cagr,pd.notna(sales_cagr) and sales_years>=5,lambda x:x>20),("Profit CAGR > 15% for >=5Y",profit_cagr,pd.notna(profit_cagr) and profit_years>=5,lambda x:x>15)];rows=[];passed=0;assessed=0
    for name,value,available,rule in checks:
        ok=bool(rule(value)) if available else None
        if ok is not None:assessed+=1;passed+=int(ok)
        rows.append({"Criterion":name,"Value":"N/A" if not available else ("Yes" if value is True else "No" if value is False else f"{value:.2f}"),"Result":"N/A" if ok is None else "PASS" if ok else "FAIL"})
    return {"Company":info.get("longName") or ticker,"Sector":info.get("sector") or "N/A","Industry":info.get("industry") or "N/A","PE":pe,"Score":100*passed/assessed if assessed else np.nan,"Passed":passed,"Assessed":assessed,"Checks":pd.DataFrame(rows),"Source":"Yahoo Finance quote + reported financial statements","Retrieved":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

st.title("🧠 Professional Investment Research Lab");st.caption("Integrated positional macro + LIVE swing engine + strict 26M ATH + sector rotation + fundamentals. Research tool, not investment advice.")
macro=macro_snapshot();score,regime,coverage,drivers,regime_note=long_term_macro_engine(macro);icon,regime_name,strategy,hero_class=regime_strategy(score);m1,m2,m3,m4=st.columns(4);m1.metric("Macro Score",f"{score:.1f}/10");m2.metric("Macro Regime",regime_name);m3.metric("Macro Coverage",f"{coverage}%");m4.metric("Live Data TTL","20 min")
with st.expander("🔎 Data provenance & freshness",expanded=False):
    st.markdown("**Universe:** official NIFTY Indices constituent CSV. **Market prices / macro / fundamentals:** Yahoo Finance via `yfinance`. Research feeds, not an exchange-certified terminal.")
    if not macro.empty:st.caption(f"Macro observation dates: {macro['As Of'].min()} to {macro['As Of'].max()}. {regime_note}")
    st.warning("Before order execution, verify price and filings on NSE/BSE/company filings or your broker terminal.")
main_tabs=st.tabs(["🌐 Long-Term / Positional","⚡ Swing Trading","📈 26M ATH Scanner","🔄 Sector Rotation","🧾 Fundamentals"])
with main_tabs[0]:
    st.markdown(f'<div class="hero-card {hero_class}"><b>CURRENT MACRO REGIME</b><div style="font-size:2rem;font-weight:800">{icon} {regime_name}</div><b>Macro Score: {score:.1f}/10 | Coverage: {coverage}%</b><br><br><b>Recommended Strategy:</b> {strategy}</div>',unsafe_allow_html=True);c1,c2,c3,c4=st.columns(4);c1.markdown('<div class="regime-box green-box"><b>🟢 7.5–10 SUPPORTIVE</b><br>More opportunities.</div>',unsafe_allow_html=True);c2.markdown('<div class="regime-box yellow-box"><b>🟡 5–7.49 MIXED</b><br>Leaders + quality only.</div>',unsafe_allow_html=True);c3.markdown('<div class="regime-box orange-box"><b>🟠 3–4.99 CAUTIOUS</b><br>Smaller size.</div>',unsafe_allow_html=True);c4.markdown('<div class="regime-box red-box"><b>🔴 0–2.99 RISK-OFF</b><br>Avoid aggressive fresh buying.</div>',unsafe_allow_html=True);st.markdown("### 📊 Macro Diagnostics");st.dataframe(macro,use_container_width=True,hide_index=True);st.dataframe(drivers,use_container_width=True,hide_index=True)
with main_tabs[1]:
    st.markdown("### ⚡ Live Swing Trading Engine");st.caption("हर roadmap point पर live result: Market → Sector → Strong Stock → Technical → Entry/SL → Target/Trail.");c1,c2=st.columns([2,1]);symbol=c1.text_input("NSE Symbol",value="LT",help="LT, RELIANCE, ICICIBANK, TATAMOTORS").strip().upper();run=c2.button("▶ Run Live Swing Analysis",type="primary",use_container_width=True)
    if run and symbol:
        with st.spinner("Fetching market data and calculating setup…"):
            try:st.session_state["live_swing"]=live_swing_analysis(symbol,score)
            except Exception as exc:st.error(f"Swing analysis failed: {exc}")
    sw=st.session_state.get("live_swing")
    if sw:
        st.markdown(f"### {sw['Final Label']} — **{sw['Ticker']}**");a,b,c,d,e=st.columns(5);a.metric("Swing Score",f"{sw['Final Score']:.1f}/100");b.metric("Price",f"₹{sw['Price']:.2f}");c.metric("Sector",sw['Sector']);d.metric("RSI(14)","N/A" if pd.isna(sw['RSI14']) else f"{sw['RSI14']:.1f}");e.metric("As Of",sw['As Of']);cards=[("📍 MARKET CONDITION",sw['Market Score'],sw['Market Detail']),("🥇 SECTOR LEADER",sw['Sector Score'],sw['Sector Detail']),("💪 STRONG STOCK",sw['RS Score'],sw['RS Detail']),("📈 TECHNICAL SETUP",sw['Setup Score'],sw['Setup Detail']),("📊 VOLUME",sw['Volume Score'],sw['Volume Detail']),("🌐 MACRO",sw['Macro Score'],sw['Macro Detail'])];cols=st.columns(3)+st.columns(3)
        for col,(title,val,detail) in zip(cols,cards):css="live-good" if val>=70 else "live-mid" if val>=45 else "live-bad";col.markdown(f'<div class="live-card {css}"><b>{title}</b><div style="font-size:1.55rem;font-weight:800">{val:.0f}/100</div><span class="small-note">{detail}</span></div>',unsafe_allow_html=True)
        st.markdown("### 🛑 Entry, Stop Loss, Target & Trail");x1,x2,x3,x4=st.columns(4);x1.metric("Entry Reference",f"₹{sw['Entry']:.2f}");x2.metric("Stop Loss",f"₹{sw['Stop Loss']:.2f}");x3.metric("Target (2R)",f"₹{sw['Target 2R']:.2f}");x4.metric("Trail Reference",f"₹{sw['Trail']:.2f}");tbl=pd.DataFrame([("Weekly Sector Leadership",20,sw['Sector Score'],sw['Sector Detail']),("Stock Relative Strength",20,sw['RS Score'],sw['RS Detail']),("Daily Trend",15,sw['Trend Score'],sw['Trend Detail']),("Breakout / Pullback",20,sw['Setup Score'],sw['Setup Detail']),("Volume",10,sw['Volume Score'],sw['Volume Detail']),("Macro",5,sw['Macro Score'],sw['Macro Detail']),("Risk : Reward",10,sw['RR Score'],sw['RR Detail'])],columns=["Component","Weight %","Live Score /100","Live Result"]);st.dataframe(tbl,use_container_width=True,hide_index=True,column_config={"Weight %":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%d%%"),"Live Score /100":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f")});st.warning("Entry/SL/Target are model-generated technical reference levels; verify on broker/TradingView before trade.")
    else:st.info("Symbol डालकर Run Live Swing Analysis दबाएँ.")
    st.markdown("### 🥇 Live Weekly Sector Leadership")
    try:ss=weekly_sector_scores()
    except Exception:ss=pd.DataFrame()
    if not ss.empty:st.dataframe(ss.round(2),use_container_width=True,hide_index=True)
with main_tabs[2]:
    st.subheader("Strict monthly 26M ATH scanner")
    try:universe=load_nifty500()
    except Exception as exc:st.error(str(exc));universe=pd.DataFrame()
    if not universe.empty:
        c1,c2,c3=st.columns(3);min_gap=c1.number_input("Minimum ATH gap",12,120,26,1);batch=c2.selectbox("Universe size",[50,100,200,500],index=1);near_pct=c3.slider("Near ATH range %",1,10,5,1);industries=st.multiselect("Industry filter",sorted(universe["Industry"].dropna().unique()));scan_u=universe[universe["Industry"].isin(industries)] if industries else universe;scan_u=scan_u.head(int(batch)).copy()
        if st.button(f"Run professional scan on {len(scan_u)} stocks",type="primary",use_container_width=True):
            tickers=scan_u["Ticker"].tolist();raw=download_prices(tickers,period="max");records=[];monthlies={};prog=st.progress(0)
            for i,(_,r) in enumerate(scan_u.iterrows()):
                d=extract_one(raw,r["Ticker"],len(tickers));m=to_monthly(d);monthlies[r["Ticker"]]=m;sig=strict_26m_signal(m,int(min_gap))
                if sig:records.append({"Symbol":r["Symbol"],"Company":r["Company Name"],"Industry":r["Industry"],**sig})
                prog.progress((i+1)/max(1,len(scan_u)))
            prog.empty();st.session_state["pro_scan"]=pd.DataFrame(records);st.session_state["pro_monthlies"]=monthlies
        results=st.session_state.get("pro_scan",pd.DataFrame())
        if not results.empty:st.dataframe(results,use_container_width=True,hide_index=True)
with main_tabs[3]:
    results=st.session_state.get("pro_scan",pd.DataFrame());monthlies=st.session_state.get("pro_monthlies",{})
    if results.empty:st.info("Run 26M ATH scanner first.")
    else:
        bm=to_monthly(extract_one(download_prices(["^NSEI"],period="2y"),"^NSEI",1));sectors=sector_table(results,monthlies,bm);st.plotly_chart(rotation_chart(sectors),use_container_width=True);st.dataframe(sectors,use_container_width=True,hide_index=True)
with main_tabs[4]:
    sym=st.text_input("NSE Symbol",value="LT",key="fund_symbol").strip().upper()
    if st.button("Run fundamental analysis",type="primary",use_container_width=True):
        try:st.session_state["pro_fund"]=fundamental_snapshot(sym+".NS" if not sym.endswith(".NS") else sym)
        except Exception as exc:st.error(str(exc))
    fund=st.session_state.get("pro_fund")
    if fund:st.metric("Company",fund["Company"]);st.metric("Quality Score","N/A" if pd.isna(fund["Score"]) else f"{fund['Score']:.0f}%");st.dataframe(fund["Checks"],use_container_width=True,hide_index=True)
st.divider();st.caption("Professional Research Lab v3 · Integrated positional + LIVE swing engine · Research heuristics, not return predictions.")