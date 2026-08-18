import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Professional Research Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1600px}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.2);border-radius:12px;padding:10px}
.card{padding:16px;border-radius:16px;border:1px solid rgba(120,120,120,.2);min-height:150px;box-shadow:0 5px 16px rgba(0,0,0,.05)}
.good{background:rgba(34,197,94,.10);border-left:5px solid #22c55e}.mid{background:rgba(234,179,8,.11);border-left:5px solid #eab308}.bad{background:rgba(239,68,68,.09);border-left:5px solid #ef4444}
.buygate{padding:20px;border-radius:18px;background:linear-gradient(135deg,rgba(34,197,94,.15),rgba(59,130,246,.06));border:2px solid rgba(34,197,94,.45);margin:12px 0}
.waitgate{padding:20px;border-radius:18px;background:linear-gradient(135deg,rgba(239,68,68,.12),rgba(245,158,11,.06));border:2px solid rgba(239,68,68,.35);margin:12px 0}
.small{font-size:.88rem;opacity:.78}
@media(max-width:900px){.block-container{padding:1rem .7rem!important}.card{min-height:auto}}
</style>
""",unsafe_allow_html=True)

NIFTY500_URLS=["https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv","https://niftyindices.com/IndexConstituent/ind_nifty500list.csv"]
MACRO={"NIFTY 50":"^NSEI","India VIX":"^INDIAVIX","USD/INR":"INR=X","Brent Crude":"BZ=F","Dollar Index":"DX-Y.NYB","US 10Y Yield":"^TNX","Gold":"GC=F"}
SECTORS={"NIFTY Auto":"^CNXAUTO","NIFTY Bank":"^NSEBANK","NIFTY Financial Services":"NIFTY_FIN_SERVICE.NS","NIFTY FMCG":"^CNXFMCG","NIFTY IT":"^CNXIT","NIFTY Media":"^CNXMEDIA","NIFTY Metal":"^CNXMETAL","NIFTY Pharma":"^CNXPHARMA","NIFTY PSU Bank":"^CNXPSUBANK","NIFTY Realty":"^CNXREALTY","NIFTY Energy":"^CNXENERGY","NIFTY Infrastructure":"^CNXINFRA","NIFTY Commodities":"^CNXCMDT","NIFTY Consumption":"^CNXCONSUM","NIFTY Services Sector":"^CNXSERVICE"}
HINTS={"technology":"NIFTY IT","financial":"NIFTY Financial Services","bank":"NIFTY Bank","healthcare":"NIFTY Pharma","basic materials":"NIFTY Metal","consumer cyclical":"NIFTY Auto","consumer defensive":"NIFTY FMCG","real estate":"NIFTY Realty","energy":"NIFTY Energy","communication":"NIFTY Media","industrials":"NIFTY Infrastructure","utilities":"NIFTY Energy"}

def sf(x):
    try:
        x=float(x);return x if np.isfinite(x) else np.nan
    except:return np.nan

def ret(s,n):
    s=pd.to_numeric(s,errors="coerce").dropna()
    return (s.iloc[-1]/s.iloc[-n-1]-1)*100 if len(s)>n and s.iloc[-n-1]!=0 else np.nan

@st.cache_data(ttl=1200,show_spinner=False)
def dl(tickers,period="2y",interval="1d",adjust=False):
    try:return yf.download(tickers,period=period,interval=interval,group_by="ticker",auto_adjust=adjust,threads=True,progress=False,timeout=25)
    except TypeError:return yf.download(tickers,period=period,interval=interval,group_by="ticker",auto_adjust=adjust,threads=True,progress=False)

def one(raw,t,total):
    if raw is None or raw.empty:return pd.DataFrame()
    try:
        if total==1 and not isinstance(raw.columns,pd.MultiIndex):d=raw.copy()
        elif isinstance(raw.columns,pd.MultiIndex) and t in raw.columns.get_level_values(0):d=raw[t].copy()
        elif isinstance(raw.columns,pd.MultiIndex) and t in raw.columns.get_level_values(-1):d=raw.xs(t,axis=1,level=-1).copy()
        else:return pd.DataFrame()
        return d[[c for c in ["Open","High","Low","Close","Volume"] if c in d.columns]].dropna(how="all")
    except:return pd.DataFrame()

def rsi14(s):
    s=pd.to_numeric(s,errors="coerce").dropna();d=s.diff();up=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean();rs=up/dn.replace(0,np.nan);return sf((100-100/(1+rs)).iloc[-1]) if len(s)>15 else np.nan

def atr14(d):
    h=pd.to_numeric(d["High"],errors="coerce");l=pd.to_numeric(d["Low"],errors="coerce");c=pd.to_numeric(d["Close"],errors="coerce");tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);return sf(tr.rolling(14).mean().iloc[-1])

def sector_for(ticker):
    try:sec=(yf.Ticker(ticker).info.get("sector") or "").lower()
    except:sec=""
    for k,v in HINTS.items():
        if k in sec:return v
    return "NIFTY Services Sector"

@st.cache_data(ttl=1200,show_spinner=False)
def macro_snapshot():
    ts=list(MACRO.values());raw=dl(ts,"2y");rows=[]
    for name,t in MACRO.items():
        d=one(raw,t,len(ts))
        if d.empty:continue
        c=d["Close"]
        rows.append({"Indicator":name,"Latest":sf(c.iloc[-1]),"1M %":ret(c,21),"3M %":ret(c,63),"12M %":ret(c,252),"As Of":pd.to_datetime(c.dropna().index[-1]).strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)

def macro_score(df):
    if df.empty:return 5.0
    cfg={"NIFTY 50":1,"India VIX":-1,"USD/INR":-1,"Brent Crude":-1,"Dollar Index":-1,"US 10Y Yield":-1,"Gold":-1};vals=[]
    for _,r in df.iterrows():
        direction=cfg.get(r["Indicator"],1);moves=[sf(r["3M %"]),sf(r["12M %"])];moves=[x for x in moves if pd.notna(x)]
        if moves:vals.append(np.tanh(direction*np.mean(moves)/15))
    return float(np.clip(5+5*np.mean(vals),0,10)) if vals else 5.0

@st.cache_data(ttl=1200,show_spinner=False)
def weekly_sector_scores():
    ts=["^NSEI"]+list(SECTORS.values());raw=dl(ts,"2y","1wk",True);b=one(raw,"^NSEI",len(ts))
    if b.empty:return pd.DataFrame()
    bc=b["Close"];rows=[]
    for name,t in SECTORS.items():
        d=one(raw,t,len(ts))
        if d.empty:continue
        s=d["Close"];r4=ret(s,4)-ret(bc,4);r13=ret(s,13)-ret(bc,13);r26=ret(s,26)-ret(bc,26);r52=ret(s,52)-ret(bc,52);acc=r4-r13;score=np.nanmean([r13,r26,r52]);status="🟢 LEADER" if score>0 and r13>0 and r26>0 else "🔵 IMPROVING" if r4>0 and acc>0 else "🟠 WEAKENING" if score>0 else "🔴 LAGGARD";rows.append({"Sector":name,"Status":status,"4W RS":r4,"13W RS":r13,"26W RS":r26,"52W RS":r52,"Acceleration":acc,"RS Score":score})
    return pd.DataFrame(rows).sort_values(["RS Score","Acceleration"],ascending=False)

def swing(symbol,mscore):
    ticker=symbol if symbol.endswith(".NS") else symbol+".NS";sname=sector_for(ticker);sticker=SECTORS.get(sname,"^CNXSERVICE");ts=[ticker,"^NSEI",sticker];raw=dl(ts,"1y");d=one(raw,ticker,3);n=one(raw,"^NSEI",3);sec=one(raw,sticker,3)
    if d.empty or n.empty:return None
    c=pd.to_numeric(d["Close"],errors="coerce").dropna();nc=n["Close"];sc=sec["Close"] if not sec.empty else pd.Series(dtype=float);price=sf(c.iloc[-1]);ma20=sf(c.rolling(20).mean().iloc[-1]);ma50=sf(c.rolling(50).mean().iloc[-1]);ma200=sf(c.rolling(200).mean().iloc[-1]) if len(c)>=200 else np.nan
    n20=sf(nc.rolling(20).mean().iloc[-1]);n50=sf(nc.rolling(50).mean().iloc[-1]);np_=sf(nc.iloc[-1]);market=100*sum([np_>n20,np_>n50,ret(nc,20)>0])/3
    sr4=ret(sc,20)-ret(nc,20) if len(sc)>21 else np.nan;sr13=ret(sc,65)-ret(nc,65) if len(sc)>66 else np.nan;sector_score=float(np.clip(50+8*(0 if pd.isna(sr4) else sr4)+4*(0 if pd.isna(sr13) else sr13),0,100))
    rsn=ret(c,20)-ret(nc,20);rss=ret(c,20)-ret(sc,20) if len(sc)>21 else np.nan;rs_score=float(np.clip(50+8*rsn+6*(0 if pd.isna(rss) else rss),0,100))
    high20=sf(pd.to_numeric(d["High"],errors="coerce").iloc[-21:-1].max());low20=sf(pd.to_numeric(d["Low"],errors="coerce").iloc[-20:].min());vol=pd.to_numeric(d["Volume"],errors="coerce").dropna();vr=sf(vol.iloc[-1]/vol.iloc[-21:-1].mean()) if len(vol)>21 else np.nan;rsi=rsi14(c);atr=atr14(d)
    breakout=bool(pd.notna(high20) and price>high20);trend=bool(price>ma20 and price>ma50 and (pd.isna(ma200) or price>ma200) and ma20>ma50);momentum=bool(pd.notna(rsi) and 50<=rsi<=75);volume=bool(pd.notna(vr) and vr>=1.2);relative=bool(rsn>0 and (pd.isna(rss) or rss>0));sector_ok=bool(sector_score>=55);market_ok=bool(market>=50)
    # Mandatory technical BUY gate: BUY cannot appear unless ALL core technical conditions pass.
    core={"Price above 20DMA":price>ma20,"Price above 50DMA":price>ma50,"20DMA above 50DMA":ma20>ma50,"20-day breakout":breakout,"RSI 50–75":momentum,"Volume >= 1.2x 20D avg":volume,"Stock RS > NIFTY":rsn>0,"Sector supportive":sector_ok}
    passed=sum(bool(v) for v in core.values());technical_gate=all(core.values())
    setup_score=100*passed/len(core);macro_component=float(np.clip(mscore*10,0,100));weighted=market*.10+sector_score*.15+rs_score*.20+setup_score*.35+macro_component*.05+(100 if volume else 35)*.15
    # Hard gate overrides score: no BUY label without technical confirmation.
    if technical_gate and weighted>=70:label="🟢 BUY CONDITION SATISFIED"
    elif passed>=6 and weighted>=60:label="🟡 WATCH — WAIT FOR TECHNICAL TRIGGER"
    elif passed>=4:label="🟠 WEAK SETUP — WAIT"
    else:label="🔴 AVOID / WAIT"
    stop_candidates=[x for x in [ma20,low20,price-1.5*atr if pd.notna(atr) else np.nan] if pd.notna(x) and x<price];stop=max(stop_candidates) if stop_candidates else price*.96;risk=price-stop;target=price+2*risk
    checklist=pd.DataFrame([{"Technical BUY Condition":k,"Live Result":"PASS ✅" if v else "FAIL ❌"} for k,v in core.items()])
    missing=[k for k,v in core.items() if not v]
    return {"Ticker":ticker,"Price":price,"Sector":sname,"As Of":pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d"),"Final":label,"Score":weighted,"Gate":technical_gate,"Passed":passed,"Total":len(core),"Checklist":checklist,"Missing":missing,"Market":market,"MarketDetail":f"Nifty {np_:.1f} | 20DMA {n20:.1f} | 50DMA {n50:.1f}","SectorScore":sector_score,"SectorDetail":f"1M RS {sr4:.2f}% | ~3M RS {sr13:.2f}%","RSScore":rs_score,"RSDetail":f"Stock vs Nifty {rsn:.2f}% | vs Sector {rss:.2f}%" if pd.notna(rss) else f"Stock vs Nifty {rsn:.2f}%","SetupScore":setup_score,"SetupDetail":f"{passed}/{len(core)} mandatory conditions passed | 20D high ₹{high20:.2f}","VolumeScore":100 if volume else 35,"VolumeDetail":f"{vr:.2f}x vs 20D avg" if pd.notna(vr) else "N/A","MacroScore":macro_component,"RSI":rsi,"MA20":ma20,"MA50":ma50,"MA200":ma200,"High20":high20,"Entry":price,"SL":stop,"Target":target,"Trail":ma20}

macro=macro_snapshot();mscore=macro_score(macro);regime="🟢 SUPPORTIVE" if mscore>=7.5 else "🟡 MIXED / SELECTIVE" if mscore>=5 else "🟠 CAUTIOUS" if mscore>=3 else "🔴 RISK-OFF"
st.title("🧠 Professional Investment Research Lab")
a,b,c=st.columns(3);a.metric("Macro Score",f"{mscore:.1f}/10");b.metric("Macro Regime",regime);c.metric("Data Cache","20 min")

tabs=st.tabs(["🌐 Long-Term / Positional","⚡ Swing Trading","📈 26M ATH Scanner","🔄 Sector Rotation","🧾 Fundamentals"])
with tabs[0]:
    st.subheader("🌐 Macro Regime Diagnostics");st.info("Macro positional/long-term background filter है; direct BUY signal नहीं.");st.dataframe(macro,use_container_width=True,hide_index=True)
    st.markdown("**Decision:** Macro → Sector Leadership → 26M ATH → Fundamentals → Final Opportunity")
with tabs[1]:
    st.subheader("⚡ Live Swing Trading Engine — Technical BUY Gate")
    st.caption("BUY तभी आएगा जब mandatory technical setup satisfy हो. केवल high total score से BUY नहीं मिलेगा.")
    c1,c2=st.columns([2,1]);symbol=c1.text_input("NSE Symbol",value="NMDC").strip().upper();run=c2.button("▶ Run Live Swing Analysis",type="primary",use_container_width=True)
    if run and symbol:
        with st.spinner("Checking live/recent technical conditions…"):
            try:st.session_state["sw"]=swing(symbol,mscore)
            except Exception as e:st.error(f"Swing analysis failed: {e}")
    sw=st.session_state.get("sw")
    if sw:
        st.markdown(f"## {sw['Final']} — {sw['Ticker']}")
        a,b,c,d,e=st.columns(5);a.metric("Swing Score",f"{sw['Score']:.1f}/100");b.metric("Price",f"₹{sw['Price']:.2f}");c.metric("Technical Gate",f"{sw['Passed']}/{sw['Total']}");d.metric("RSI(14)",f"{sw['RSI']:.1f}");e.metric("As Of",sw['As Of'])
        cls="buygate" if sw["Gate"] else "waitgate";msg="ALL mandatory technical conditions PASS. BUY condition is unlocked." if sw["Gate"] else "BUY LOCKED. Missing: "+", ".join(sw["Missing"])
        st.markdown(f'<div class="{cls}"><b>🔐 TECHNICAL BUY GATE</b><br><span style="font-size:1.15rem">{msg}</span></div>',unsafe_allow_html=True)
        st.markdown("### ✅ Mandatory Technical BUY Checklist")
        st.dataframe(sw["Checklist"],use_container_width=True,hide_index=True)
        cards=[("📍 MARKET",sw['Market'],sw['MarketDetail']),("🥇 SECTOR",sw['SectorScore'],sw['SectorDetail']),("💪 RELATIVE STRENGTH",sw['RSScore'],sw['RSDetail']),("📈 TECHNICAL",sw['SetupScore'],sw['SetupDetail']),("📊 VOLUME",sw['VolumeScore'],sw['VolumeDetail']),("🌐 MACRO",sw['MacroScore'],f"{mscore:.1f}/10 — low swing weight")];r1=st.columns(3);r2=st.columns(3)
        for col,(title,val,detail) in zip(r1+r2,cards):
            css="good" if val>=70 else "mid" if val>=45 else "bad";col.markdown(f'<div class="card {css}"><b>{title}</b><div style="font-size:1.5rem;font-weight:800">{val:.0f}/100</div><span class="small">{detail}</span></div>',unsafe_allow_html=True)
        st.markdown("### 🎯 Trade Plan (active only after BUY gate passes)");x1,x2,x3,x4=st.columns(4);x1.metric("Entry Reference",f"₹{sw['Entry']:.2f}");x2.metric("Stop Loss",f"₹{sw['SL']:.2f}");x3.metric("Target 2R",f"₹{sw['Target']:.2f}");x4.metric("Trail 20DMA",f"₹{sw['Trail']:.2f}")
        if not sw["Gate"]:st.warning("ऊपर के Entry/SL/Target केवल reference हैं; Technical BUY Gate FAIL होने पर dashboard BUY recommend नहीं करता.")
    else:st.info("Symbol डालकर Run Live Swing Analysis दबाएँ.")
    st.subheader("🥇 Live Weekly Sector Leadership")
    try:ss=weekly_sector_scores();st.dataframe(ss.round(2),use_container_width=True,hide_index=True)
    except Exception as e:st.warning(f"Sector feed unavailable: {e}")
with tabs[2]:
    st.subheader("📈 26M ATH Scanner")
    st.info("Existing positional rule: monthly close above prior ATH after >=26 months. Full-universe scan remains in your dedicated scanner workflow.")
with tabs[3]:
    st.subheader("🔄 Sector Rotation")
    try:st.dataframe(weekly_sector_scores().round(2),use_container_width=True,hide_index=True)
    except Exception as e:st.warning(str(e))
with tabs[4]:
    st.subheader("🧾 Fundamentals")
    st.info("Use fundamentals as quality confirmation for positional trades; swing BUY is controlled by the technical gate above.")

st.divider();st.caption("Professional Research Lab v4 · BUY requires mandatory technical confirmation · Research heuristic, not investment advice.")