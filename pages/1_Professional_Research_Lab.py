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
.confirmed-banner{padding:18px 20px;border-radius:16px;background:linear-gradient(135deg,rgba(34,197,94,.20),rgba(16,185,129,.08));border:2px solid #22c55e;margin:10px 0 14px 0;font-size:1.05rem}
.confirmed-pill{display:inline-block;padding:5px 10px;margin:4px 6px 4px 0;border-radius:999px;background:#16a34a;color:white;font-weight:800;font-size:.9rem}
.small{font-size:.88rem;opacity:.78}
@media(max-width:900px){.block-container{padding:1rem .7rem!important}.card{min-height:auto}}
</style>
""",unsafe_allow_html=True)

MACRO={"NIFTY 50":"^NSEI","India VIX":"^INDIAVIX","USD/INR":"INR=X","Brent Crude":"BZ=F","Dollar Index":"DX-Y.NYB","US 10Y Yield":"^TNX","Gold":"GC=F"}
SECTORS={"NIFTY Auto":"^CNXAUTO","NIFTY Bank":"^NSEBANK","NIFTY Financial Services":"NIFTY_FIN_SERVICE.NS","NIFTY FMCG":"^CNXFMCG","NIFTY IT":"^CNXIT","NIFTY Media":"^CNXMEDIA","NIFTY Metal":"^CNXMETAL","NIFTY Pharma":"^CNXPHARMA","NIFTY PSU Bank":"^CNXPSUBANK","NIFTY Realty":"^CNXREALTY","NIFTY Energy":"^CNXENERGY","NIFTY Infrastructure":"^CNXINFRA","NIFTY Commodities":"^CNXCMDT","NIFTY Consumption":"^CNXCONSUM","NIFTY Services Sector":"^CNXSERVICE"}
HINTS={"technology":"NIFTY IT","financial":"NIFTY Financial Services","bank":"NIFTY Bank","healthcare":"NIFTY Pharma","basic materials":"NIFTY Metal","consumer cyclical":"NIFTY Auto","consumer defensive":"NIFTY FMCG","real estate":"NIFTY Realty","energy":"NIFTY Energy","communication":"NIFTY Media","industrials":"NIFTY Infrastructure","utilities":"NIFTY Energy"}
NIFTY500_SOURCES=["https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv","https://niftyindices.com/IndexConstituent/ind_nifty500list.csv","https://www.nseindia.com/content/indices/ind_nifty500list.csv","https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data/main/nifty500_symbols.csv"]

def sf(x):
    try:
        x=float(x);return x if np.isfinite(x) else np.nan
    except:return np.nan

def ret(s,n):
    s=pd.to_numeric(s,errors="coerce").dropna();return (s.iloc[-1]/s.iloc[-n-1]-1)*100 if len(s)>n and s.iloc[-n-1]!=0 else np.nan

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

def rsi_series(s):
    s=pd.to_numeric(s,errors="coerce").dropna();d=s.diff();up=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean();rs=up/dn.replace(0,np.nan);return 100-100/(1+rs)

def rsi14(s):
    r=rsi_series(s);return sf(r.iloc[-1]) if len(r) else np.nan

def atr14(d):
    h=pd.to_numeric(d["High"],errors="coerce");l=pd.to_numeric(d["Low"],errors="coerce");c=pd.to_numeric(d["Close"],errors="coerce");tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);return sf(tr.rolling(14).mean().iloc[-1])

def sector_for(ticker):
    try:sec=(yf.Ticker(ticker).info.get("sector") or "").lower()
    except:sec=""
    for k,v in HINTS.items():
        if k in sec:return v
    return "NIFTY Services Sector"

@st.cache_data(ttl=86400,show_spinner=False)
def load_nifty500():
    headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/136 Safari/537.36","Accept":"text/csv,text/plain,*/*","Referer":"https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-500"};errors=[]
    for url in NIFTY500_SOURCES:
        try:
            r=requests.get(url,headers=headers,timeout=20);r.raise_for_status();text=r.text.strip()
            if not text:continue
            df=pd.read_csv(io.StringIO(text));cols={str(c).strip().lower():c for c in df.columns};symcol=cols.get("symbol",df.columns[0]);syms=df[symcol].astype(str).str.strip().str.upper().str.replace(".NS","",regex=False);syms=syms[syms.str.match(r"^[A-Z0-9&.-]+$",na=False)];syms=syms[~syms.isin(["SYMBOL","TICKER","NIFTY500"])];out=pd.DataFrame({"Symbol":syms.drop_duplicates()})
            if len(out)>=400:out["Ticker"]=out["Symbol"]+".NS";return out.reset_index(drop=True),url
            errors.append(f"{url}: only {len(out)} valid symbols")
        except Exception as e:errors.append(f"{url}: {type(e).__name__} {e}")
    raise RuntimeError("NIFTY 500 universe unavailable after all fallbacks. "+" | ".join(errors[-4:]))

def monthly_26m_signal(d,min_gap=26):
    if d is None or d.empty or "High" not in d.columns or "Close" not in d.columns:return None
    x=d.copy();x.index=pd.to_datetime(x.index).tz_localize(None);m=x.resample("ME").agg({"High":"max","Close":"last","Volume":"sum" if "Volume" in x.columns else "size"}).dropna(subset=["High","Close"])
    if len(m)<min_gap+2:return None
    highs=pd.to_numeric(m["High"],errors="coerce");closes=pd.to_numeric(m["Close"],errors="coerce");latest=m.index[-1];latest_close=sf(closes.iloc[-1]);prior=highs.iloc[:-1].dropna()
    if prior.empty:return None
    old_ath=sf(prior.max());ath_dates=prior.index[np.isclose(prior.values,old_ath,rtol=1e-10,atol=1e-10)];ath_date=ath_dates[-1] if len(ath_dates) else prior.idxmax();gap=(latest.year-ath_date.year)*12+(latest.month-ath_date.month);breakout=bool(pd.notna(latest_close) and latest_close>old_ath and gap>=min_gap);distance=(latest_close/old_ath-1)*100 if pd.notna(latest_close) and old_ath else np.nan
    return {"Status":"26M ATH BREAKOUT" if breakout else "Near / No Breakout","Monthly Close":latest_close,"Old ATH":old_ath,"ATH Date":ath_date.strftime("%Y-%m-%d"),"Months Gap":gap,"Breakout %":distance,"As Of":latest.strftime("%Y-%m-%d"),"Confirmed":breakout}

@st.cache_data(ttl=1200,show_spinner=False)
def macro_snapshot():
    ts=list(MACRO.values());raw=dl(ts,"2y");rows=[]
    for name,t in MACRO.items():
        d=one(raw,t,len(ts))
        if d.empty:continue
        c=d["Close"];rows.append({"Indicator":name,"Latest":sf(c.iloc[-1]),"1M %":ret(c,21),"3M %":ret(c,63),"12M %":ret(c,252),"As Of":pd.to_datetime(c.dropna().index[-1]).strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)

def macro_score(df):
    if df.empty:return 5.0
    def val(name,col="3M %"):
        r=df.loc[df["Indicator"]==name,col]
        return sf(r.iloc[0]) if len(r) else np.nan
    rules=[("NIFTY 50",1,2.0),("India VIX",-1,1.5),("USD/INR",-1,1.5),("Brent Crude",-1,1.5),("Dollar Index",-1,1.0),("US 10Y Yield",-1,1.0),("Gold",1,0.5)]
    s=0.0;w=0.0
    for name,direction,weight in rules:
        x=val(name)
        if pd.isna(x):continue
        s+=np.tanh((direction*x)/5.0)*weight;w+=weight
    return float(np.clip(5+5*(s/w),0,10)) if w else 5.0

@st.cache_data(ttl=1200,show_spinner=False)
def weekly_sector_scores():
    ts=["^NSEI"]+list(SECTORS.values());raw=dl(ts,"2y","1wk",True);b=one(raw,"^NSEI",len(ts))
    if b.empty:return pd.DataFrame()
    bc=b["Close"];rows=[]
    for name,t in SECTORS.items():
        d=one(raw,t,len(ts))
        if d.empty:continue
        s=d["Close"];r4=ret(s,4)-ret(bc,4);r13=ret(s,13)-ret(bc,13);r26=ret(s,26)-ret(bc,26);r52=ret(s,52)-ret(bc,52);acc=r4-r13;score=np.nanmean([r13,r26,r52]);status="LEADER" if score>0 and r13>0 and r26>0 else "IMPROVING" if r4>0 and acc>0 else "WEAKENING" if score>0 else "LAGGARD";rows.append({"Sector":name,"Status":status,"4W RS":r4,"13W RS":r13,"26W RS":r26,"52W RS":r52,"Acceleration":acc,"RS Score":score})
    return pd.DataFrame(rows).sort_values(["RS Score","Acceleration"],ascending=False)

def evaluate_setups(d,nc,sc,sector_score):
    c=pd.to_numeric(d["Close"],errors="coerce").dropna();o=pd.to_numeric(d["Open"],errors="coerce");h=pd.to_numeric(d["High"],errors="coerce");l=pd.to_numeric(d["Low"],errors="coerce");p=sf(c.iloc[-1]);green=bool(p>sf(o.iloc[-1]));sma=lambda k:c.rolling(k).mean();ema=lambda k:c.ewm(span=k,adjust=False).mean();ma200=sma(200);ma198=sma(198);ma30=sma(30);ma28=sma(28);ma62=sma(62);ma20=sma(20);ema200=ema(200);prev50=sf(h.iloc[-51:-1].max()) if len(h)>=51 else np.nan
    s1={"Close > 200DMA":p>sf(ma200.iloc[-1]),"Close > previous 50-day High":pd.notna(prev50) and p>prev50,"Green candle (Close > Open)":green};s2={"Close < 198DMA":p<sf(ma198.iloc[-1]),"Close > 200DMA":p>sf(ma200.iloc[-1]),"198DMA >= 200DMA":sf(ma198.iloc[-1])>=sf(ma200.iloc[-1])};prev20_ma30=sf(ma30.iloc[-21:-1].max()) if len(ma30)>=21 else np.nan;s3={"Close > previous 20-day max of 30DMA":pd.notna(prev20_ma30) and p>prev20_ma30,"Close > 30DMA":p>sf(ma30.iloc[-1]),"Close < 28DMA":p<sf(ma28.iloc[-1]),"Daily Low <= 30DMA":sf(l.iloc[-1])<=sf(ma30.iloc[-1]),"Green candle":green};stock22=ret(c,22);nifty22=ret(nc,22);rs22=stock22-nifty22 if pd.notna(stock22) and pd.notna(nifty22) else np.nan;s4={"Stock 22D return > NIFTY 50 22D return":pd.notna(rs22) and rs22>0,"NIFTY 22D return > 0%":pd.notna(nifty22) and nifty22>0,"NIFTY 22D return < 50%":pd.notna(nifty22) and nifty22<50,"NIFTY 22D absolute move < 3.7%":pd.notna(nifty22) and abs(nifty22)<3.7};mid=ma20;std=c.rolling(20).std(ddof=0);upper=mid+2*std;lower=mid-2*std;bbw=(upper-lower)/mid;cross_upper=bool(len(h)>1 and sf(h.iloc[-1])>sf(upper.iloc[-1]) and sf(h.iloc[-2])<=sf(upper.iloc[-2]));s5={"BB Width <= 5%":sf(bbw.iloc[-1])<=0.05,"Daily High crossed above Upper BB(20,2)":cross_upper,"Close > EMA200":p>sf(ema200.iloc[-1])};r=rsi_series(c);rnow=sf(r.iloc[-1]);rprev=sf(r.iloc[-2]) if len(r)>1 else np.nan;rising62=all(sf(ma62.iloc[-i])>sf(ma62.iloc[-i-1]) for i in range(1,6)) if len(ma62)>=7 else False;s6={"62DMA rising for 5 sessions":rising62,"RSI(14) > 30":pd.notna(rnow) and rnow>30,"RSI(14) < 43":pd.notna(rnow) and rnow<43,"RSI crossed above 35":pd.notna(rprev) and rprev<=35 and rnow>35};old_ma20_max=sf(ma20.iloc[-30:-10].max()) if len(ma20)>=30 else np.nan;s7={"Current 20DMA > 10-days-ago Max(20, 20DMA)":pd.notna(old_ma20_max) and sf(ma20.iloc[-1])>old_ma20_max,"Daily Low <= Lower Bollinger Band(20,2)":sf(l.iloc[-1])<=sf(lower.iloc[-1]),"Daily Close > Lower Bollinger Band(20,2)":p>sf(lower.iloc[-1]),"Green candle (Close > Open)":green};raw=[("#1 50D High + 200DMA + Green",s1),("#2 198/200 DMA Goldmine",s2),("#3 30MA + Green Candle",s3),("#4 Stock Outperform NIFTY-50",s4),("#5 Bollinger Squeeze Breakout",s5),("#6 R62 Trend + RSI Recovery",s6),("#7 20DMA Uptrend + LBB Bounce + Green",s7)];rows=[];details={}
    for name,checks in raw:
        passed=sum(bool(v) for v in checks.values());confirmed=all(checks.values());rows.append({"Confirmed":"YES" if confirmed else "—","Technical Setup":name,"Status":"CONFIRMED" if confirmed else "Not Confirmed","Passed":f"{passed}/{len(checks)}","Priority":1 if confirmed else 2});details[name]=pd.DataFrame([{"Condition":k,"Result":"PASS" if v else "FAIL"} for k,v in checks.items()])
    table=pd.DataFrame(rows).sort_values(["Priority","Technical Setup"]).drop(columns=["Priority"]);return table,details,[name for name,checks in raw if all(checks.values())],{"Stock22":stock22,"Nifty22":nifty22,"RS22":rs22}

def swing(symbol,mscore):
    ticker=symbol if symbol.endswith(".NS") else symbol+".NS";sname=sector_for(ticker);sticker=SECTORS.get(sname,"^CNXSERVICE");ts=[ticker,"^NSEI",sticker];raw=dl(ts,"2y");d=one(raw,ticker,3);n=one(raw,"^NSEI",3);sec=one(raw,sticker,3)
    if d.empty or n.empty:return None
    c=pd.to_numeric(d["Close"],errors="coerce").dropna();nc=pd.to_numeric(n["Close"],errors="coerce").dropna();sc=pd.to_numeric(sec["Close"],errors="coerce").dropna() if not sec.empty else pd.Series(dtype=float);price=sf(c.iloc[-1]);n20=sf(nc.rolling(20).mean().iloc[-1]);n50=sf(nc.rolling(50).mean().iloc[-1]);np_=sf(nc.iloc[-1]);market=100*sum([np_>n20,np_>n50,ret(nc,20)>0])/3;sr4=ret(sc,20)-ret(nc,20) if len(sc)>21 else np.nan;sr13=ret(sc,65)-ret(nc,65) if len(sc)>66 else np.nan;sector_score=float(np.clip(50+8*(0 if pd.isna(sr4) else sr4)+4*(0 if pd.isna(sr13) else sr13),0,100));rsn=ret(c,20)-ret(nc,20);rss=ret(c,20)-ret(sc,20) if len(sc)>21 else np.nan;rs_score=float(np.clip(50+8*rsn+6*(0 if pd.isna(rss) else rss),0,100));setups,details,confirmed,rsmeta=evaluate_setups(d,nc,sc,sector_score);setup_score=100*len(confirmed)/7;macro_component=float(np.clip(mscore*10,0,100));technical_gate=len(confirmed)>0;weighted=market*.15+sector_score*.20+rs_score*.20+setup_score*.40+macro_component*.05;label="BUY SETUP FOUND" if technical_gate and weighted>=55 else "TECHNICAL SETUP FOUND — CHECK CONTEXT" if technical_gate else "NO APPROVED TECHNICAL SETUP / WAIT";ma20=sf(c.rolling(20).mean().iloc[-1]);low20=sf(pd.to_numeric(d["Low"],errors="coerce").iloc[-20:].min());atr=atr14(d);stops=[x for x in [ma20,low20,price-1.5*atr if pd.notna(atr) else np.nan] if pd.notna(x) and x<price];stop=max(stops) if stops else price*.96;risk=price-stop;target=price+2*risk
    return {"Ticker":ticker,"Price":price,"Sector":sname,"As Of":pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d"),"Final":label,"Score":weighted,"Gate":technical_gate,"Setups":setups,"SetupDetails":details,"Confirmed":confirmed,"Market":market,"SectorScore":sector_score,"RSScore":rs_score,"RSN":rsn,"RSS":rss,"RSI":rsi14(c),"Entry":price,"SL":stop,"Target":target,"Trail":ma20,"RSmeta":rsmeta}

macro=macro_snapshot();mscore=macro_score(macro);regime="SUPPORTIVE / RISK-ON" if mscore>=7.5 else "MIXED / SELECTIVE" if mscore>=5 else "CAUTIOUS" if mscore>=3 else "RISK-OFF"
st.title("Professional Investment Research Lab");a,b,c=st.columns(3);a.metric("Macro Score",f"{mscore:.1f}/10");b.metric("Macro Regime",regime);c.metric("Data Cache","20 min");tabs=st.tabs(["Long-Term / Positional","Swing Trading","26M ATH Scanner","Sector Rotation","Fundamentals"])
with tabs[0]:st.subheader("Macro Regime Diagnostics");st.info("Macro positional/long-term background filter है; direct BUY signal नहीं.");st.dataframe(macro,use_container_width=True,hide_index=True);st.markdown("**Decision:** Macro → Sector Leadership → 26M ATH → Fundamentals → Final Opportunity")
with tabs[1]:
    st.subheader("Live Swing Trading Engine — Your Technical Setups #1 to #7");st.caption("BUY technical gate आपके approved setups पर आधारित है. कोई approved setup confirmed न हो तो BUY नहीं आएगा.");c1,c2=st.columns([2,1]);symbol=c1.text_input("NSE Symbol",value="NMDC").strip().upper();run=c2.button("Run Live Swing Analysis",type="primary",use_container_width=True)
    if run and symbol:
        with st.spinner("Checking your 7 technical setups…"):
            try:st.session_state["sw"]=swing(symbol,mscore)
            except Exception as e:st.error(f"Analysis failed: {e}")
    x=st.session_state.get("sw")
    if x:
        st.header(f"{x['Final']} — {x['Ticker']}");m1,m2,m3,m4,m5=st.columns(5);m1.metric("Swing Score",f"{x['Score']:.1f}/100");m2.metric("Price",f"₹{x['Price']:.2f}");m3.metric("Sector",x['Sector']);m4.metric("RSI(14)",f"{x['RSI']:.1f}");m5.metric("As Of",x['As Of'])
        if x['Gate']:
            pills=''.join([f"<span class='confirmed-pill'>{name}</span>" for name in x['Confirmed']]);st.markdown(f"<div class='confirmed-banner'><b>CONFIRMED TECHNICAL SETUP DETECTED</b><br>{pills}</div>",unsafe_allow_html=True)
        else:st.markdown("<div class='waitgate'><h3>BUY LOCKED</h3>None of Technical Setups #1–#7 is fully confirmed.</div>",unsafe_allow_html=True)
        st.subheader("Technical Setups #1–#7 — Live Results");st.dataframe(x['Setups'],use_container_width=True,hide_index=True)
        for name in x['Setups']['Technical Setup']:
            row=x['Setups'][x['Setups']['Technical Setup']==name].iloc[0];prefix="CONFIRMED — " if row['Confirmed']=="YES" else ""
            with st.expander(prefix+name):st.dataframe(x['SetupDetails'][name],use_container_width=True,hide_index=True)
        st.subheader("Context Filters");q1,q2,q3=st.columns(3);q1.metric("Market Condition",f"{x['Market']:.0f}/100");q2.metric("Sector Leadership",f"{x['SectorScore']:.0f}/100");q3.metric("Stock Relative Strength",f"{x['RSScore']:.0f}/100");st.caption(f"Stock vs NIFTY 1M RS: {x['RSN']:.2f}%"+(f" | Stock vs Sector: {x['RSS']:.2f}%" if pd.notna(x['RSS']) else ""));st.subheader("Trade Planning Reference");r1,r2,r3,r4=st.columns(4);r1.metric("Entry Reference",f"₹{x['Entry']:.2f}");r2.metric("Stop Loss",f"₹{x['SL']:.2f}");r3.metric("2R Target",f"₹{x['Target']:.2f}");r4.metric("Trail Ref (20DMA)",f"₹{x['Trail']:.2f}")
    st.subheader("Live Weekly Sector Leadership");st.dataframe(weekly_sector_scores().round(2),use_container_width=True,hide_index=True)
with tabs[2]:
    st.subheader("Strict 26-Month ATH Breakout Scanner");st.caption("Rule: Monthly CLOSE must be above every prior monthly HIGH, and the old ATH month must be at least 26 months old.")
    try:universe,universe_source=load_nifty500();st.success(f"NIFTY 500 universe loaded: {len(universe)} symbols. 403-safe fallback active.");st.caption(f"Universe source used: {universe_source}")
    except Exception as e:universe=pd.DataFrame();st.error(str(e))
    if not universe.empty:
        c1,c2,c3=st.columns(3);batch=c1.selectbox("Stocks to scan",[50,100,200,500],index=1);gap=c2.number_input("Minimum ATH gap (months)",12,120,26,1);near=c3.slider("Near ATH range",1,10,5,1)
        if st.button(f"Run 26M ATH Scan on {min(batch,len(universe))} stocks",type="primary",use_container_width=True):
            scan_u=universe.head(int(batch)).copy();tickers=scan_u["Ticker"].tolist();rows=[];prog=st.progress(0,text="Downloading historical prices…")
            try:raw26=dl(tickers,period="max",interval="1d",adjust=False)
            except Exception as e:raw26=pd.DataFrame();st.error(f"Price download failed: {e}")
            for i,(_,r) in enumerate(scan_u.iterrows()):
                d=one(raw26,r["Ticker"],len(tickers));sig=monthly_26m_signal(d,int(gap))
                if sig:rows.append({"Symbol":r["Symbol"],**sig})
                prog.progress((i+1)/max(1,len(scan_u)),text=f"Scanning {i+1}/{len(scan_u)} — {r['Symbol']}")
            prog.empty();st.session_state["scan26"]=pd.DataFrame(rows)
        s26=st.session_state.get("scan26",pd.DataFrame())
        if not s26.empty:
            confirmed=s26[s26["Confirmed"]==True].copy();near_df=s26[(s26["Confirmed"]==False)&(s26["Breakout %"]>=-near)&(s26["Months Gap"]>=gap)].copy();a1,a2,a3=st.columns(3);a1.metric("Scanned",len(s26));a2.metric("Confirmed 26M ATH",len(confirmed));a3.metric(f"Near ATH <= {near}%",len(near_df));show=pd.concat([confirmed.sort_values("Breakout %",ascending=False),near_df.sort_values("Breakout %",ascending=False)]).drop_duplicates("Symbol");st.dataframe(show,use_container_width=True,hide_index=True)
        else:st.info("Run the scanner to generate results.")
with tabs[3]:st.subheader("Weekly Sector Rotation / Leadership");st.dataframe(weekly_sector_scores().round(2),use_container_width=True,hide_index=True)
with tabs[4]:st.info("Fundamental quality remains part of the long-term/positional workflow.")