from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from graham import fetch_graham_data

st.set_page_config(page_title="VALUE 4-of-6 Qualification", page_icon="✅", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1650px;}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.20); border-radius:12px; padding:10px; min-width:0;}
[data-testid="stMetricValue"] {white-space:normal; overflow:visible; text-overflow:clip;}
@media (max-width: 900px) {
  .block-container {padding:1.1rem .65rem 1.5rem !important;}
  h1 {font-size:1.7rem !important;}
  [data-testid="stMetricValue"] {font-size:1.25rem;}
  .stDataFrame {overflow-x:auto;}
}
</style>
""", unsafe_allow_html=True)

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

MACRO_ASSETS = {
    "NIFTY 50": "^NSEI", "India VIX": "^INDIAVIX", "USD/INR": "INR=X",
    "Brent Crude": "BZ=F", "Dollar Index": "DX-Y.NYB", "US 10Y Yield": "^TNX", "Gold": "GC=F",
}


def safe_float(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def value_migration_score(industry):
    text=str(industry).lower()
    for keys, score, theme in VALUE_MIGRATION_MAP:
        if any(k in text for k in keys):
            return score, theme
    return 50, "Broad / Neutral"


def _close_series(raw, ticker, total):
    try:
        if raw.empty: return pd.Series(dtype=float)
        if total==1 and not isinstance(raw.columns,pd.MultiIndex):
            d=raw
        elif isinstance(raw.columns,pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
            d=raw[ticker]
        elif isinstance(raw.columns,pd.MultiIndex) and ticker in raw.columns.get_level_values(-1):
            d=raw.xs(ticker,axis=1,level=-1)
        else:
            return pd.Series(dtype=float)
        return pd.to_numeric(d["Close"],errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=1200, show_spinner=False)
def macro_support_score():
    tickers=list(MACRO_ASSETS.values())
    try:
        raw=yf.download(tickers=tickers,period="2y",interval="1d",group_by="ticker",auto_adjust=False,threads=True,progress=False,timeout=25)
    except TypeError:
        raw=yf.download(tickers=tickers,period="2y",interval="1d",group_by="ticker",auto_adjust=False,threads=True,progress=False)
    except Exception:
        raw=pd.DataFrame()
    specs=[
        ("NIFTY 50",+1,2.2,8.0),("India VIX",-1,1.7,15.0),("USD/INR",-1,1.5,4.0),
        ("Brent Crude",-1,1.5,12.0),("Dollar Index",-1,1.0,6.0),("US 10Y Yield",-1,1.0,8.0),("Gold",-1,.6,12.0),
    ]
    weighted=weights=0.0
    rows=[]
    for name,direction,weight,scale in specs:
        c=_close_series(raw,MACRO_ASSETS[name],len(tickers))
        if len(c)<65: continue
        last=safe_float(c.iloc[-1])
        r1=(last/safe_float(c.iloc[-22])-1)*100 if len(c)>22 else np.nan
        r3=(last/safe_float(c.iloc[-64])-1)*100
        impulse=.75*r3+.25*(r1 if pd.notna(r1) else r3)
        sig=float(np.tanh(direction*impulse/scale))
        weighted += sig*weight; weights += weight
        rows.append({"Driver":name,"1M %":r1,"3M %":r3,"Contribution":sig*weight,"As Of":pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d")})
    score=float(np.clip(5+5*weighted/weights,0,10))*10 if weights else 50.0
    coverage=int(round(100*weights/sum(s[2] for s in specs))) if specs else 0
    return score, coverage, pd.DataFrame(rows)


def graham_score(symbol):
    try:
        g=fetch_graham_data(symbol)
    except Exception:
        return np.nan,0
    tests=[
        None if pd.isna(g["sales"]) else g["sales"]>500e7,
        None if pd.isna(g["current_ratio"]) else g["current_ratio"]>2,
        None if pd.isna(g["long_debt"]) or pd.isna(g["nwc"]) else g["long_debt"]<g["nwc"],
        g["eps_positive_10y"],g["dividend20"],g["eps_growth_10y"],
        None if pd.isna(g["pe3"]) else g["pe3"]<15,
        None if pd.isna(g["pb"]) else g["pb"]<1.5,
        None if pd.isna(g["combined"]) else g["combined"]<22.5,
        None if pd.isna(g["graham_no"]) or pd.isna(g["price"]) else g["graham_no"]>g["price"],
    ]
    ev=[x for x in tests if x is not None]
    return (100*sum(bool(x) for x in ev)/len(ev) if ev else np.nan),len(ev)


def qualification_label(count):
    if count>=6: return "💎 ELITE VALUE — 6/6"
    if count==5: return "🟢 STRONG VALUE — 5/6"
    if count==4: return "✅ VALUE STOCK — 4/6"
    if count==3: return "🟡 NEAR VALUE — 3/6"
    return "⚪ NOT QUALIFIED"


st.title("✅ VALUE Stock Qualification — 4 out of 6 Rule")
st.caption("Your rule: a stock becomes a VALUE STOCK when at least 4 of the 6 master factors pass. Five passes = Strong Value; six passes = Elite Value.")

macro_score, macro_cov, macro_df = macro_support_score()
scan=st.session_state.get("value_scan",pd.DataFrame())
leaders=st.session_state.get("value_leaders",pd.DataFrame())
pfactors=dict(st.session_state.get("value_pfactors",{}))
graham_scores=dict(st.session_state.get("value_graham_scores",{}))

m1,m2,m3,m4=st.columns(4)
m1.metric("Qualification Rule","≥4 / 6")
m2.metric("Macro Support",f"{macro_score:.0f}/100")
m3.metric("Macro Pass","PASS" if macro_score>=60 else "FAIL")
m4.metric("Macro Coverage",f"{macro_cov}%")

with st.expander("How each master factor becomes PASS",expanded=True):
    st.markdown("""
1. **Value Migration PASS** → theme score **≥70/100**.  
2. **Macro Support PASS** → live macro score **≥60/100**.  
3. **P Factor PASS** → at least **2 of its 3 sub-rules** pass: market-cap ₹5,000–25,000 Cr, D/E <0.5, profit breakout.  
4. **Leader PASS** → sector Leader Score **≥65/100** versus NIFTY 50 and other sectors.  
5. **Value/Graham PASS** → Graham score **≥60%** with at least **4 evaluable rules**.  
6. **B Factor PASS** → any valid photo-defined breakout setup with **B-Factor score ≥70**.

**Final classification:** 4/6 = VALUE STOCK, 5/6 = STRONG VALUE, 6/6 = ELITE VALUE.
""")

if scan.empty:
    st.warning("पहले **VALUE Stock Engine → VALUE Finder** में price scan चलाएँ। उसके बाद यहाँ 4-of-6 qualification बनेगा।")
    st.stop()

lscore=leaders.set_index("Industry")["Leader Score"].to_dict() if not leaders.empty else {}
rows=[]
for _,r in scan.iterrows():
    sym=str(r["Symbol"])
    vm_score,vm_theme=value_migration_score(r.get("Industry",""))
    p=pfactors.get(sym,{})
    p_passed=p.get("P Passed",np.nan)
    p_available=p.get("P Available",np.nan)
    p_master=(pd.notna(p_passed) and p_passed>=2)
    leader=safe_float(lscore.get(r.get("Industry"),np.nan))
    g=graham_scores.get(sym,{}) if isinstance(graham_scores.get(sym,{}),dict) else {}
    gs=safe_float(g.get("score",np.nan)); ga=safe_float(g.get("assessed",np.nan))

    passes={
        "Value Migration": vm_score>=70,
        "Macro Support": macro_score>=60,
        "P Factor": bool(p_master),
        "Leader": pd.notna(leader) and leader>=65,
        "Graham Value": pd.notna(gs) and pd.notna(ga) and ga>=4 and gs>=60,
        "B Factor": safe_float(r.get("B Factor Score"))>=70,
    }
    count=sum(bool(x) for x in passes.values())
    pending=[]
    if pd.isna(p_passed): pending.append("P Factor")
    if pd.isna(gs): pending.append("Graham")
    rows.append({
        "Symbol":sym,"Company":r.get("Company",""),"Industry":r.get("Industry",""),
        "VALUE Qualification":qualification_label(count),"Factors Passed":count,
        "1 Value Migration":"PASS" if passes["Value Migration"] else "FAIL",
        "2 Macro":"PASS" if passes["Macro Support"] else "FAIL",
        "3 P Factor":"PASS" if passes["P Factor"] else "PENDING" if pd.isna(p_passed) else "FAIL",
        "4 Leader":"PASS" if passes["Leader"] else "FAIL",
        "5 Graham":"PASS" if passes["Graham Value"] else "PENDING" if pd.isna(gs) else "FAIL",
        "6 B Factor":"PASS" if passes["B Factor"] else "FAIL",
        "VM Theme":vm_theme,"VM Score":vm_score,"Leader Score":leader,
        "P Sub-points Passed":p_passed,"Graham Score %":gs,"B Factor":r.get("B Factor",""),"B Score":r.get("B Factor Score",np.nan),
        "Pending":", ".join(pending) if pending else "None",
    })

out=pd.DataFrame(rows).sort_values(["Factors Passed","B Score","Leader Score"],ascending=[False,False,False])

q1,q2,q3,q4=st.columns(4)
q1.metric("Elite 6/6",int((out["Factors Passed"]==6).sum()))
q2.metric("Strong 5/6",int((out["Factors Passed"]==5).sum()))
q3.metric("VALUE 4/6",int((out["Factors Passed"]==4).sum()))
q4.metric("Qualified ≥4",int((out["Factors Passed"]>=4).sum()))

st.subheader("💎 Your VALUE Stock List")
qualified=out[out["Factors Passed"]>=4]
if qualified.empty:
    st.info("अभी 4/6 complete passes वाला stock नहीं है — लेकिन P Factor/Graham pending stocks को complete करने पर qualification बदल सकती है।")
else:
    st.dataframe(qualified,use_container_width=True,hide_index=True,column_config={
        "Factors Passed":st.column_config.ProgressColumn(min_value=0,max_value=6,format="%d/6"),
        "VM Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
        "Leader Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
        "Graham Score %":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f%%"),
        "B Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
    })

st.subheader("Complete pending P/Graham factors")
base_order=out.sort_values(["Factors Passed","B Score"],ascending=[False,False])
topn=st.selectbox("Graham batch",[5,10,20,30],index=1)
if st.button(f"🧮 Run Graham for Top {topn}",type="primary",use_container_width=True):
    prog=st.progress(0,text="Calculating Graham confirmation…")
    for i,(_,r) in enumerate(base_order.head(int(topn)).iterrows()):
        sym=str(r["Symbol"])
        try:
            score,assessed=graham_score(sym)
            graham_scores[sym]={"score":score,"assessed":assessed}
        except Exception:
            graham_scores[sym]={"score":np.nan,"assessed":0}
        prog.progress((i+1)/max(1,min(int(topn),len(base_order))),text=f"Graham {i+1}: {sym}")
    prog.empty()
    st.session_state["value_graham_scores"]=graham_scores
    st.rerun()

st.caption("P Factor is intentionally one master factor, not three separate master votes. This prevents its three sub-rules from overpowering Value Migration, Macro, Leader, Graham and B Factor. Run P-Factor batch from the VALUE Stock Engine to fill pending P results.")

with st.expander("All scanned stocks",expanded=False):
    st.dataframe(out,use_container_width=True,hide_index=True)

st.divider()
st.caption(f"4-of-6 VALUE Qualification v1 · Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Research filter only, not investment advice.")