import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Race Strategy", page_icon="🏁", layout="wide")

st.title("🏁 Race Strategy — Sector → Stock Leadership")
st.caption("Weekly Relative Strength | NIFTY benchmark | 3M + 6M confirmation | Emerging Leader detection")

SECTORS = {
    "Auto": "^CNXAUTO",
    "Bank": "^NSEBANK",
    "Financial Services": "NIFTY_FIN_SERVICE.NS",
    "FMCG": "^CNXFMCG",
    "IT": "^CNXIT",
    "Media": "^CNXMEDIA",
    "Metal": "^CNXMETAL",
    "Pharma": "^CNXPHARMA",
    "PSU Bank": "^CNXPSUBANK",
    "Realty": "^CNXREALTY",
    "Energy": "^CNXENERGY",
    "Infrastructure": "^CNXINFRA",
}
BENCHMARK = "^NSEI"

STOCKS = {
    "Auto": ["MARUTI.NS","M&M.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","TVSMOTOR.NS","HEROMOTOCO.NS","ASHOKLEY.NS"],
    "Bank": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","INDUSINDBK.NS","BANKBARODA.NS","FEDERALBNK.NS"],
    "Financial Services": ["BAJFINANCE.NS","BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS","JIOFIN.NS","CHOLAFIN.NS","SHRIRAMFIN.NS"],
    "FMCG": ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","GODREJCP.NS","MARICO.NS","TATACONSUM.NS"],
    "IT": ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","LTIM.NS","PERSISTENT.NS","COFORGE.NS"],
    "Metal": ["TATASTEEL.NS","HINDALCO.NS","JSWSTEEL.NS","VEDL.NS","NMDC.NS","SAIL.NS","NATIONALUM.NS","JINDALSTEL.NS"],
    "Pharma": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","LUPIN.NS","AUROPHARMA.NS","TORNTPHARM.NS","ALKEM.NS"],
    "PSU Bank": ["SBIN.NS","BANKBARODA.NS","PNB.NS","CANBK.NS","UNIONBANK.NS","INDIANB.NS","BANKINDIA.NS"],
    "Realty": ["DLF.NS","LODHA.NS","GODREJPROP.NS","OBEROIRLTY.NS","PRESTIGE.NS","PHOENIXLTD.NS","BRIGADE.NS"],
    "Energy": ["RELIANCE.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","COALINDIA.NS","BPCL.NS","IOC.NS","TATAPOWER.NS"],
    "Infrastructure": ["LT.NS","ADANIPORTS.NS","ULTRACEMCO.NS","GRASIM.NS","SIEMENS.NS","ABB.NS","CUMMINSIND.NS"],
}

@st.cache_data(ttl=3600, show_spinner=False)
def closes(tickers, period="1y"):
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            c = raw["Close"].copy()
        except Exception:
            c = raw.xs("Close", axis=1, level=1).copy()
    else:
        c = raw[["Close"]].copy()
        c.columns = [tickers[0] if isinstance(tickers, list) else tickers]
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c.dropna(how="all")

def ret(s, days):
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    j = max(0, len(s)-1-days)
    if s.iloc[j] <= 0:
        return np.nan
    return (s.iloc[-1]/s.iloc[j]-1)*100

def weekly_slope(rs):
    x = rs.dropna().resample("W-FRI").last().dropna()
    if len(x) < 5:
        return np.nan
    y = x.iloc[-5:].values.astype(float)
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])

def build_race(price, mapping):
    if BENCHMARK not in price.columns:
        return pd.DataFrame()
    n = price[BENCHMARK].dropna()
    rows=[]
    for name,t in mapping.items():
        if t not in price.columns:
            continue
        pair=pd.concat([price[t],n],axis=1).dropna()
        if len(pair)<30:
            continue
        p,b=pair.iloc[:,0],pair.iloc[:,1]
        r3=ret(p,63)-ret(b,63)
        r6=ret(p,126)-ret(b,126)
        rs=(p/b)*100
        slope=weekly_slope(rs)
        accel=r3-r6
        score=(0 if pd.isna(r3) else r3*0.35)+(0 if pd.isna(r6) else r6*0.45)+(0 if pd.isna(accel) else accel*0.20)
        if r3>0 and r6>0 and slope>0:
            label="🟢 STRONG LEADER"
        elif r3>0 and slope>0 and (r6<=0 or accel>2):
            label="🔵 EMERGING LEADER"
        elif r6>0 and slope<=0:
            label="🟡 WEAKENING"
        else:
            label="🔴 UNDERPERFORMER"
        rows.append({"Sector":name,"Ticker":t,"3M RS vs NIFTY %":r3,"6M RS vs NIFTY %":r6,"Acceleration %":accel,"Weekly RS Slope":slope,"Race Score":score,"Race Status":label})
    return pd.DataFrame(rows).sort_values("Race Score",ascending=False).reset_index(drop=True)

with st.spinner("Calculating weekly 3M/6M Race Strategy..."):
    px=closes([BENCHMARK]+list(SECTORS.values()),"1y")
    race=build_race(px,SECTORS)

if race.empty:
    st.error("Sector data is unavailable right now. Refresh later; no leader is inferred from missing data.")
    st.stop()

strong=race[race["Race Status"].str.contains("STRONG")]
emerging=race[race["Race Status"].str.contains("EMERGING")]

c1,c2,c3,c4=st.columns(4)
c1.metric("Strong Leaders",len(strong))
c2.metric("Emerging Leaders",len(emerging))
c3.metric("Top Race Sector",race.iloc[0]["Sector"])
c4.metric("Top Race Score",f"{race.iloc[0]['Race Score']:.1f}")

st.subheader("Sector Race Leaderboard")
st.dataframe(race[["Sector","3M RS vs NIFTY %","6M RS vs NIFTY %","Acceleration %","Weekly RS Slope","Race Score","Race Status"]].style.format({"3M RS vs NIFTY %":"{:.2f}","6M RS vs NIFTY %":"{:.2f}","Acceleration %":"{:.2f}","Weekly RS Slope":"{:.3f}","Race Score":"{:.2f}"}),use_container_width=True,hide_index=True)

fig=go.Figure()
for _,r in race.iterrows():
    fig.add_trace(go.Scatter(x=[r["6M RS vs NIFTY %"]],y=[r["3M RS vs NIFTY %"]],mode="markers+text",text=[r["Sector"]],textposition="top center",marker=dict(size=12),name=r["Sector"],hovertemplate=f"{r['Sector']}<br>6M RS: {r['6M RS vs NIFTY %']:.2f}%<br>3M RS: {r['3M RS vs NIFTY %']:.2f}%<extra></extra>"))
fig.add_hline(y=0,line_dash="dash")
fig.add_vline(x=0,line_dash="dash")
fig.update_layout(title="Race Map — 6M Strength vs 3M Strength",xaxis_title="6M RS vs NIFTY (%)",yaxis_title="3M RS vs NIFTY (%)",height=600,showlegend=False)
st.plotly_chart(fig,use_container_width=True)

st.info("Rule: Leader is not decided by one week's rank. STRONG = 3M & 6M outperform NIFTY + weekly RS rising. EMERGING = recent 3M strength/rising RS while longer 6M leadership is still developing.")

st.divider()
st.subheader("Sector → Stock Race")
default_sector=(emerging.iloc[0]["Sector"] if not emerging.empty else race.iloc[0]["Sector"])
choices=[x for x in race["Sector"].tolist() if x in STOCKS]
idx=choices.index(default_sector) if default_sector in choices else 0
sel=st.selectbox("Select leader / emerging sector",choices,index=idx)
stock_tickers=STOCKS.get(sel,[])
spx=closes([BENCHMARK]+stock_tickers,"1y")
stock_map={x.replace(".NS",""):x for x in stock_tickers}
srace=build_race(spx,stock_map)
if not srace.empty:
    srace=srace.rename(columns={"Sector":"Stock"})
    st.dataframe(srace[["Stock","3M RS vs NIFTY %","6M RS vs NIFTY %","Acceleration %","Weekly RS Slope","Race Score","Race Status"]].style.format({"3M RS vs NIFTY %":"{:.2f}","6M RS vs NIFTY %":"{:.2f}","Acceleration %":"{:.2f}","Weekly RS Slope":"{:.3f}","Race Score":"{:.2f}"}),use_container_width=True,hide_index=True)
    st.success(f"Current Race candidate in {sel}: {srace.iloc[0]['Stock']} — {srace.iloc[0]['Race Status']}. Confirm actual weekly/ATH/sideways breakout before entry.")
else:
    st.warning("Stock Race data unavailable for this sector.")

st.divider()
st.subheader("Race Strategy Checklist")
st.markdown("""
1. **Benchmark:** NIFTY 50.  
2. **Timeframe:** Weekly trend; use **3M + 6M** together.  
3. **Sector:** Prefer rising/outperforming sectors; detect **Emerging Leaders**, not only already-extended #1 sectors.  
4. **Stock:** Run the same race inside the selected sector.  
5. **Entry confirmation:** Relative-strength winner alone is **not a Buy**; confirm ATH / sideways / weekly breakout.  
6. **Risk:** Weak/unclear leadership → cash is allowed; keep sector concentration controlled.  
""")
st.caption("Research tool only. Scores are heuristic relative-strength rankings, not guaranteed returns or investment advice.")