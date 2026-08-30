import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Weekly Sector Leaders", page_icon="📈", layout="wide")

st.title("📈 Weekly Sector Leaders vs NIFTY 50")
st.caption("Weekly relative-performance view. Every line is rebased to 100 at the selected starting point so sectors with different index levels can be compared fairly against NIFTY 50.")

SECTORS = {
    # Core sectoral indices
    "NIFTY Auto": "^CNXAUTO",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY Consumer Durables": "NIFTY_CONSR_DURBL.NS",
    "NIFTY Financial Services": "NIFTY_FIN_SERVICE.NS",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY Healthcare": "NIFTY_HEALTHCARE.NS",
    "NIFTY IT": "^CNXIT",
    "NIFTY Media": "^CNXMEDIA",
    "NIFTY Metal": "^CNXMETAL",
    "NIFTY Oil & Gas": "NIFTY_OIL_AND_GAS.NS",
    "NIFTY Pharma": "^CNXPHARMA",
    "NIFTY Private Bank": "NIFTY_PVT_BANK.NS",
    "NIFTY PSU Bank": "^CNXPSUBANK",
    "NIFTY Realty": "^CNXREALTY",

    # Broad sector / economy and high-priority thematic indices
    "NIFTY Commodities": "^CNXCMDT",
    "NIFTY Consumption": "^CNXCONSUM",
    "NIFTY Energy": "^CNXENERGY",
    "NIFTY Infrastructure": "^CNXINFRA",
    "NIFTY Services Sector": "^CNXSERVICE",
    "NIFTY India Defence": "NIFTY_IND_DEFENCE.NS",
    "NIFTY India Digital": "NIFTY_IND_DIGITAL.NS",
    "NIFTY India Manufacturing": "NIFTY_INDIA_MFG.NS",
}
BENCHMARK = "^NSEI"

@st.cache_data(ttl=1200, show_spinner=False)
def load_weekly(years):
    """Download daily prices first, then build Friday weekly closes.

    Yahoo does not consistently return interval="1wk" history for newer Indian
    indices (notably NIFTY India Defence). Daily data is more complete and the
    local resample keeps every sector on the same weekly calendar.
    """
    tickers = [BENCHMARK] + list(SECTORS.values())
    period = {1:"2y", 2:"3y", 3:"5y", 5:"10y"}.get(years, "5y")
    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": BENCHMARK})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.dropna(how="all").resample("W-FRI").last().dropna(how="all")
    return close.tail(max(55, years * 53 + 4))

def rebase(s):
    s = s.dropna()
    return (s / s.iloc[0] * 100.0) if len(s) else s

def perf(s, weeks):
    s = s.dropna()
    if len(s) <= weeks:
        return np.nan
    return (s.iloc[-1] / s.iloc[-1-weeks] - 1) * 100

def build_scores(close):
    rows=[]
    if BENCHMARK not in close:
        return pd.DataFrame()
    b=close[BENCHMARK]
    for name,ticker in SECTORS.items():
        if ticker not in close:
            continue
        s=close[ticker]
        r4=perf(s,4)-perf(b,4)
        r13=perf(s,13)-perf(b,13)
        r26=perf(s,26)-perf(b,26)
        r52=perf(s,52)-perf(b,52)
        accel=r4-r13
        score=np.nanmean([r13,r26,r52])
        if np.isnan(score):
            label="N/A"
        elif score > 0 and r13 > 0 and r26 > 0:
            label="🟢 LEADER"
        elif r4 > 0 and accel > 0:
            label="🔵 IMPROVING"
        elif score > 0 and (r4 < 0 or accel < 0):
            label="🟠 WEAKENING"
        else:
            label="🔴 LAGGARD"
        rows.append({"Sector":name,"Ticker":ticker,"4W RS":r4,"13W RS":r13,"26W RS":r26,"52W RS":r52,"Acceleration":accel,"RS Score":score,"Status":label})
    return pd.DataFrame(rows).sort_values(["RS Score","Acceleration"],ascending=False)

c1,c2,c3=st.columns([1,1,2])
with c1:
    years=st.selectbox("Weekly history",[1,2,3,5],index=2,format_func=lambda x:f"{x} year" if x==1 else f"{x} years")
with c2:
    max_lines=st.slider("Sector lines", 3, len(SECTORS), 6, help="Choose how many sector lines may be shown together.")

try:
    close=load_weekly(years)
except Exception as exc:
    st.error(f"Weekly market data could not be loaded: {exc}")
    st.stop()

scores=build_scores(close)
if scores.empty:
    st.warning("Sector data is currently unavailable. Try refreshing after a few minutes.")
    st.stop()

leaders=scores[scores["Status"].str.contains("LEADER",na=False)].head(3)
improving=scores[scores["Status"].str.contains("IMPROVING",na=False)].sort_values("Acceleration",ascending=False).head(3)
defaults=pd.concat([leaders,improving]).drop_duplicates("Sector")["Sector"].tolist()[:max_lines]
if not defaults:
    defaults=scores.head(max_lines)["Sector"].tolist()

with c3:
    selected=st.multiselect("Compare sectors against NIFTY 50",scores["Sector"].tolist(),default=defaults,max_selections=len(SECTORS))

m1,m2,m3,m4=st.columns(4)
m1.metric("Benchmark","NIFTY 50")
m2.metric("Timeframe","Weekly")
m3.metric("Current Leader",scores.iloc[0]["Sector"])
imp=scores.sort_values("Acceleration",ascending=False).iloc[0]
m4.metric("Improving Fastest",imp["Sector"])

fig=go.Figure()
b=rebase(close[BENCHMARK])
# NIFTY 50 is the permanent benchmark: bright white and thicker so it stays visible on the dark dashboard.
fig.add_trace(go.Scatter(x=b.index,y=b,name="NIFTY 50",mode="lines",line={"width":4,"color":"#FFFFFF"},hovertemplate="NIFTY 50<br>%{x|%d %b %Y}<br>Rebased: %{y:.1f}<extra></extra>"))
for name in selected:
    ticker=SECTORS[name]
    if ticker not in close:
        continue
    s=rebase(close[ticker])
    status=scores.loc[scores["Sector"]==name,"Status"].iloc[0]
    fig.add_trace(go.Scatter(x=s.index,y=s,name=f"{name} · {status}",mode="lines",line={"width":2.5},hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>Rebased: %{{y:.1f}}<extra></extra>"))
fig.add_hline(y=100,line_dash="dot",line_color="rgba(100,100,100,.45)")
fig.update_layout(height=650,hovermode="x unified",legend={"orientation":"h","y":1.12,"x":0},margin=dict(l=15,r=15,t=65,b=15),xaxis_title="Weekly timeframe",yaxis_title="Performance rebased to 100",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"responsive":True})

st.info("How to read: a sector line rising faster than NIFTY 50 is outperforming. LEADER means sustained positive relative strength; IMPROVING means recent relative strength is accelerating and may be moving toward leadership. This is a ranking heuristic, not an investment recommendation.")

show=scores[["Sector","Status","4W RS","13W RS","26W RS","52W RS","Acceleration","RS Score"]].copy()
for col in ["4W RS","13W RS","26W RS","52W RS","Acceleration","RS Score"]:
    show[col]=show[col].round(2)
st.subheader("Sector Leadership Table")
st.dataframe(show,use_container_width=True,hide_index=True)

st.subheader("Scan stocks in a Leader / Improving sector")
choices=scores[scores["Status"].str.contains("LEADER|IMPROVING",regex=True,na=False)]["Sector"].tolist()
chosen=st.selectbox("Select sector",choices if choices else scores["Sector"].tolist())
st.caption(f"Selected: {chosen}. Use this sector as the priority universe in the Value / P-Factor / B-Factor workflow.")
