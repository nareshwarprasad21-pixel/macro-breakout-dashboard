import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Weekly Sector Leaders", page_icon="📈", layout="wide")

st.title("📈 Weekly Sector Leaders vs NIFTY 50")
st.caption("TradingView-style weekly percentage comparison. Every line starts at 0% on the same visible starting date. NIFTY Defence Basket is a free equal-weight basket of current Defence constituents, not the official licensed index.")

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
    "NIFTY Defence Basket": "DEFENCE_BASKET",
    "NIFTY India Digital": "NIFTY_IND_DIGITAL.NS",
    "NIFTY India Manufacturing": "NIFTY_INDIA_MFG.NS",
}
BENCHMARK = "^NSEI"
DEFENCE_TICKER = "DEFENCE_BASKET"
DEFENCE_CONSTITUENTS = [
    "HAL.NS", "BEL.NS", "SOLARINDS.NS", "BHARATFORG.NS", "MAZDOCK.NS",
    "BDL.NS", "COCHINSHIP.NS", "GRSE.NS", "DATAPATTNS.NS", "MTARTECH.NS",
    "BEML.NS", "ASTRAMICRO.NS", "PARAS.NS", "ZENTEC.NS", "DCXINDIA.NS",
    "DYNAMATECH.NS", "MIDHANI.NS", "IDEAFORGE.NS", "CYIENTDLM.NS",
]

def load_defence_basket(period):
    """Build a free equal-weight Defence basket from current constituents."""
    raw = yf.download(
        DEFENCE_CONSTITUENTS,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        return pd.Series(dtype=float, name=DEFENCE_TICKER)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(how="all")
    returns = prices.pct_change(fill_method=None)
    # Require at least half the basket to have a price on a date.
    minimum = max(3, len(DEFENCE_CONSTITUENTS) // 2)
    basket_returns = returns.mean(axis=1, skipna=True).where(returns.notna().sum(axis=1) >= minimum)
    first_valid = basket_returns.first_valid_index()
    if first_valid is None:
        return pd.Series(dtype=float, name=DEFENCE_TICKER)
    basket = (1.0 + basket_returns.loc[first_valid:].fillna(0.0)).cumprod() * 100.0
    basket.name = DEFENCE_TICKER
    return basket

def load_official_yahoo_defence(years):
    """Read the official index ticker through Yahoo's raw chart endpoint."""
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.DateOffset(years=max(2, years + 1))
    response = requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/NIFTY_IND_DEFENCE.NS",
        params={
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": "1d",
            "events": "history",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    result = response.json().get("chart", {}).get("result") or []
    if not result:
        return pd.Series(dtype=float, name=DEFENCE_TICKER)
    item = result[0]
    timestamps = item.get("timestamp") or []
    quotes = (item.get("indicators", {}).get("quote") or [{}])[0]
    closes = quotes.get("close") or []
    if not timestamps or len(timestamps) != len(closes):
        return pd.Series(dtype=float, name=DEFENCE_TICKER)
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
    series = pd.Series(pd.to_numeric(closes, errors="coerce"), index=dates, name=DEFENCE_TICKER)
    return series.dropna().sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]

def load_official_nse_defence(years):
    """Fetch official NIFTY India Defence daily closes from NSE India."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/136 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
    }
    session = requests.Session()
    session.get("https://www.nseindia.com/", headers=headers, timeout=15)

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=max(2, years + 1))
    records = []
    cursor = start

    # NSE's historical endpoint is most reliable with roughly one-year chunks.
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        response = session.get(
            "https://www.nseindia.com/api/historical/indicesHistory",
            params={
                "indexType": DEFENCE_NSE_NAME,
                "from": cursor.strftime("%d-%m-%Y"),
                "to": chunk_end.strftime("%d-%m-%Y"),
            },
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        records.extend(payload.get("data", {}).get("indexCloseOnlineRecords", []))
        cursor = chunk_end + pd.Timedelta(days=1)

    if not records:
        return pd.Series(dtype=float, name=DEFENCE_TICKER)

    frame = pd.DataFrame(records)
    if "EOD_TIMESTAMP" not in frame or "EOD_CLOSE_INDEX_VAL" not in frame:
        return pd.Series(dtype=float, name=DEFENCE_TICKER)

    dates = pd.to_datetime(frame["EOD_TIMESTAMP"], errors="coerce", dayfirst=True)
    values = pd.to_numeric(
        frame["EOD_CLOSE_INDEX_VAL"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    series = pd.Series(values.to_numpy(), index=dates, name=DEFENCE_TICKER)
    return series[~series.index.isna()].dropna().sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]

@st.cache_data(ttl=1200, show_spinner=False)
def load_weekly(years):
    """Download daily prices first, then build Friday weekly closes.

    Yahoo does not consistently return interval="1wk" history for newer Indian
    indices (notably NIFTY India Defence). Daily data is more complete and the
    local resample keeps every sector on the same weekly calendar.
    """
    tickers = [BENCHMARK] + [ticker for ticker in SECTORS.values() if ticker != DEFENCE_TICKER]
    period = {0.5:"1y", 1:"2y", 2:"3y", 3:"5y", 5:"10y"}.get(years, "5y")
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

    # yfinance can omit a valid NSE index from a large multi-ticker request.
    # Retry every missing ticker individually so official indices such as
    # NIFTY India Defence remain official index data (no ETF proxy).
    missing = [ticker for ticker in tickers if ticker not in close.columns or close[ticker].dropna().empty]
    for ticker in missing:
        try:
            one = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if one.empty:
                continue
            if isinstance(one.columns, pd.MultiIndex):
                one_close = one["Close"]
                one_close = one_close.iloc[:, 0] if isinstance(one_close, pd.DataFrame) else one_close
            else:
                one_close = one["Close"]
            one_close.name = ticker
            if ticker in close.columns:
                close[ticker] = one_close
            else:
                close = close.join(one_close, how="outer")
        except Exception:
            continue

    # The official Yahoo quote exists but its history is often empty. Use the
    # official NSE index-history endpoint—not an ETF—when that happens.
    # Free, transparent Defence proxy: equal-weight return basket of current
    # constituent stocks. This enables line chart and 4W/13W/26W/52W RS.
    try:
        defence = load_defence_basket(period)
        if not defence.empty:
            close = close.join(defence, how="outer")
    except Exception:
        pass

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.dropna(how="all").resample("W-FRI").last().dropna(how="all")
    return close.tail(max(55, int(years * 53) + 4))

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
    years=st.selectbox("Weekly history",[0.5,1,2,3,5],index=1,format_func=lambda x:"6 months" if x==0.5 else (f"{int(x)} year" if x==1 else f"{int(x)} years"))
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
# Keep at least 52 weeks loaded for RS scores, but show only the selected visual history.
chart_weeks=max(26,int(years*53))
chart_close=close.tail(chart_weeks+1)
chart_items=[("NIFTY 50",BENCHMARK)]+[(name,SECTORS[name]) for name in selected if SECTORS[name] in chart_close]
valid_starts=[chart_close[ticker].dropna().index.min() for _,ticker in chart_items if ticker in chart_close and not chart_close[ticker].dropna().empty]
common_start=max(valid_starts) if valid_starts else chart_close.index.min()
colors=["#FFFFFF","#22D3EE","#22C55E","#F59E0B","#3B82F6","#A855F7","#EF4444","#EC4899","#14B8A6","#F97316"]

for i,(name,ticker) in enumerate(chart_items):
    if ticker not in chart_close:
        continue
    s=chart_close.loc[common_start:,ticker].dropna()
    if len(s)<2:
        continue
    pct=(s/s.iloc[0]-1.0)*100.0
    color=colors[i%len(colors)]
    width=3.5 if ticker==BENCHMARK else 2.5
    current=pct.iloc[-1]
    label=f"{name}  {current:+.1f}%"
    fig.add_trace(go.Scatter(
        x=pct.index,y=pct,name=name,mode="lines",
        line={"width":width,"color":color},
        hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>Change: %{{y:+.2f}}%<extra></extra>",
    ))
    fig.add_annotation(
        x=pct.index[-1],y=current,text=label,showarrow=False,
        xanchor="left",xshift=8,font={"size":11,"color":color},
        bgcolor="rgba(6,16,24,.86)",bordercolor=color,borderwidth=1,borderpad=3,
    )

fig.add_hline(y=0,line_dash="dot",line_color="rgba(180,190,200,.55)",line_width=1)
fig.update_layout(
    height=650,hovermode="x unified",
    legend={"orientation":"h","y":1.12,"x":0},
    margin=dict(l=15,r=175,t=65,b=15),
    xaxis_title="Weekly timeframe · last available close",
    yaxis_title="Change from common start (%)",
    yaxis={"ticksuffix":"%","zeroline":False},
    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
)
if valid_starts:
    st.caption(f"Chart comparison period: {common_start:%d %b %Y} to {chart_close.index.max():%d %b %Y} · All lines start at 0%.")
st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"responsive":True})

st.info("How to read: every line starts at 0% on the same date. A sector ending above NIFTY 50 has outperformed over the selected period. The 4W/13W/26W/52W RS calculations below remain independent of the visual chart scale. This is a ranking heuristic, not an investment recommendation.")

show=scores[["Sector","Status","4W RS","13W RS","26W RS","52W RS","Acceleration","RS Score"]].copy()
for col in ["4W RS","13W RS","26W RS","52W RS","Acceleration","RS Score"]:
    show[col]=show[col].round(2)
st.subheader("Sector Leadership Table")
st.dataframe(show,use_container_width=True,hide_index=True)

st.subheader("Scan stocks in a Leader / Improving sector")
choices=scores[scores["Status"].str.contains("LEADER|IMPROVING",regex=True,na=False)]["Sector"].tolist()
chosen=st.selectbox("Select sector",choices if choices else scores["Sector"].tolist())
st.caption(f"Selected: {chosen}. Use this sector as the priority universe in the Value / P-Factor / B-Factor workflow.")
