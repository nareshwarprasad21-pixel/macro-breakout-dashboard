from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Long Term + Swing Trading", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1500px;}
.hero-card {padding:22px 24px;border-radius:20px;border:1px solid rgba(120,120,120,.20);box-shadow:0 8px 24px rgba(0,0,0,.08);margin-bottom:14px;}
.hero-green {background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(34,197,94,.06));border-left:6px solid #16a34a;}
.hero-yellow {background:linear-gradient(135deg,rgba(250,204,21,.20),rgba(245,158,11,.06));border-left:6px solid #eab308;}
.hero-orange {background:linear-gradient(135deg,rgba(251,146,60,.20),rgba(249,115,22,.06));border-left:6px solid #f97316;}
.hero-red {background:linear-gradient(135deg,rgba(248,113,113,.20),rgba(220,38,38,.06));border-left:6px solid #dc2626;}
.regime-box {padding:14px 16px;border-radius:14px;border:1px solid rgba(120,120,120,.18);min-height:125px;}
.green-box {background:rgba(34,197,94,.10);}
.yellow-box {background:rgba(234,179,8,.12);}
.orange-box {background:rgba(249,115,22,.12);}
.red-box {background:rgba(220,38,38,.10);}
.support-box {background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.28);padding:16px;border-radius:14px;}
.negative-box {background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.28);padding:16px;border-radius:14px;}
.confirm-box {background:rgba(234,179,8,.09);border:1px solid rgba(234,179,8,.30);padding:16px;border-radius:14px;}
.flow-card {padding:15px 12px;border-radius:14px;border:1px solid rgba(120,120,120,.18);text-align:center;min-height:110px;background:rgba(99,102,241,.06);}
.small-note {font-size:.88rem;opacity:.78;}
@media(max-width:900px){.block-container{padding:1rem .7rem 1.5rem!important}.hero-card{padding:16px}.regime-box{min-height:auto}}
</style>
""",
    unsafe_allow_html=True,
)

MACRO_ASSETS = {
    "NIFTY 50": "^NSEI",
    "India VIX": "^INDIAVIX",
    "USD/INR": "INR=X",
    "Brent Crude": "BZ=F",
    "Dollar Index": "DX-Y.NYB",
    "US 10Y Yield": "^TNX",
    "Gold": "GC=F",
}


def safe_float(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def pct_return(c, periods):
    c = pd.to_numeric(c, errors="coerce").dropna()
    if len(c) <= periods:
        return np.nan
    base = safe_float(c.iloc[-periods - 1])
    last = safe_float(c.iloc[-1])
    return (last / base - 1) * 100 if pd.notna(base) and base != 0 else np.nan


@st.cache_data(ttl=1200, show_spinner=False)
def macro_snapshot():
    tickers = list(MACRO_ASSETS.values())
    try:
        raw = yf.download(
            tickers=tickers,
            period="2y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
            timeout=25,
        )
    except TypeError:
        raw = yf.download(tickers=tickers, period="2y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except Exception:
        raw = pd.DataFrame()

    rows = []
    for name, ticker in MACRO_ASSETS.items():
        try:
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                d = raw[ticker].copy()
            elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(-1):
                d = raw.xs(ticker, axis=1, level=-1).copy()
            else:
                continue
            c = pd.to_numeric(d["Close"], errors="coerce").dropna()
            if len(c) < 25:
                continue
            rows.append({
                "Indicator": name,
                "Ticker": ticker,
                "Latest": safe_float(c.iloc[-1]),
                "1M %": pct_return(c, 21),
                "3M %": pct_return(c, 63),
                "12M %": pct_return(c, 252),
                "As Of": pd.to_datetime(c.index[-1]).strftime("%Y-%m-%d"),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def long_term_macro_engine(mdf):
    """Positional/long-term macro filter. 3M and 12M matter more than 1M."""
    specs = [
        ("NIFTY 50", +1, 2.2, 10.0, "Domestic equity trend"),
        ("India VIX", -1, 1.6, 18.0, "Volatility / fear"),
        ("USD/INR", -1, 1.5, 6.0, "Rupee / imported inflation pressure"),
        ("Brent Crude", -1, 1.5, 18.0, "India import-cost pressure"),
        ("Dollar Index", -1, 1.0, 8.0, "Global dollar liquidity"),
        ("US 10Y Yield", -1, 1.0, 10.0, "Global cost of money"),
        ("Gold", -1, 0.6, 18.0, "Defensive / uncertainty proxy"),
    ]
    if mdf is None or mdf.empty:
        return 5.0, "🟡 MIXED / SELECTIVE", 0, pd.DataFrame(), "Low data"

    total = sum(x[2] for x in specs)
    weighted = 0.0
    used = 0.0
    rows = []

    for name, direction, weight, scale, meaning in specs:
        r = mdf[mdf["Indicator"] == name]
        if r.empty:
            continue
        r1 = safe_float(r.iloc[0]["1M %"])
        r3 = safe_float(r.iloc[0]["3M %"])
        r12 = safe_float(r.iloc[0]["12M %"])
        parts = []
        if pd.notna(r1): parts.append((r1, 0.15))
        if pd.notna(r3): parts.append((r3, 0.40))
        if pd.notna(r12): parts.append((r12, 0.45))
        if not parts:
            continue
        horizon_move = sum(v * w for v, w in parts) / sum(w for _, w in parts)
        raw = float(np.tanh(direction * horizon_move / scale))
        weighted += raw * weight
        used += weight
        signal = "🟢 Supportive" if raw > 0.15 else "🔴 Negative" if raw < -0.15 else "🟡 Neutral/Mixed"
        rows.append({
            "Driver": name,
            "Meaning": meaning,
            "1M %": r1,
            "3M %": r3,
            "12M %": r12,
            "Contribution": raw * weight,
            "Signal": signal,
        })

    if used == 0:
        return 5.0, "🟡 MIXED / SELECTIVE", 0, pd.DataFrame(rows), "No evaluable inputs"

    score = float(np.clip(5 + 5 * weighted / used, 0, 10))
    coverage = int(round(100 * used / total))
    if score >= 7.5:
        regime = "🟢 SUPPORTIVE / RISK-ON"
    elif score >= 5.0:
        regime = "🟡 MIXED / SELECTIVE"
    elif score >= 3.0:
        regime = "🟠 CAUTIOUS"
    else:
        regime = "🔴 RISK-OFF"
    return score, regime, coverage, pd.DataFrame(rows).sort_values("Contribution", ascending=False), f"{coverage}% weighted input coverage"


def regime_strategy(score):
    if score >= 7.5:
        return "🟢", "SUPPORTIVE / RISK-ON", "Look for more opportunities; still demand sector leadership, quality and valid breakout structure.", "hero-green"
    if score >= 5.0:
        return "🟡", "MIXED / SELECTIVE", "Focus only on sector leaders + strong fundamentals + confirmed breakouts. Avoid broad-market buying.", "hero-yellow"
    if score >= 3.0:
        return "🟠", "CAUTIOUS", "Reduce position size, raise confirmation quality and remain highly selective.", "hero-orange"
    return "🔴", "RISK-OFF", "Avoid aggressive fresh buying; capital protection and only exceptional setups get attention.", "hero-red"


st.title("🧭 Long-Term / Positional + Swing Trading Lab")
st.caption("Long-term macro regime is a background filter. Sector leadership and stock technicals determine selection and timing.")

macro = macro_snapshot()
score, regime, coverage, drivers, note = long_term_macro_engine(macro)
icon, regime_name, strategy, hero_class = regime_strategy(score)

long_tab, swing_tab = st.tabs(["🌐 Long-Term / Positional", "⚡ Swing Trading Roadmap"])

with long_tab:
    st.markdown(
        f"""
<div class="hero-card {hero_class}">
  <div style="font-size:.92rem;font-weight:700;letter-spacing:.06em;opacity:.75">CURRENT MACRO REGIME</div>
  <div style="font-size:2rem;font-weight:800;margin:4px 0 2px">{icon} {regime_name}</div>
  <div style="font-size:1.18rem;font-weight:700">Macro Score: {score:.1f} / 10 &nbsp; | &nbsp; Coverage: {coverage}%</div>
  <div style="margin-top:10px;font-size:1.02rem"><b>Recommended Strategy:</b> {strategy}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### 🎚️ Permanent Macro Regime Scale")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="regime-box green-box"><b>🟢 7.5–10<br>SUPPORTIVE / RISK-ON</b><br><br>Macro equities के favour में.<br><b>Action:</b> ज्यादा opportunities खोजें.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="regime-box yellow-box"><b>🟡 5.0–7.49<br>MIXED / SELECTIVE</b><br><br>Positive + negative signals दोनों.<br><b>Action:</b> leaders + quality breakouts.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="regime-box orange-box"><b>🟠 3.0–4.99<br>CAUTIOUS</b><br><br>Macro headwinds बढ़ रहे हैं.<br><b>Action:</b> smaller size + highly selective.</div>', unsafe_allow_html=True)
    c4.markdown('<div class="regime-box red-box"><b>🔴 0–2.99<br>RISK-OFF</b><br><br>Macro environment adverse.<br><b>Action:</b> aggressive fresh buying avoid.</div>', unsafe_allow_html=True)

    st.markdown("### 🔍 Why this regime?")
    supportive = drivers[drivers["Signal"].str.contains("Supportive", na=False)] if not drivers.empty else pd.DataFrame()
    negative = drivers[drivers["Signal"].str.contains("Negative", na=False)] if not drivers.empty else pd.DataFrame()
    neutral = drivers[drivers["Signal"].str.contains("Neutral", na=False)] if not drivers.empty else pd.DataFrame()

    a, b, c = st.columns(3)
    with a:
        items = "".join([f"<li><b>{r['Driver']}</b> — {r['Meaning']}</li>" for _, r in supportive.iterrows()]) or "<li>No strong supportive driver</li>"
        st.markdown(f'<div class="support-box"><b>🟢 SUPPORTIVE DRIVERS</b><ul>{items}</ul></div>', unsafe_allow_html=True)
    with b:
        items = "".join([f"<li><b>{r['Driver']}</b> — {r['Meaning']}</li>" for _, r in negative.iterrows()]) or "<li>No strong negative driver</li>"
        st.markdown(f'<div class="negative-box"><b>🔴 NEGATIVE DRIVERS</b><ul>{items}</ul></div>', unsafe_allow_html=True)
    with c:
        if macro.empty:
            conf = "Live macro feed unavailable"
        else:
            n = macro[macro["Indicator"] == "NIFTY 50"]
            if not n.empty:
                conf = f"Nifty: 1M {safe_float(n.iloc[0]['1M %']):.2f}% · 3M {safe_float(n.iloc[0]['3M %']):.2f}% · 12M {safe_float(n.iloc[0]['12M %']):.2f}%"
            else:
                conf = "Nifty confirmation unavailable"
        st.markdown(f'<div class="confirm-box"><b>🟡 MARKET CONFIRMATION</b><br><br>{conf}<br><br>{note}</div>', unsafe_allow_html=True)

    st.markdown("### 🧠 Positional Decision Chain")
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    steps = [
        (f1, "🌐", "MACRO REGIME", "Background filter"),
        (f2, "🔄", "SECTOR ROTATION", "Where money is moving"),
        (f3, "🥇", "WEEKLY LEADERS", "Find strongest sectors"),
        (f4, "📈", "26M ATH", "Major breakout confirmation"),
        (f5, "🧾", "FUNDAMENTALS", "Quality confirmation"),
        (f6, "⭐", "FINAL OPPORTUNITY", "Rank + decision"),
    ]
    for col, em, title, desc in steps:
        col.markdown(f'<div class="flow-card"><div style="font-size:1.5rem">{em}</div><b>{title}</b><br><span class="small-note">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("### 📊 Macro Regime Diagnostics")
    if macro.empty:
        st.warning("Macro data feed unavailable right now.")
    else:
        st.dataframe(macro, use_container_width=True, hide_index=True)
        st.markdown("#### Driver Attribution")
        st.dataframe(drivers, use_container_width=True, hide_index=True)

    st.info("Long-term rule: 1M move gets low weight; 3M and 12M trends get higher weight. Macro is a filter, not a direct buy/sell signal.")

with swing_tab:
    st.markdown("### ⚡ Swing Trading Roadmap")
    st.caption("For swing trading, macro gets lower weight than positional trading. Sector leadership + stock relative strength + technical setup drive the trade.")

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    swing_steps = [
        (s1, "🌤️", "MARKET CONDITION", "Nifty daily/weekly trend"),
        (s2, "🥇", "SECTOR LEADER", "Weekly outperformance vs Nifty"),
        (s3, "💪", "STRONG STOCK", "Stock RS vs sector + Nifty"),
        (s4, "📉", "TECHNICAL SETUP", "Breakout / pullback / HH-HL"),
        (s5, "🛑", "ENTRY + SL", "Structure-based risk"),
        (s6, "🎯", "TARGET / TRAIL", "R:R + trailing stop"),
    ]
    for col, em, title, desc in swing_steps:
        col.markdown(f'<div class="flow-card"><div style="font-size:1.5rem">{em}</div><b>{title}</b><br><span class="small-note">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Swing Trade Interpretation")
    left, right = st.columns(2)
    with left:
        st.success("🟢 **Preferred setup**\n\nMacro Supportive/Mixed + Weekly Sector Leader + Strong Stock RS + Daily technical breakout/pullback = High-quality swing candidate.")
        st.warning("🟡 **Macro Mixed** does NOT automatically reject a swing trade. If sector and stock are strong, the setup can still qualify with disciplined risk.")
    with right:
        st.error("🔴 **Avoid / downgrade**\n\nRisk-Off macro + weak sector + isolated stock breakout = higher failure risk. Demand stronger confirmation or skip.")
        st.info("Next implementation phase: automatic sector-leader → strong-stock scanner → daily setup score → Entry / SL / Target panel.")

    st.markdown("### 🧩 Planned Swing Score")
    swing_score = pd.DataFrame([
        ["Weekly Sector Leadership", "20%", "Leader / Improving sector"],
        ["Stock Relative Strength", "20%", "Outperform sector + Nifty"],
        ["Daily Trend", "15%", "Price above key averages / HH-HL"],
        ["Breakout / Pullback Quality", "20%", "Clean level + confirmation"],
        ["Volume Confirmation", "10%", "Participation confirms move"],
        ["Market / Macro Condition", "10%", "Background risk filter"],
        ["Risk : Reward", "5%", "Trade structure quality"],
    ], columns=["Component", "Weight", "What it checks"])
    st.dataframe(swing_score, use_container_width=True, hide_index=True)

st.divider()
st.caption(f"Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Research heuristics only; verify prices and filings before execution.")
