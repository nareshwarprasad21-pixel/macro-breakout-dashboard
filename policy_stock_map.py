import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Curated representative NSE beneficiaries by long-run policy theme.
# These are research mappings, not recommendations or guaranteed beneficiaries.
POLICY_STOCK_MAP = {
    "Power Transmission, Grid & Energy Storage": [
        ("POWERGRID", "Transmission utility", 96), ("HITACHIENER", "T&D equipment", 95),
        ("GEVERNOVA", "T&D equipment", 94), ("CGPOWER", "Electrical equipment", 91),
        ("APARINDS", "Conductors / cables", 93), ("KEC", "Transmission EPC", 89),
        ("KPIL", "Transmission EPC", 88), ("POLYCAB", "Cables", 87),
        ("KEI", "Cables", 85), ("TATAPOWER", "Power + storage", 84),
    ],
    "Renewable Energy – Solar, Wind & Hybrid": [
        ("WAAREEENER", "Solar modules", 95), ("PREMIERENE", "Solar modules", 92),
        ("SUZLON", "Wind turbines", 91), ("INOXWIND", "Wind turbines", 88),
        ("TATAPOWER", "Renewable developer", 89), ("JSWENERGY", "Renewable + storage", 88),
        ("NTPC", "Renewable developer", 84), ("POWERGRID", "Grid enabler", 82),
        ("BORORENEW", "Solar glass", 80), ("STERLINGG", "Solar EPC", 78),
    ],
    "Defence, Aerospace, Drones & Electronics": [
        ("HAL", "Aerospace platforms", 97), ("BEL", "Defence electronics", 96),
        ("BDL", "Missiles", 93), ("MAZDOCK", "Naval shipbuilding", 92),
        ("COCHINSHIP", "Shipbuilding", 88), ("GRSE", "Naval shipbuilding", 87),
        ("DATAPATTNS", "Defence electronics", 86), ("PARAS", "Defence components", 82),
        ("SOLARINDS", "Defence explosives", 84), ("ASTRAMICRO", "Defence electronics", 80),
    ],
    "Electronics, Components & Semiconductors": [
        ("DIXON", "EMS", 95), ("KAYNES", "EMS / semiconductor", 94),
        ("SYRMA", "EMS", 90), ("AMBER", "Electronics manufacturing", 87),
        ("PGEL", "Electronics manufacturing", 85), ("BEL", "Industrial/defence electronics", 81),
        ("NETWEB", "Compute hardware", 83), ("MOSCHIP", "Semiconductor design", 78),
        ("CENTUM", "Electronics systems", 77), ("AVALON", "EMS", 76),
    ],
    "Railways, High-Speed Rail & Freight": [
        ("RVNL", "Rail EPC", 94), ("IRCON", "Rail EPC", 91), ("RAILTEL", "Rail digital/signalling", 88),
        ("TITAGARH", "Rolling stock", 90), ("JWL", "Wagons / mobility", 86),
        ("BEML", "Rolling stock", 85), ("KERNEX", "Rail safety/signalling", 76),
        ("HBLENGINE", "Rail safety electronics", 79), ("LT", "Large EPC", 80), ("KEC", "Rail EPC", 78),
    ],
    "Green Hydrogen & Green Ammonia": [
        ("LT", "Engineering / electrolyser ecosystem", 87), ("RELIANCE", "Integrated clean energy", 86),
        ("NTPC", "Green hydrogen developer", 84), ("IOC", "Refining + hydrogen projects", 82),
        ("GAIL", "Gas / hydrogen infrastructure", 80), ("TATAPOWER", "Renewable power", 78),
        ("JSWENERGY", "Renewable power", 78), ("INOXINDIA", "Cryogenic equipment", 81),
        ("THERMAX", "Industrial clean-energy equipment", 79), ("PRAJIND", "Process engineering", 77),
    ],
    "Roads, Expressways & Multimodal Logistics": [
        ("LT", "Infrastructure EPC", 93), ("KNRCON", "Road EPC", 87), ("PNCINFRA", "Road EPC", 84),
        ("HGIEL", "Road EPC", 82), ("GRINFRA", "Road EPC", 83), ("ASHOKA", "Road EPC / HAM", 78),
        ("ULTRACEMCO", "Cement", 80), ("AMBUJACEM", "Cement", 78),
        ("CONCOR", "Multimodal logistics", 86), ("ADANIPORTS", "Ports / logistics", 88),
    ],
    "EVs, Charging & Battery Ecosystem": [
        ("TATAMOTORS", "EV OEM", 91), ("M&M", "EV OEM", 88), ("SONACOMS", "EV driveline/components", 92),
        ("MOTHERSON", "Auto components", 82), ("EXIDEIND", "Battery", 87), ("ARE&M", "Battery", 84),
        ("OLECTRA", "Electric buses", 84), ("ABB", "Charging / power electronics", 82),
        ("CGPOWER", "Motors / electrical", 80), ("TATAPOWER", "EV charging", 83),
    ],
    "AI, Cloud, Data Centres & Digital Infrastructure": [
        ("NETWEB", "AI/HPC hardware", 92), ("ANANTRAJ", "Data centres", 87),
        ("TECHM", "Cloud / AI services", 78), ("TCS", "AI / cloud services", 80),
        ("INFY", "AI / cloud services", 80), ("ABB", "Data-centre electrical", 83),
        ("SIEMENS", "Data-centre electrical", 82), ("CUMMINSIND", "Backup power", 80),
        ("BLUESTARCO", "Cooling", 84), ("VOLTAS", "Cooling", 78),
    ],
    "Pharma APIs, Complex Drugs & Medical Devices": [
        ("DIVISLAB", "APIs / custom synthesis", 91), ("LAURUSLABS", "APIs / CDMO", 86),
        ("NEULANDLAB", "APIs", 90), ("SYNGENE", "CRDMO", 89), ("PIRAMALPH", "CDMO", 82),
        ("SUNPHARMA", "Complex pharma", 84), ("DRREDDY", "Complex generics", 83),
        ("POLYMED", "Medical devices", 88), ("MAXHEALTH", "Healthcare", 76), ("LALPATHLAB", "Diagnostics", 74),
    ],
}

@st.cache_data(ttl=1800, show_spinner=False)
def _live_policy_stock_data(symbols):
    tickers = [s + ".NS" for s in symbols]
    out = {}
    try:
        raw = yf.download(tickers, period="8mo", interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
    except Exception:
        raw = pd.DataFrame()
    for s, t in zip(symbols, tickers):
        try:
            if raw.empty:
                continue
            if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
                px = raw["Close"]
            else:
                px = raw[t]["Close"]
            px = pd.to_numeric(px, errors="coerce").dropna()
            if len(px) < 25:
                continue
            last = float(px.iloc[-1])
            r1 = (last / float(px.iloc[-22]) - 1) * 100 if len(px) > 22 else np.nan
            r3 = (last / float(px.iloc[-64]) - 1) * 100 if len(px) > 64 else np.nan
            out[s] = {"CMP": last, "1M %": r1, "3M %": r3}
        except Exception:
            continue
    return out


def _policy_status(policy_fit, r3, scan_status, fund_score):
    score = policy_fit
    if pd.notna(r3):
        score += 4 if r3 > 10 else 2 if r3 > 0 else -3
    if isinstance(scan_status, str):
        if scan_status.startswith("Fresh") or scan_status.startswith("Breakout"):
            score += 4
        elif "Near" in scan_status:
            score += 1
    if pd.notna(fund_score):
        score += 3 if fund_score >= 70 else 1 if fund_score >= 50 else -2
    if score >= 96:
        return "STRONG POLICY CANDIDATE"
    if score >= 89:
        return "POLICY BENEFICIARY / WATCH"
    if score >= 82:
        return "EMERGING CANDIDATE"
    return "RESEARCH / WAIT"


def render_policy_stock_mapping(policy_df):
    st.markdown("#### Policy Theme → Listed Stock Opportunity Map")
    st.caption(
        "Select a 10-year policy theme to see representative listed beneficiaries on the same screen. "
        "Policy Fit is a curated business-exposure heuristic; live momentum, 26M status and fundamentals are confirmation layers."
    )

    themes = [t for t in policy_df["Sector / Theme"].tolist() if t in POLICY_STOCK_MAP]
    selected = st.selectbox("Choose policy theme", themes, key="policy_stock_theme")
    theme_row = policy_df.loc[policy_df["Sector / Theme"] == selected].iloc[0]
    mapped = POLICY_STOCK_MAP.get(selected, [])
    symbols = [x[0] for x in mapped]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Policy Theme Score", f"{theme_row['10Y Opportunity Score']:.1f}/10")
    c2.metric("Policy Strength", f"{theme_row['Policy Strength']:.1f}/10")
    c3.metric("Mapped Stocks", len(mapped))
    c4.metric("Horizon", str(theme_row["Horizon"]))

    if st.button("Refresh mapped stock market data", key="refresh_policy_stocks", use_container_width=True):
        _live_policy_stock_data.clear()
        st.rerun()

    live = _live_policy_stock_data(symbols)
    latest_ranked = st.session_state.get("latest_ranked", pd.DataFrame())
    auto_funds = st.session_state.get("auto_fund_scores", {})

    rows = []
    for symbol, industry, fit in mapped:
        d = live.get(symbol, {})
        scan_status = "Not scanned"
        sector_score = np.nan
        pro_score = np.nan
        if isinstance(latest_ranked, pd.DataFrame) and not latest_ranked.empty and "Symbol" in latest_ranked.columns:
            hit = latest_ranked[latest_ranked["Symbol"] == symbol]
            if not hit.empty:
                rr = hit.iloc[0]
                scan_status = rr.get("Status", "Not scanned")
                sector_score = rr.get("Sector Score", np.nan)
                pro_score = rr.get("Pro Final Score", np.nan)
        fund = auto_funds.get(symbol, np.nan)
        rows.append({
            "Stock": symbol,
            "Industry / Role": industry,
            "Policy Fit %": fit,
            "CMP": d.get("CMP", np.nan),
            "1M %": d.get("1M %", np.nan),
            "3M %": d.get("3M %", np.nan),
            "26M Technical": scan_status,
            "Sector Score": sector_score,
            "Fundamental %": fund,
            "Pro Final Score": pro_score,
            "Research Status": _policy_status(fit, d.get("3M %", np.nan), scan_status, fund),
        })

    df = pd.DataFrame(rows)
    status_order = {"STRONG POLICY CANDIDATE":0, "POLICY BENEFICIARY / WATCH":1, "EMERGING CANDIDATE":2, "RESEARCH / WAIT":3}
    df["_order"] = df["Research Status"].map(status_order).fillna(9)
    df = df.sort_values(["_order", "Policy Fit %", "3M %"], ascending=[True, False, False]).drop(columns="_order")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Policy Fit %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "CMP": st.column_config.NumberColumn(format="₹%.2f"),
            "1M %": st.column_config.NumberColumn(format="%+.2f%%"),
            "3M %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Sector Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
            "Fundamental %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "Pro Final Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
        },
    )

    chosen = st.selectbox("Select mapped stock for next action", df["Stock"].tolist(), key="policy_stock_action")
    a, b = st.columns(2)
    if a.button("Send to research workflow", key="send_policy_stock", use_container_width=True):
        st.session_state["policy_selected_symbol"] = chosen
        st.success(f"{chosen} selected. Use it in Stock Detail / Professional Research Lab for detailed analysis.")
    if b.button("Show policy rationale", key="policy_rationale", use_container_width=True):
        st.info(
            f"{chosen} is mapped to **{selected}** because its business role falls under "
            f"**{df.loc[df['Stock']==chosen, 'Industry / Role'].iloc[0]}**. "
            "This mapping identifies policy exposure; valuation, execution, fundamentals and technical confirmation still matter."
        )

    st.markdown("**How to use this map:** Policy theme → beneficiary role → live momentum → sector strength → 26M technical confirmation → fundamentals → final research decision.")
    st.warning("Mapped stocks are representative research candidates, not buy recommendations. A policy beneficiary can still underperform because of valuation, execution, competition or balance-sheet risk.")
