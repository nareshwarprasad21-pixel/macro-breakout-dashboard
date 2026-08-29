import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from ui_polish import apply_professional_ui

st.set_page_config(page_title="Value Migration Engine", page_icon="🚀", layout="wide")
apply_professional_ui()

NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]

@st.cache_data(ttl=86400, show_spinner=False)
def load_nifty500():
    last_err = None
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in NIFTY500_URLS:
        try:
            r = requests.get(url, timeout=15, headers=headers)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # Common columns: Company Name, Industry, Symbol, Series, ISIN Code
            if "Symbol" not in df.columns:
                continue
            df["Ticker"] = df["Symbol"].astype(str).str.strip() + ".NS"
            if "Industry" not in df.columns:
                df["Industry"] = "Unknown"
            if "Company Name" not in df.columns:
                df["Company Name"] = df["Symbol"]
            return df[["Company Name", "Industry", "Symbol", "Ticker"]].drop_duplicates("Ticker")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"NIFTY 500 constituent list fetch failed: {last_err}")

VALUE_MIGRATION_BASKETS = {
    "Power Grid, Transformers & Transmission": [
        "POWERGRID.NS","ABB.NS","SIEMENS.NS","CGPOWER.NS","APARINDS.NS",
        "HITACHIENER.NS","GEVERNOVA.NS","KEC.NS","KPIL.NS","POLYCAB.NS"
    ],
    "AI Data Centre Infrastructure": [
        "NETWEB.NS","ANANTRAJ.NS","ABB.NS","SIEMENS.NS","CUMMINSIND.NS",
        "BLUESTARCO.NS","VOLTAS.NS","POLYCAB.NS","KEI.NS","TECHM.NS"
    ],
    "Battery Energy Storage (BESS) & Power Electronics": [
        "TATAPOWER.NS","JSWENERGY.NS","EXIDEIND.NS","AMARAJABAT.NS",
        "WAAREEENER.NS","ABB.NS","SIEMENS.NS","CGPOWER.NS"
    ],
    "Electronics Components / EMS / Semiconductor Ecosystem": [
        "DIXON.NS","KAYNES.NS","SYRMA.NS","AMBER.NS","PGEL.NS",
        "BEL.NS","NETWEB.NS","MOSCHIP.NS"
    ],
    "Defence Indigenisation & Component Suppliers": [
        "HAL.NS","BEL.NS","BDL.NS","MAZDOCK.NS","COCHINSHIP.NS",
        "GRSE.NS","DATAPATTNS.NS","PARAS.NS"
    ],
    "Grain / Flexible-feed Ethanol": [
        "BALRAMCHIN.NS","TRIVENI.NS","GLOBUSSPR.NS","RENUKA.NS",
        "EIDPARRY.NS","BAJAJHIND.NS"
    ],
}

VALUE_MIGRATION_ROLES = {
    "Power Grid, Transformers & Transmission": {
        "POWERGRID.NS": "Power transmission utility", "ABB.NS": "Grid automation & switchgear",
        "SIEMENS.NS": "Electrification & grid automation", "CGPOWER.NS": "Transformers & switchgear",
        "APARINDS.NS": "Conductors & power cables", "HITACHIENER.NS": "Grid equipment & transformers",
        "GEVERNOVA.NS": "Power transmission equipment", "KEC.NS": "Transmission EPC",
        "KPIL.NS": "Transmission EPC", "POLYCAB.NS": "Power cables",
    },
    "AI Data Centre Infrastructure": {
        "NETWEB.NS": "AI servers, HPC & storage", "ANANTRAJ.NS": "Data-centre operator/developer",
        "ABB.NS": "Electrical distribution & automation", "SIEMENS.NS": "Electrification & automation",
        "CUMMINSIND.NS": "Backup power systems", "BLUESTARCO.NS": "Precision cooling & HVAC",
        "VOLTAS.NS": "Cooling & HVAC", "POLYCAB.NS": "Power and data cables",
        "KEI.NS": "Power cables", "TECHM.NS": "Cloud & digital services (indirect)",
    },
    "Battery Energy Storage (BESS) & Power Electronics": {
        "TATAPOWER.NS": "Renewable power & storage", "JSWENERGY.NS": "Utility-scale energy storage",
        "EXIDEIND.NS": "Battery cells & packs", "AMARAJABAT.NS": "Battery cells & energy storage",
        "WAAREEENER.NS": "Solar modules & storage integration", "ABB.NS": "Power conversion & automation",
        "SIEMENS.NS": "Grid integration & power electronics", "CGPOWER.NS": "Electrical equipment",
    },
    "Electronics Components / EMS / Semiconductor Ecosystem": {
        "DIXON.NS": "Electronics manufacturing services", "KAYNES.NS": "EMS & semiconductor packaging",
        "SYRMA.NS": "Electronics manufacturing services", "AMBER.NS": "Electronics & components",
        "PGEL.NS": "Electronics manufacturing services", "BEL.NS": "Defence electronics",
        "NETWEB.NS": "Computing systems manufacturing", "MOSCHIP.NS": "Semiconductor design",
    },
    "Defence Indigenisation & Component Suppliers": {
        "HAL.NS": "Military aircraft & aerospace", "BEL.NS": "Defence electronics & radar",
        "BDL.NS": "Missile systems", "MAZDOCK.NS": "Warships & submarines",
        "COCHINSHIP.NS": "Naval shipbuilding", "GRSE.NS": "Warships & naval vessels",
        "DATAPATTNS.NS": "Defence electronics", "PARAS.NS": "Defence engineering & optics",
    },
    "Grain / Flexible-feed Ethanol": {
        "BALRAMCHIN.NS": "Sugar & ethanol producer", "TRIVENI.NS": "Sugar & multi-feed ethanol",
        "GLOBUSSPR.NS": "Grain-based ethanol", "RENUKA.NS": "Sugar & ethanol producer",
        "EIDPARRY.NS": "Sugar & distillery", "BAJAJHIND.NS": "Sugar & ethanol producer",
    },
}

def _vm_policy_rows():
    """Transparent policy evidence. Scores are anchored to quantified official targets/capex."""
    return [
        {
            "Theme": "Power Grid, Transformers & Transmission",
            "Old Value Pool": "Generation-led power capex",
            "New Value Pool": "Grid expansion, HV equipment, transmission, automation",
            "Primary Catalysts": "RE integration + EV load + data centres + manufacturing electrification",
            "Bottleneck / Picks & Shovels": "Transformers, switchgear, HVDC, conductors, cables, substations, grid automation",
            "Policy Evidence": "National Electricity Plan: transmission network ~4.98 lakh ckm (Nov-2025) → ~6.48 lakh ckm by 2032; transformation capacity 1,398 → 2,345 GVA; plan cost ~₹9.16 lakh Cr.",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2215187&lang=1&reg=1",
            "Policy / Capex": 9.9, "5-6Y Runway": 9.7, "Bottleneck Intensity": 9.8,
            "Early-stage Score": 8.3, "Stage": "Early-to-Mid",
            "Key Risk": "Rich valuations, execution delays, commodity costs, tender competition",
        },
        {
            "Theme": "AI Data Centre Infrastructure",
            "Old Value Pool": "Traditional enterprise IT / servers",
            "New Value Pool": "AI compute + data centres + electrical/cooling infrastructure",
            "Primary Catalysts": "AI adoption, cloud growth, sovereign data, hyperscaler capex",
            "Bottleneck / Picks & Shovels": "Power distribution, transformers, UPS, cooling, cables, backup power, racks",
            "Policy Evidence": "India data-centre buildout is primarily private-capex led; dashboard therefore gives this theme a lower policy weight and relies more on live market confirmation.",
            "Official Source": "https://www.meity.gov.in/",
            "Policy / Capex": 7.8, "5-6Y Runway": 9.8, "Bottleneck Intensity": 9.7,
            "Early-stage Score": 9.0, "Stage": "Early",
            "Key Risk": "Capex concentration, power/water constraints, technology changes, rich valuations",
        },
        {
            "Theme": "Battery Energy Storage (BESS) & Power Electronics",
            "Old Value Pool": "Renewable generation without storage",
            "New Value Pool": "Renewables + storage + dispatchable clean power",
            "Primary Catalysts": "Variable solar/wind, peak demand, grid balancing, storage tenders",
            "Bottleneck / Picks & Shovels": "Cells/packs, BMS, PCS/inverters, EMS, grid integration, storage EPC",
            "Policy Evidence": "NEP projects BESS requirement of 47.24 GW / 236 GWh by 2031-32 with estimated investment ~₹3.49 lakh Cr; VGF schemes target large storage additions.",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2290015&lang=1&reg=3",
            "Policy / Capex": 9.8, "5-6Y Runway": 9.8, "Bottleneck Intensity": 9.5,
            "Early-stage Score": 9.3, "Stage": "Early",
            "Key Risk": "Battery price compression, imports, chemistry shifts, tender economics",
        },
        {
            "Theme": "Electronics Components / EMS / Semiconductor Ecosystem",
            "Old Value Pool": "Imported electronics and low-value assembly",
            "New Value Pool": "Indian components, high-value EMS, packaging and power electronics",
            "Primary Catalysts": "China+1, localisation, ECMS, export manufacturing",
            "Bottleneck / Picks & Shovels": "PCB/PCBA, components, enclosures, connectors, OSAT/ATMP, capital equipment",
            "Policy Evidence": "ECMS approvals: 46 proposals had ~₹54,567 Cr projected investment; a further 29 proposals added ~₹7,104 Cr in Mar-2026.",
            "Official Source": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2247040&lang=1&reg=3",
            "Policy / Capex": 9.7, "5-6Y Runway": 9.5, "Bottleneck Intensity": 8.9,
            "Early-stage Score": 8.7, "Stage": "Early-to-Mid",
            "Key Risk": "Customer concentration, fast tech cycles, valuation, import dependence",
        },
        {
            "Theme": "Defence Indigenisation & Component Suppliers",
            "Old Value Pool": "Imported defence platforms/components",
            "New Value Pool": "Domestic platforms, electronics, precision components and exports",
            "Primary Catalysts": "Indigenisation lists, domestic procurement, export push",
            "Bottleneck / Picks & Shovels": "Radar/electronics, propulsion parts, precision engineering, drones, defence materials",
            "Policy Evidence": "Structural policy support remains strong, but many listed defence names have already re-rated; live market breadth is used to detect whether leadership is still broadening.",
            "Official Source": "https://www.mod.gov.in/",
            "Policy / Capex": 9.5, "5-6Y Runway": 9.0, "Bottleneck Intensity": 8.7,
            "Early-stage Score": 6.8, "Stage": "Mid",
            "Key Risk": "Valuation, order timing, customer concentration, execution",
        },
        {
            "Theme": "Grain / Flexible-feed Ethanol",
            "Old Value Pool": "Pure petrol + sugar-heavy ethanol feedstock",
            "New Value Pool": "E20 + grain/flexible-feed ethanol supply",
            "Primary Catalysts": "E20 demand, flexible feedstock, OMC procurement",
            "Bottleneck / Picks & Shovels": "Flexible-feed distilleries, grain logistics, enzymes, DDGS integration",
            "Policy Evidence": "E20 is already a mature policy milestone; upside now depends more on feedstock economics, utilisation and company-level margins than on a new blending step-up.",
            "Official Source": "https://mopng.gov.in/",
            "Policy / Capex": 7.8, "5-6Y Runway": 6.5, "Bottleneck Intensity": 6.6,
            "Early-stage Score": 5.2, "Stage": "Mid-to-Late",
            "Key Risk": "E20 ceiling, oversupply, feedstock prices, policy allocation",
        },
    ]

@st.cache_data(ttl=1800, show_spinner=False)
def live_value_migration_market_signals():
    """Live market confirmation from representative NSE baskets (Yahoo Finance)."""
    rows = []
    for theme, tickers in VALUE_MIGRATION_BASKETS.items():
        rets = {"1M": [], "3M": [], "6M": [], "12M": []}
        available = 0
        positive6 = 0
        try:
            raw = yf.download(
                tickers=tickers, period="15mo", interval="1d",
                group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
        except Exception:
            raw = pd.DataFrame()

        for ticker in tickers:
            try:
                if raw.empty:
                    continue
                if len(tickers) == 1:
                    px = raw["Close"]
                else:
                    px = raw[ticker]["Close"]
                px = pd.to_numeric(px, errors="coerce").dropna()
                if len(px) < 40:
                    continue
                available += 1
                last = float(px.iloc[-1])
                def ret(days):
                    if len(px) <= days:
                        return np.nan
                    return (last / float(px.iloc[-days-1]) - 1) * 100
                r1, r3, r6, r12 = ret(21), ret(63), ret(126), ret(252)
                for k, v in [("1M",r1),("3M",r3),("6M",r6),("12M",r12)]:
                    if pd.notna(v):
                        rets[k].append(v)
                if pd.notna(r6) and r6 > 0:
                    positive6 += 1
            except Exception:
                continue

        med = {k: (float(np.nanmedian(v)) if v else np.nan) for k,v in rets.items()}
        breadth = 100 * positive6 / available if available else np.nan

        # Live score rewards medium-term trend and breadth, while capping extreme moves.
        parts = []
        if pd.notna(med["3M"]): parts.append(5 + 5*np.tanh(med["3M"]/20))
        if pd.notna(med["6M"]): parts.append(5 + 5*np.tanh(med["6M"]/35))
        if pd.notna(med["12M"]): parts.append(5 + 5*np.tanh(med["12M"]/60))
        if pd.notna(breadth): parts.append(np.clip(breadth/10,0,10))
        live_score = float(np.mean(parts)) if parts else np.nan

        rows.append({
            "Theme": theme, "Basket Stocks": available,
            "Median 1M %": med["1M"], "Median 3M %": med["3M"],
            "Median 6M %": med["6M"], "Median 12M %": med["12M"],
            "6M Positive Breadth %": breadth, "Live Market Score": live_score,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=False)
def value_migration_themes():
    """Policy + structural runway + LIVE market confirmation."""
    df = pd.DataFrame(_vm_policy_rows())
    live = live_value_migration_market_signals()
    df = df.merge(live, on="Theme", how="left")

    # Live score carries 30% of the final result; if unavailable, use neutral 5/10.
    live_component = df["Live Market Score"].fillna(5.0)
    df["Value Migration Score"] = (
        0.25*df["Policy / Capex"] +
        0.20*df["5-6Y Runway"] +
        0.15*df["Bottleneck Intensity"] +
        0.10*df["Early-stage Score"] +
        0.30*live_component
    ) * 10

    df["Rank"] = df["Value Migration Score"].rank(method="first", ascending=False).astype(int)
    return df.sort_values("Value Migration Score", ascending=False).reset_index(drop=True)

def _theme_industry_match(theme, industry):
    s = str(industry).lower()
    mapping = {
        "Power Grid, Transformers & Transmission": ["power", "electric", "electrical", "cable", "transform", "transmission", "capital goods", "engineering"],
        "AI Data Centre Infrastructure": ["software", "it ", "telecom", "electrical", "power", "cooling", "air condition", "capital goods", "engineering"],
        "Battery Energy Storage (BESS) & Power Electronics": ["battery", "storage", "power", "electrical", "renewable", "energy", "electronics"],
        "Electronics Components / EMS / Semiconductor Ecosystem": ["electronics", "electronic", "semiconductor", "telecom", "consumer durables", "capital goods"],
        "Defence Indigenisation & Component Suppliers": ["defence", "aerospace", "engineering", "electronics", "ship", "capital goods"],
        "Grain / Flexible-feed Ethanol": ["sugar", "distiller", "alcohol", "beverage", "agri", "food products"],
    }
    return any(k in s for k in mapping.get(theme, []))

def value_migration_candidate_score(theme_score, live_score, breadth, stock_score=np.nan,
                                    fundamental_score=np.nan):
    """Availability-aware candidate heuristic; inputs are normalized to 0–100."""
    components = [(theme_score, 0.30), (live_score * 10, 0.15), (breadth, 0.10)]
    if pd.notna(stock_score):
        components.append((stock_score * 10, 0.25))
    if pd.notna(fundamental_score):
        components.append((fundamental_score, 0.20))
    available = [(float(v), w) for v, w in components if pd.notna(v)]
    if not available:
        return np.nan, 0
    weight = sum(w for _, w in available)
    return float(np.clip(sum(v * w for v, w in available) / weight, 0, 100)), len(available)

def render_value_migration_page():
    st.title("🚀 Value Migration → Real-Data Multibagger Hunting Engine")
    st.caption(
        "Final theme score now combines quantified policy/capex evidence with LIVE NSE basket momentum and breadth. "
        "A high score identifies a strong migration setup; it does not predict or guarantee 10x/40x returns."
    )

    if st.button("🔄 Refresh Live Theme Data", use_container_width=True):
        live_value_migration_market_signals.clear()
        value_migration_themes.clear()
        st.rerun()

    vm = value_migration_themes()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Top Theme", vm.iloc[0]["Theme"])
    c2.metric("Top VM Score", f"{vm.iloc[0]['Value Migration Score']:.0f}/100")
    c3.metric("Themes Tracked", len(vm))
    c4.metric("Live Data TTL", "30 min")

    st.markdown("### 1️⃣ Real-Data Value Migration Ranking")
    st.caption("किसी भी Theme की row पर click करें; नीचे के सभी sections और stock table उसी theme के अनुसार update होंगे।")
    display = vm[[
        "Rank","Theme","Stage","Policy / Capex","5-6Y Runway",
        "Live Market Score","Median 6M %","6M Positive Breadth %",
        "Value Migration Score","Key Risk"
    ]].copy()
    ranking_event = st.dataframe(
        display, use_container_width=True, hide_index=True,
        key="vm_theme_ranking",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Policy / Capex": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            "5-6Y Runway": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            "Live Market Score": st.column_config.ProgressColumn(min_value=0,max_value=10,format="%.1f"),
            "6M Positive Breadth %": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f%%"),
            "Value Migration Score": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
        }
    )

    theme_options = vm["Theme"].tolist()
    if "vm_selected_theme" not in st.session_state or st.session_state["vm_selected_theme"] not in theme_options:
        st.session_state["vm_selected_theme"] = theme_options[0]
    selected_rows = ranking_event.selection.rows
    if selected_rows:
        st.session_state["vm_selected_theme"] = str(display.iloc[selected_rows[0]]["Theme"])
    selected_theme = st.session_state["vm_selected_theme"]

    fig = go.Figure(go.Bar(
        x=vm["Value Migration Score"], y=vm["Theme"], orientation="h",
        text=vm["Value Migration Score"].round(0), textposition="auto"
    ))
    fig.update_layout(
        title="Policy + Runway + Live Market Confirmation",
        xaxis_title="Dynamic Score / 100", yaxis_title="", height=430,
        yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=50,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2️⃣ Migration Chain + Live Confirmation")
    st.info(f"**Selected Theme:** {selected_theme}")
    r = vm[vm["Theme"]==selected_theme].iloc[0]

    a,b,c,d = st.columns(4)
    a.metric("Policy / Capex", f"{r['Policy / Capex']:.1f}/10")
    b.metric("5–6Y Runway", f"{r['5-6Y Runway']:.1f}/10")
    c.metric("Live Market", "N/A" if pd.isna(r["Live Market Score"]) else f"{r['Live Market Score']:.1f}/10")
    d.metric("6M Breadth", "N/A" if pd.isna(r["6M Positive Breadth %"]) else f"{r['6M Positive Breadth %']:.0f}%")

    st.info(
        f"**OLD VALUE POOL** → {r['Old Value Pool']}\n\n"
        f"**CATALYST** → {r['Primary Catalysts']}\n\n"
        f"**NEW VALUE POOL** → {r['New Value Pool']}\n\n"
        f"**BOTTLENECK / PICKS & SHOVELS** → {r['Bottleneck / Picks & Shovels']}"
    )
    st.markdown(f"**Policy evidence:** {r['Policy Evidence']}")
    st.link_button("🏛️ Open Official Source", r["Official Source"], use_container_width=True)
    st.warning(f"Key risk: {r['Key Risk']}")

    st.markdown("### 3️⃣ Live Theme Market Diagnostics")
    diag_cols = ["Theme","Basket Stocks","Median 1M %","Median 3M %","Median 6M %","Median 12M %","6M Positive Breadth %","Live Market Score"]
    st.dataframe(vm[diag_cols], use_container_width=True, hide_index=True)

    st.markdown("### 4️⃣ What Can Create a 10x/40x Candidate?")
    st.markdown("""
**Sector migration alone is not enough.** Prefer companies where several conditions overlap:
- Addressable opportunity is large relative to the company's current revenue/market-cap base.
- Capacity/order book can multiply revenue over several years.
- Earnings growth is sustained and ROCE stays healthy/improves.
- Debt and cash-flow remain manageable.
- The company supplies a genuine bottleneck / picks-and-shovels product.
- Valuation still leaves room for earnings-led compounding.
- The theme has broad market confirmation, not only one speculative stock.
- Price confirmation: **monthly close above an ≥26-month-old ATH**.
""")

    st.markdown("### 5️⃣ Candidate Discovery — NIFTY 500")
    st.caption("Candidate classification is then linked to your latest 26M breakout/fundamental scan.")
    try:
        uni = load_nifty500()
        theme_tickers = set(VALUE_MIGRATION_BASKETS.get(selected_theme, []))
        cand = uni[uni["Ticker"].isin(theme_tickers)].copy()
        cand = cand.rename(columns={"Company Name":"Company"})
        cand["Theme Role"] = cand["Ticker"].map(VALUE_MIGRATION_ROLES.get(selected_theme, {})).fillna("Theme beneficiary")
        st.metric("Theme-linked NIFTY 500 universe", len(cand))

        latest_ranked = st.session_state.get("latest_ranked", pd.DataFrame())
        if not latest_ranked.empty:
            cols = [c for c in ["Symbol","Status","Sector Score","Policy Score","Fundamental Score %","Pro Final Score"] if c in latest_ranked.columns]
            joined = cand.merge(latest_ranked[cols], on="Symbol", how="left")
            joined["Migration Theme Score"] = float(r["Value Migration Score"])
            scored = joined.apply(
                lambda x: value_migration_candidate_score(
                    r["Value Migration Score"], r["Live Market Score"],
                    r["6M Positive Breadth %"], x.get("Pro Final Score", np.nan),
                    x.get("Fundamental Score %", np.nan)), axis=1
            )
            joined["Candidate Heuristic %"] = scored.apply(lambda x: x[0])
            joined["Inputs Available"] = scored.apply(lambda x: f"{x[1]}/5")
            joined = joined.sort_values(["Candidate Heuristic %","Pro Final Score"], ascending=False, na_position="last")
            st.success("Latest main-dashboard scan is linked to this Value Migration page.")
            st.dataframe(joined, use_container_width=True, hide_index=True)
        else:
            st.dataframe(cand[["Symbol","Company","Industry","Theme Role"]], use_container_width=True, hide_index=True)
            st.info("Run the Main Dashboard scan once; then return here to merge 26M breakout + fundamentals into the migration ranking.")
    except Exception as e:
        st.warning(f"Candidate universe unavailable right now: {e}")

    st.markdown("### 6️⃣ Dynamic Scoring Formula")
    st.code(
        "Value Migration Score = 25% Policy/Capex + 20% 5–6Y Runway + "
        "15% Bottleneck + 10% Early-stage + 30% LIVE Market Confirmation"
    )
    st.caption(
        "Candidate Heuristic = 30% theme + 15% live basket confirmation + 10% breadth + "
        "25% stock confirmation + 20% fundamentals. Missing stock/fundamental inputs are excluded "
        "and the remaining weights are normalized; Inputs Available makes that coverage explicit."
    )
    st.success(
        "MACRO CHANGE → POLICY/CAPEX → VALUE MIGRATION → BOTTLENECK → LIVE MARKET BREADTH → "
        "SMALL/MID-CAP BENEFICIARY → FUNDAMENTALS → VALUATION → 26M ATH BREAKOUT → WATCHLIST"
    )
    st.caption("Research tool only. Live market data comes from Yahoo Finance and can be temporarily unavailable. Official policy links are included for verification.")


render_value_migration_page()
