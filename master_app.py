import streamlit as st

st.set_page_config(
    page_title="My Dashboard Hub",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAINTENANCE_URL = "https://aqpl-maintenance-dashboard.streamlit.app/"
NATUROPATHY_URL = "https://natural-health-knowledge-dashboard-c79tpzz94uakfktnvt3gga.streamlit.app/"
INVESTMENT_URL = "https://macro-breakout-dashboard-ub3hdzxk9bjj5wrbexmmw8.streamlit.app/"

st.markdown(
    """
    <style>
        .block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 2rem;}
        .hub-title {text-align:center; font-size:2.2rem; font-weight:800; margin-bottom:.2rem;}
        .hub-subtitle {text-align:center; color:#777; margin-bottom:2rem;}
        .dashboard-card {border:1px solid rgba(128,128,128,.25); border-radius:18px; padding:22px; min-height:205px; box-shadow:0 4px 14px rgba(0,0,0,.06); margin-bottom:1rem;}
        .dashboard-icon {font-size:2.2rem; margin-bottom:.4rem;}
        .dashboard-title {font-size:1.3rem; font-weight:750; margin-bottom:.4rem;}
        .dashboard-desc {color:#777; min-height:52px; margin-bottom:.8rem;}
        div[data-testid="stLinkButton"] > a {width:100%; justify-content:center; border-radius:10px; font-weight:650;}
        @media (max-width: 700px) {.hub-title{font-size:1.8rem;} .dashboard-card{min-height:auto;}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hub-title">🏠 My Dashboard Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="hub-subtitle">Maintenance • Naturopathy • Investment — sabhi dashboards ek jagah</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('''<div class="dashboard-card"><div class="dashboard-icon">🔧</div><div class="dashboard-title">Maintenance Dashboard</div><div class="dashboard-desc">PM, breakdown, machine history, check sheets aur maintenance records.</div></div>''', unsafe_allow_html=True)
    st.link_button("Open Maintenance", MAINTENANCE_URL, use_container_width=True)

with c2:
    st.markdown('''<div class="dashboard-card"><div class="dashboard-icon">🌿</div><div class="dashboard-title">Naturopathy Dashboard</div><div class="dashboard-desc">Natural health knowledge, topics, notes aur searchable learning records.</div></div>''', unsafe_allow_html=True)
    st.link_button("Open Naturopathy", NATUROPATHY_URL, use_container_width=True)

with c3:
    st.markdown('''<div class="dashboard-card"><div class="dashboard-icon">📈</div><div class="dashboard-title">Investment Dashboard</div><div class="dashboard-desc">Macro, sector rotation, value migration, breakouts aur stock research.</div></div>''', unsafe_allow_html=True)
    st.link_button("Open Investment", INVESTMENT_URL, use_container_width=True)

st.divider()
st.caption("Tip: Is Master Dashboard ko mobile Home Screen par add kar den, phir isi ek icon se tino dashboards open honge.")
