import streamlit as st


def apply_professional_ui():
    """UI-only polish. Does not change calculations, filters, scans, or trading logic."""
    st.markdown(
        r"""
<style>
:root {
  --panel: rgba(15, 23, 42, .72);
  --panel-2: rgba(20, 30, 52, .88);
  --line: rgba(148, 163, 184, .16);
  --muted: #94a3b8;
  --text: #f8fafc;
  --accent: #7c5cff;
  --accent-2: #9b87ff;
  --good: #22c55e;
  --watch: #eab308;
  --caution: #f59e0b;
  --bad: #ef4444;
}

/* Main canvas */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 85% 0%, rgba(124,92,255,.10), transparent 28%),
    linear-gradient(180deg, #081120 0%, #0a1425 48%, #0b1423 100%);
}
[data-testid="stMainBlockContainer"] {
  max-width: 1500px;
  padding-top: 2.1rem !important;
  padding-bottom: 3rem !important;
}
.block-container { max-width: 1500px; }

/* Typography hierarchy */
h1 { font-size: clamp(2rem, 3.0vw, 3.15rem) !important; line-height: 1.08 !important; letter-spacing: -.03em; }
h2 { font-size: clamp(1.55rem, 2.2vw, 2.15rem) !important; margin-top: 1.35rem !important; }
h3 { font-size: clamp(1.22rem, 1.65vw, 1.55rem) !important; }
p, li, label { line-height: 1.55; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b172b 0%, #0b1424 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1.15rem; }
[data-testid="stSidebar"] button {
  border-radius: 10px !important;
  min-height: 42px;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(19,31,54,.96), rgba(10,20,38,.96));
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px 18px;
  min-height: 132px;
  box-shadow: 0 10px 28px rgba(0,0,0,.18);
  overflow: visible !important;
}
[data-testid="stMetricLabel"] {
  color: #cbd5e1 !important;
  font-weight: 650;
}

/* Long metric values (regime / engine names) must wrap instead of ellipsis */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] p,
[data-testid="stMetricValue"] span,
[data-testid="stMetricValue"] * {
  overflow: visible !important;
  text-overflow: unset !important;
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: anywhere !important;
  max-width: 100% !important;
}
[data-testid="stMetricValue"] {
  display: block !important;
  width: 100% !important;
  font-size: clamp(1.25rem, 1.65vw, 1.85rem) !important;
  letter-spacing: -.025em;
  line-height: 1.12 !important;
}
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] p {
  display: block !important;
  width: 100% !important;
  line-height: 1.12 !important;
}
[data-testid="stMetricDelta"] { font-weight: 700; }

/* Give metric columns room to display long regime/engine labels */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
  min-width: 0;
}
[data-testid="stHorizontalBlock"] [data-testid="stMetric"] {
  width: 100%;
}

/* Tabs: larger, more breathable, clearer active state */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  padding: 6px;
  border: 1px solid var(--line);
  background: rgba(15,23,42,.62);
  border-radius: 14px;
  overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
  height: 46px;
  padding: 0 17px;
  border-radius: 10px;
  color: #aebcd0;
  font-size: .95rem;
  font-weight: 650;
  white-space: nowrap;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(124,92,255,.28), rgba(91,74,196,.18));
  color: #ffffff !important;
  box-shadow: inset 0 -2px 0 var(--accent), 0 6px 18px rgba(124,92,255,.12);
}

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a {
  min-height: 44px;
  border-radius: 11px !important;
  font-weight: 700 !important;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stLinkButton"] a:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(0,0,0,.22);
}
button[kind="primary"] {
  background: linear-gradient(135deg, #7455ff, #8b68ff) !important;
  border: 0 !important;
}

/* Inputs */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="base-input"] {
  border-radius: 11px !important;
  border-color: rgba(148,163,184,.20) !important;
  background: rgba(17,29,50,.82) !important;
}

/* Dataframes/tables */
[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 10px 28px rgba(0,0,0,.14);
}
[data-testid="stDataFrame"] [role="columnheader"] { font-weight: 750 !important; }

/* Expanders */
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 13px !important;
  background: rgba(15,23,42,.44);
  overflow: hidden;
}

/* Info / warning / success / error */
[data-testid="stAlert"] {
  border-radius: 13px;
  border-width: 1px;
  box-shadow: 0 7px 20px rgba(0,0,0,.10);
}

/* Dividers */
hr { border-color: rgba(148,163,184,.15) !important; margin: 1.7rem 0 !important; }

/* Plot containers */
[data-testid="stPlotlyChart"] {
  background: rgba(10,20,38,.42);
  border: 1px solid var(--line);
  border-radius: 15px;
  padding: 5px;
}

/* Captions muted but readable */
[data-testid="stCaptionContainer"], .stCaption { color: #91a0b6 !important; }

/* Mid-size screens: keep long values readable */
@media (max-width: 1250px) {
  [data-testid="stMetricValue"] {
    font-size: 1.22rem !important;
  }
  [data-testid="stMetric"] {
    min-height: 136px;
    padding: 15px 14px;
  }
}

/* Mobile/tablet */
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] { padding: 1.25rem .8rem 2rem !important; }
  [data-testid="stMetric"] { min-height: 118px; padding: 13px 14px; }
  [data-testid="stMetricValue"] { font-size: 1.18rem !important; }
  .stTabs [data-baseweb="tab"] { height: 42px; padding: 0 12px; font-size: .88rem; }
  h1 { font-size: 2rem !important; }
  h2 { font-size: 1.5rem !important; }
}
</style>
""",
        unsafe_allow_html=True,
    )
