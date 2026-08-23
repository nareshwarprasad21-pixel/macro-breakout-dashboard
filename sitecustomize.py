"""Runtime bootstrap for Streamlit Cloud/Railway.

Keeps optional integrations attached to whichever Streamlit entry file is
deployed. Patches are idempotent and must never block app startup.
"""
from pathlib import Path


def _patch_entry(app: Path):
    if not app.exists():
        return

    text = app.read_text(encoding="utf-8")
    changed = False

    # Government Policy stock-map integration.
    policy_import = "from policy_stock_map import render_policy_stock_mapping\n"
    if policy_import not in text:
        for anchor in ("from graham import fetch_graham_data\n", "import yfinance as yf\n"):
            if anchor in text:
                text = text.replace(anchor, anchor + policy_import, 1)
                changed = True
                break

    policy_call = "    render_policy_stock_mapping(policy_df)\n"
    if policy_call not in text:
        for anchor in (
            "    st.dataframe(top5, use_container_width=True, hide_index=True)\n\n",
            "    st.dataframe(top5,use_container_width=True,hide_index=True)\n\n",
        ):
            if anchor in text:
                text = text.replace(anchor, anchor + "    st.markdown(\"---\")\n" + policy_call + "\n", 1)
                changed = True
                break

    # NIFTY 500 scanner: apply granular sector classification after the official
    # constituent CSV has been loaded and Ticker has been created.
    sector_import = "from scanner_sectors import apply_scanner_sector_classification\n"
    if sector_import not in text:
        anchor = "from ui_polish import apply_professional_ui\n"
        if anchor in text:
            text = text.replace(anchor, anchor + sector_import, 1)
            changed = True
        elif "import yfinance as yf\n" in text:
            text = text.replace("import yfinance as yf\n", "import yfinance as yf\n" + sector_import, 1)
            changed = True

    old_return = '            return df[["Company Name", "Industry", "Symbol", "Ticker"]].drop_duplicates("Ticker")\n'
    new_return = '            return apply_scanner_sector_classification(df[["Company Name", "Industry", "Symbol", "Ticker"]].drop_duplicates("Ticker"))\n'
    if old_return in text and new_return not in text:
        text = text.replace(old_return, new_return, 1)
        changed = True

    if changed:
        app.write_text(text, encoding="utf-8")


def _integrate_runtime_features():
    root = Path(__file__).parent
    for name in (
        "app.py",
        "app_v4_professional.py",
        "app_v5_rotation_map.py",
        "app_upgraded.py",
    ):
        try:
            _patch_entry(root / name)
        except Exception:
            pass


try:
    _integrate_runtime_features()
except Exception:
    # Optional integrations must never block app startup.
    pass
