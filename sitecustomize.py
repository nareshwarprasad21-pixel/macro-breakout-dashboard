"""Runtime bootstrap for Streamlit Cloud/Railway.

Ensures the Government Policy stock-map UI is integrated into whichever
Streamlit entry file is deployed. The patch is idempotent and only touches
known policy-report anchors.
"""
from pathlib import Path


def _patch_entry(app: Path):
    if not app.exists():
        return

    text = app.read_text(encoding="utf-8")
    changed = False

    import_line = "from policy_stock_map import render_policy_stock_mapping\n"
    if import_line not in text:
        import_anchors = [
            "from graham import fetch_graham_data\n",
            "import yfinance as yf\n",
        ]
        for anchor in import_anchors:
            if anchor in text:
                text = text.replace(anchor, anchor + import_line, 1)
                changed = True
                break

    call_line = "    render_policy_stock_mapping(policy_df)\n"
    if call_line not in text:
        call_anchors = [
            "    st.dataframe(top5, use_container_width=True, hide_index=True)\n\n",
            "    st.dataframe(top5,use_container_width=True,hide_index=True)\n\n",
        ]
        for anchor in call_anchors:
            if anchor in text:
                text = text.replace(
                    anchor,
                    anchor + "    st.markdown(\"---\")\n" + call_line + "\n",
                    1,
                )
                changed = True
                break

    if changed:
        app.write_text(text, encoding="utf-8")


def _integrate_policy_stock_map():
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
    _integrate_policy_stock_map()
except Exception:
    # Optional UI integration must never block app startup.
    pass
