"""Runtime bootstrap for Streamlit Cloud/Railway.

Keeps the Government Policy stock map integrated even when the host launches
`streamlit run app.py` directly. The patch is idempotent and only touches the
known policy-report anchors.
"""
from pathlib import Path


def _integrate_policy_stock_map():
    app = Path(__file__).with_name("app.py")
    if not app.exists():
        return
    text = app.read_text(encoding="utf-8")
    changed = False

    import_line = "from policy_stock_map import render_policy_stock_mapping\n"
    if import_line not in text:
        anchor = "from graham import fetch_graham_data\n"
        if anchor in text:
            text = text.replace(anchor, anchor + import_line, 1)
            changed = True

    call_line = "    render_policy_stock_mapping(policy_df)\n"
    if call_line not in text:
        anchor = "    st.dataframe(top5, use_container_width=True, hide_index=True)\n\n"
        if anchor in text:
            text = text.replace(
                anchor,
                anchor + "    st.markdown(\"---\")\n" + call_line + "\n",
                1,
            )
            changed = True

    if changed:
        app.write_text(text, encoding="utf-8")


try:
    _integrate_policy_stock_map()
except Exception:
    # Never block application startup because of an optional UI integration.
    pass
