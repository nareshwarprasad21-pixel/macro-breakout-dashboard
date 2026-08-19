from pathlib import Path

ENTRY_FILES = [
    'app.py',
    'app_v4_professional.py',
    'app_v5_rotation_map.py',
    'app_upgraded.py',
]


def patch_file(path: Path):
    if not path.exists():
        return False

    text = path.read_text(encoding='utf-8')
    original = text

    # Policy-stock mapping import.
    policy_import = 'from policy_stock_map import render_policy_stock_mapping\n'
    if policy_import not in text:
        for anchor in ('from graham import fetch_graham_data\n', 'import yfinance as yf\n'):
            if anchor in text:
                text = text.replace(anchor, anchor + policy_import, 1)
                break

    # Global UI-only polish import.
    ui_import = 'from ui_polish import apply_professional_ui\n'
    if ui_import not in text:
        for anchor in (policy_import, 'import yfinance as yf\n'):
            if anchor in text:
                text = text.replace(anchor, anchor + ui_import, 1)
                break

    # Apply polish immediately after set_page_config; no analytics are touched.
    ui_call = 'apply_professional_ui()\n'
    if ui_call not in text:
        marker = 'st.set_page_config('
        pos = text.find(marker)
        if pos != -1:
            end = text.find('\n', pos)
            if end != -1:
                text = text[:end+1] + ui_call + text[end+1:]

    # Policy-stock mapping renderer.
    call = '    render_policy_stock_mapping(policy_df)\n'
    if call not in text:
        for anchor in (
            '    st.dataframe(top5, use_container_width=True, hide_index=True)\n\n',
            '    st.dataframe(top5,use_container_width=True,hide_index=True)\n\n',
        ):
            if anchor in text:
                text = text.replace(anchor, anchor + '    st.markdown("---")\n' + call + '\n', 1)
                break

    if text != original:
        path.write_text(text, encoding='utf-8')
        return True
    return False


changed = []
for filename in ENTRY_FILES:
    p = Path(filename)
    try:
        if patch_file(p):
            changed.append(filename)
    except Exception as exc:
        print(f'Warning: {filename}: {exc}')

print('Dashboard integrations ready:', ', '.join(changed) if changed else 'already integrated')
