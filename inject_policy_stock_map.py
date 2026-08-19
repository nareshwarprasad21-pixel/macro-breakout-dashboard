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

    import_line = 'from policy_stock_map import render_policy_stock_mapping\n'
    if import_line not in text:
        for anchor in ('from graham import fetch_graham_data\n', 'import yfinance as yf\n'):
            if anchor in text:
                text = text.replace(anchor, anchor + import_line, 1)
                break

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

print('Policy stock mapping integration ready:', ', '.join(changed) if changed else 'already integrated')
