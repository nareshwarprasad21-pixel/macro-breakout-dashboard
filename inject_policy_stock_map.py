from pathlib import Path

p = Path('app.py')
text = p.read_text(encoding='utf-8')

import_line = 'from policy_stock_map import render_policy_stock_mapping\n'
if import_line not in text:
    anchor = 'from graham import fetch_graham_data\n'
    if anchor not in text:
        raise RuntimeError('Import anchor not found in app.py')
    text = text.replace(anchor, anchor + import_line, 1)

call = '    render_policy_stock_mapping(policy_df)\n'
if call not in text:
    anchor = '    st.dataframe(top5, use_container_width=True, hide_index=True)\n\n'
    if anchor not in text:
        raise RuntimeError('Policy report anchor not found in app.py')
    text = text.replace(
        anchor,
        anchor + '    st.markdown("---")\n' + call + '\n',
        1,
    )

p.write_text(text, encoding='utf-8')
print('Policy stock mapping integration ready')
