from pathlib import Path

FILES = {
    'pages/1_Professional_Research_Lab.py': [
        ('b.metric("Macro Regime",regime)', 'text_metric(b,"Macro Regime",regime)'),
    ],
    'pages/3_Decision_Engine.py': [
        ("b.metric('Live Regime',regime)", "text_metric(b,'Live Regime',regime)"),
    ],
    'pages/4_VALUE_Stock_Engine.py': [
        ('top2.metric("Macro Regime",macro_regime)', 'text_metric(top2,"Macro Regime",macro_regime)'),
        ('top4.metric("Engine","6-Factor VALUE")', 'text_metric(top4,"Engine","6-Factor VALUE")'),
        ('b.metric("Regime",macro_regime)', 'text_metric(b,"Regime",macro_regime)'),
    ],
}

HELPER = '''\n\ndef text_metric(container, label, value):\n    \"\"\"Responsive text card for long categorical values; avoids Streamlit metric ellipsis.\"\"\"\n    safe_label = str(label).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')\n    safe_value = str(value).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')\n    container.markdown(\n        f\"\"\"\n        <div style=\"min-height:118px;padding:16px 18px;border:1px solid rgba(148,163,184,.18);\n                    border-radius:16px;background:linear-gradient(145deg,rgba(19,31,54,.96),rgba(10,20,38,.96));\n                    box-shadow:0 10px 28px rgba(0,0,0,.16);display:flex;flex-direction:column;justify-content:center;\">\n          <div style=\"color:#cbd5e1;font-size:.86rem;font-weight:650;margin-bottom:9px;\">{safe_label}</div>\n          <div style=\"color:#f8fafc;font-size:clamp(1.18rem,1.55vw,1.75rem);font-weight:500;\n                      line-height:1.16;white-space:normal;overflow-wrap:anywhere;word-break:normal;\">{safe_value}</div>\n        </div>\n        \"\"\",\n        unsafe_allow_html=True,\n    )\n'''

for filename, replacements in FILES.items():
    p = Path(filename)
    if not p.exists():
        print('missing', filename)
        continue
    text = p.read_text(encoding='utf-8')
    original = text

    if 'def text_metric(container, label, value):' not in text:
        # Insert helper immediately after the first set_page_config call.
        marker = 'st.set_page_config'
        pos = text.find(marker)
        if pos >= 0:
            eol = text.find('\n', pos)
            text = text[:eol+1] + HELPER + text[eol+1:]
        else:
            text = HELPER + '\n' + text

    for old, new in replacements:
        text = text.replace(old, new)

    if text != original:
        p.write_text(text, encoding='utf-8')
        print('patched', filename)
    else:
        print('already patched', filename)
