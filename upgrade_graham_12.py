from pathlib import Path

APP = Path("app.py")
GRAHAM = Path("graham.py")


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly 1 match, found {count}")
    return text.replace(old, new, 1)


def upgrade_graham_py():
    text = GRAHAM.read_text(encoding="utf-8")

    text = text.replace("calculate the ten Graham criteria", "calculate the twelve Graham criteria")

    anchor = '''def fetch_graham_data(ticker):\n    \"\"\"Fetch live/reported inputs and calculate the twelve Graham criteria.\"\"\"\n'''
    if anchor not in text:
        # support the pre-upgrade wording
        anchor = '''def fetch_graham_data(ticker):\n    \"\"\"Fetch live/reported inputs and calculate the ten Graham criteria.\"\"\"\n'''

    helper = '''def _historical_eps_growth_rate(eps):\n    \"\"\"Annualised EPS growth proxy from Graham's first/last 3-year averages.\n\n    Requires ten consecutive annual EPS observations.  The growth interval is\n    measured between the average years of the two 3-year windows (normally\n    seven years).  This is a historical proxy for g, not an analyst forecast.\n    \"\"\"\n    if not isinstance(eps, pd.Series) or eps.empty:\n        return np.nan\n    values = pd.to_numeric(eps, errors=\"coerce\").dropna()\n    values.index = pd.to_datetime(values.index, errors=\"coerce\")\n    values = values[values.index.notna()].groupby(values.index.year).last().sort_index()\n    if len(values) < 10:\n        return np.nan\n    last = values.iloc[-10:]\n    if list(last.index) != list(range(int(last.index[-1]) - 9, int(last.index[-1]) + 1)):\n        return np.nan\n    first_avg = float(last.iloc[:3].mean())\n    last_avg = float(last.iloc[-3:].mean())\n    first_year = float(np.mean(last.index[:3]))\n    last_year = float(np.mean(last.index[-3:]))\n    years = last_year - first_year\n    if first_avg <= 0 or last_avg <= 0 or years <= 0:\n        return np.nan\n    return (last_avg / first_avg) ** (1.0 / years) - 1.0\n\n\ndef _fred_aaa_yield(session=requests):\n    \"\"\"Latest Moody's Seasoned Aaa Corporate Bond Yield from FRED (percent).\"\"\"\n    url = \"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DAAA\"\n    response = session.get(url, headers={\"User-Agent\": \"Mozilla/5.0\"}, timeout=15)\n    response.raise_for_status()\n    frame = pd.read_csv(pd.io.common.StringIO(response.text))\n    if \"DAAA\" not in frame.columns or \"DATE\" not in frame.columns:\n        return np.nan, None\n    values = pd.to_numeric(frame[\"DAAA\"], errors=\"coerce\")\n    valid = frame.loc[values.notna(), [\"DATE\"]].copy()\n    if valid.empty:\n        return np.nan, None\n    idx = values.dropna().index[-1]\n    return float(values.loc[idx]), str(frame.loc[idx, \"DATE\"])\n\n\n'''
    text = replace_once(text, anchor, helper + anchor.replace("ten Graham criteria", "twelve Graham criteria"), "insert helpers")

    old = '''    positive10, growth10, eps_years = _ten_year_tests(eps_for_history)\n    eps_for_average = eps_history if len(eps_history) >= 3 else eps_statement\n    avg3_eps = float(eps_for_average.iloc[-3:].mean()) if len(eps_for_average) >= 3 else np.nan\n'''
    new = '''    positive10, growth10, eps_years = _ten_year_tests(eps_for_history)\n    eps_growth_rate = _historical_eps_growth_rate(eps_for_history)\n    eps_for_average = eps_history if len(eps_history) >= 3 else eps_statement\n    avg3_eps = float(eps_for_average.iloc[-3:].mean()) if len(eps_for_average) >= 3 else np.nan\n    latest_eps = _latest(eps_for_history)\n    if pd.isna(latest_eps):\n        latest_eps = number(info.get(\"trailingEps\"))\n'''
    text = replace_once(text, old, new, "EPS growth inputs")

    old = '''    combined = pe3 * pb if pd.notna(pe3) and pd.notna(pb) else np.nan\n    graham_number = np.sqrt(22.5 * avg3_eps * bvps) if pd.notna(avg3_eps) and avg3_eps > 0 and pd.notna(bvps) and bvps > 0 else np.nan\n\n    return {\n'''
    new = '''    combined = pe3 * pb if pd.notna(pe3) and pd.notna(pb) else np.nan\n    graham_number = np.sqrt(22.5 * avg3_eps * bvps) if pd.notna(avg3_eps) and avg3_eps > 0 and pd.notna(bvps) and bvps > 0 else np.nan\n    margin_of_safety = ((graham_number - price) / graham_number * 100.0\n                        if pd.notna(graham_number) and graham_number > 0 and pd.notna(price) else np.nan)\n\n    try:\n        aaa_yield, aaa_yield_date = _fred_aaa_yield()\n    except Exception:\n        aaa_yield, aaa_yield_date = np.nan, None\n\n    growth_rate_pct = eps_growth_rate * 100.0 if pd.notna(eps_growth_rate) else np.nan\n    growth_value = (latest_eps * (8.5 + 2.0 * growth_rate_pct) * 4.4 / aaa_yield\n                    if pd.notna(latest_eps) and latest_eps > 0\n                    and pd.notna(growth_rate_pct) and pd.notna(aaa_yield) and aaa_yield > 0 else np.nan)\n\n    return {\n'''
    text = replace_once(text, old, new, "valuation calculations")

    old = '''        \"pb\": pb, \"combined\": combined, \"graham_no\": graham_number, \"eps_years\": eps_years,\n        \"as_of\": datetime.now(timezone.utc).strftime(\"%Y-%m-%d %H:%M UTC\"),\n        \"sources\": \"Yahoo Finance live quote, annual financial statements, fundamentals time-series and dividend events\",\n'''
    new = '''        \"pb\": pb, \"combined\": combined, \"graham_no\": graham_number, \"eps_years\": eps_years,\n        \"margin_of_safety\": margin_of_safety, \"growth_value\": growth_value,\n        \"growth_rate_pct\": growth_rate_pct, \"growth_eps\": latest_eps,\n        \"aaa_yield\": aaa_yield, \"aaa_yield_date\": aaa_yield_date,\n        \"as_of\": datetime.now(timezone.utc).strftime(\"%Y-%m-%d %H:%M UTC\"),\n        \"sources\": \"Yahoo Finance live quote, annual financial statements, fundamentals time-series and dividend events; FRED DAAA for Moody's Seasoned Aaa Corporate Bond Yield\",\n'''
    text = replace_once(text, old, new, "return fields")

    GRAHAM.write_text(text, encoding="utf-8")


def upgrade_app_py():
    text = APP.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'st.caption("Ten Graham tests calculated from Yahoo Finance live quotes, reported annual statements, fundamentals history and dividend events. Missing observations remain N/A.")',
        'st.caption("Twelve-point Graham scorecard using Yahoo Finance reported fundamentals plus the live Moody\'s Aaa corporate bond yield from FRED for the growth formula. Missing observations remain N/A.")',
        "Graham page caption",
    )

    old = '''        (\"10\", \"Graham Number\", \"√(22.5 × 3-year avg EPS × BVPS) > CMP\", d[\"graham_no\"],\n         None if pd.isna(d[\"graham_no\"]) or pd.isna(d[\"price\"]) else d[\"graham_no\"] > d[\"price\"], \"₹\"),\n    ]\n'''
    new = '''        (\"10\", \"Graham Number\", \"√(22.5 × 3-year avg EPS × BVPS) > CMP\", d[\"graham_no\"],\n         None if pd.isna(d[\"graham_no\"]) or pd.isna(d[\"price\"]) else d[\"graham_no\"] > d[\"price\"], \"₹\"),\n        (\"11\", \"Margin of Safety\", \"(Graham No. - CMP) / Graham No. > 0\", d[\"margin_of_safety\"],\n         None if pd.isna(d[\"margin_of_safety\"]) else d[\"margin_of_safety\"] > 0, \"%\"),\n        (\"12\", \"Growth Formula\", \"[EPS × (8.5 + 2g) × 4.4] / Y > CMP\", d[\"growth_value\"],\n         None if pd.isna(d[\"growth_value\"]) or pd.isna(d[\"price\"]) else d[\"growth_value\"] > d[\"price\"], \"₹\"),\n    ]\n'''
    text = replace_once(text, old, new, "add criteria 11 and 12")

    old = '''        elif unit == \"₹\":\n            display = f\"₹{value:,.2f}\"\n        else:\n            display = f\"{value:.2f}x\"\n'''
    new = '''        elif unit == \"₹\":\n            display = f\"₹{value:,.2f}\"\n        elif unit == \"%\":\n            display = f\"{value:,.2f}%\"\n        else:\n            display = f\"{value:.2f}x\"\n'''
    text = replace_once(text, old, new, "percent display")

    text = replace_once(text, 'm4.metric("Unavailable", f"{10 - assessed}/10")', 'm4.metric("Unavailable", f"{12 - assessed}/12")', "12 point denominator")

    old = '''    st.info("Sales, debt and working capital are displayed in ₹ crore (₹1 crore = ₹10,000,000); EPS, CMP and BVPS are per share in rupees. The current calendar year is excluded from the 20-year dividend test.")\n'''
    new = '''    st.info("Sales, debt and working capital are displayed in ₹ crore (₹1 crore = ₹10,000,000); EPS, CMP and BVPS are per share in rupees. The current calendar year is excluded from the 20-year dividend test.")\n    growth_note = \"N/A\" if pd.isna(d.get(\"growth_rate_pct\", np.nan)) else f\"{d['growth_rate_pct']:.2f}%\"\n    yield_note = \"N/A\" if pd.isna(d.get(\"aaa_yield\", np.nan)) else f\"{d['aaa_yield']:.2f}%\"\n    yield_date = d.get(\"aaa_yield_date\") or \"N/A\"\n    st.caption(f\"Growth Formula inputs: latest reported EPS = {'N/A' if pd.isna(d.get('growth_eps', np.nan)) else f'₹{d[\\\"growth_eps\\\"]:.2f}'}; g = {growth_note} historical EPS growth proxy from the 10-year first/last 3-year averages; Y = {yield_note} Moody's Seasoned Aaa Corporate Bond Yield (FRED DAAA, {yield_date}).\")\n'''
    text = replace_once(text, old, new, "growth input note")

    APP.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    upgrade_graham_py()
    upgrade_app_py()
    print("Graham scorecard upgraded to 12 points.")
